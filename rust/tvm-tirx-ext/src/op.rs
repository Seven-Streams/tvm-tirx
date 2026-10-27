/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

//! Registering operators from Rust: C++ `OpDef`, with its signature descriptors ([`sig`]) and
//! the attribute keys of [`ir::op_attr`](tvm_tirx::ir::op_attr) and
//! [`tirx::op_attr`](tvm_tirx::tirx::op_attr); the [`register_ops!`](crate::register_ops) blocks
//! that define operators, as C++ defines them in a `TVM_FFI_STATIC_INIT_BLOCK`; and the handles
//! that keep operators in `static`s.
//!
//! Every hook an operator gets here is native, as the C++ ones are: a Rust function compiled into
//! the C ABI adapter that C++ `NativeFunctionView` calls through a function pointer, without
//! packing arguments. A signature generates its validator from its argument types, as C++
//! `OpDef::signature` does from `sig::arg<T>`; each signature is its own compiled function.

use std::cell::RefCell;
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::OnceLock;

use tvm_ffi::tvm_ffi_sys::{TVMFFIAny, TVMFFIObject};
use tvm_ffi::{
    Any, AnyView, Array, Error, ObjectArc, ObjectRefCore, Result, String, TYPE_ERROR, VALUE_ERROR,
};
use tvm_tirx::ir::{AttrKey, CallObj, Op, RegionStmtObj, Type, Var};

use crate::registry::register_defined_ir;
use crate::sig::{self, Checks, Specs};

/// C++ `OpDef`: register a new operator from Rust.
///
/// Each step updates the registry at once, as in C++. After a failed step the rest are skipped,
/// and [`OpDef::op`] returns the error. The registry does not synchronize registration with
/// readers, so register an operator before building IR with it, as a
/// [`register_ops!`](crate::register_ops) block does. An operator and its hooks live for the rest
/// of the process.
///
/// ```no_run
/// use tvm_tirx_ext::ir::op_attr::CALL_EFFECT_KIND;
/// use tvm_tirx_ext::sig::{self, IntExpr};
/// use tvm_tirx_ext::ir::{CallEffectKind, CallObj, Type};
/// use tvm_tirx_ext::OpDef;
/// use tvm_tirx_ext::tvm_ffi::Result;
///
/// fn first_arg_type(call: &CallObj) -> Result<Type> {
///     Ok(call.args.get(0)?.ty.clone())
/// }
///
/// let op = OpDef::new("my_ext.clamp", "Clamp an index to a bound.")
///     .signature((sig::arg::<IntExpr>("index", ""), sig::arg::<IntExpr>("bound", "")))
///     .set_infer_type(first_arg_type)
///     .set_attr(CALL_EFFECT_KIND, CallEffectKind::Pure)
///     .op()?;
/// # Ok::<(), tvm_tirx_ext::tvm_ffi::Error>(())
/// ```
pub struct OpDef(Result<Op>);

impl OpDef {
    /// Register a new operator (`ir.RegisterOp`), with `doc` as its documentation, less the
    /// indentation its lines share and the blank lines around it, so that a string of several
    /// lines can be indented with the code. A name already registered is an error.
    pub fn new(name: &str, doc: &str) -> Self {
        let registered = tvm_ffi::cached_global_func!("ir.RegisterOp")
            .call_tuple((String::from(name), String::from(doc_text(doc))));
        Self(registered.and_then(|_| Op::get(name)))
    }

    /// The completed operator, or the error of the step that failed. Request its TVMScript
    /// builder through `script.expose_script_op` after its signature and attributes are ready.
    /// An absent receiver is a no-op; an installed receiver's error is returned.
    pub fn op(self) -> Result<Op> {
        let op = self.0?;
        if let Ok(expose) = tvm_ffi::Function::get_global("script.expose_script_op") {
            expose.call_tuple((op.name.clone(),))?;
        }
        Ok(op)
    }

    /// C++ `OpDef::signature`: replace the argument descriptions and the attrs type, and the
    /// validator generated from them, which checks a Call's argument counts, argument types and
    /// attrs. `specs` is a descriptor of [`sig`], or a tuple of them in order, as C++ takes them.
    /// A custom validator keeps precedence. Without [`sig::call_attrs`] the operator has no attrs
    /// type, and `Call.attrs` is unconstrained.
    pub fn signature<S: Specs>(self, specs: S) -> Self {
        let mut description = sig::Description::default();
        specs.describe(&mut description);
        let lists = description.into_lists().map(Array::new);
        let args: Vec<AnyView> = lists.iter().map(AnyView::from).collect();
        let def = self.invoke("_set_signature", &args).then(|op| {
            if !op.validator_is_custom() {
                let hook: Hook = validate_signature::<S::Checks>;
                op.write_validator(Any::from(hook as *mut c_void), false);
            }
            Ok(())
        });
        match <S::Checks as Checks>::ATTRS_TYPE_KEY {
            Some(key) => def.attrs_type_key(key),
            None => def,
        }
    }

    /// Append a value argument's description, without changing what the validator checks.
    pub fn add_arg(self, name: &str, doc: &str) -> Self {
        let (name, doc) = (String::from(name), String::from(doc));
        self.invoke("add_arg", &[AnyView::from(&name), AnyView::from(&doc)])
    }

    /// Append a type argument's description, without changing what the validator checks.
    pub fn add_ty_arg(self, name: &str, doc: &str) -> Self {
        let (name, doc) = (String::from(name), String::from(doc));
        self.invoke("add_ty_arg", &[AnyView::from(&name), AnyView::from(&doc)])
    }

    /// The type of `Call.attrs`, by its registered type key.
    pub fn attrs_type_key(self, key: &str) -> Self {
        let key = String::from(key);
        self.invoke("set_attrs_type_key", &[AnyView::from(&key)])
    }

    /// Register the attribute `key` (C++ `set_attr(key, value)`); a value already registered for
    /// this operator is an error.
    pub fn set_attr<T: Into<Any>>(self, key: AttrKey<T>, value: impl Into<T>) -> Self {
        let value: T = value.into();
        self.attr(key.name(), value.into(), false)
    }

    /// Register the attribute `key`, replacing a value already registered for this operator
    /// (C++ `set_attr(key, value, true)`).
    pub fn replace_attr<T: Into<Any>>(self, key: AttrKey<T>, value: impl Into<T>) -> Self {
        let value: T = value.into();
        self.attr(key.name(), value.into(), true)
    }

    /// Remove this operator's value of the attribute `key`.
    pub fn reset_attr<T>(self, key: AttrKey<T>) -> Self {
        let key = String::from(key.name());
        self.invoke("reset_attr", &[AnyView::from(&key)])
    }

    /// `FInferType`: the result type of a Call that does not give one, which the `Call`
    /// constructor and `Call::reinfer_type` ask for; [`op_attr::FIXED_RETURN_TYPE`](tvm_tirx::ir::op_attr::FIXED_RETURN_TYPE) takes
    /// precedence. `infer` is a function or a closure that captures nothing.
    pub fn set_infer_type<F>(self, infer: F) -> Self
    where
        F: Fn(&CallObj) -> Result<Type> + 'static,
    {
        let _ = infer;
        let hook: Hook = infer_type_abi::<F>;
        self.attr("FInferType", Any::from(hook as *mut c_void), false)
    }

    /// A custom validator, run by `Call::validate`, which reports an invalid Call as an error. It
    /// replaces the one a signature generated, and a later signature keeps it; a second custom
    /// validator is an error. `validate` is a function or a closure that captures nothing.
    pub fn set_validator<F>(self, validate: F) -> Self
    where
        F: Fn(&CallObj) -> Result<()> + 'static,
    {
        let _ = validate;
        self.then(|op| {
            if op.validator_is_custom() {
                let message = format!("Validator of {} is already registered", op.name.as_str());
                return Err(Error::new(VALUE_ERROR, &message, ""));
            }
            let hook: Hook = validate_abi::<F>;
            op.write_validator(Any::from(hook as *mut c_void), true);
            Ok(())
        })
    }

    /// `FRegionGetBodyParams`, which makes the operator a region operation: the variables a
    /// region binds in its body, from its Call (the operation, operands and attributes), which
    /// the `RegionStmt` constructor checks the body parameters against. A region that binds none
    /// returns none. `body_params` is a function or a closure that captures nothing.
    pub fn set_region_body_params<F>(self, body_params: F) -> Self
    where
        F: Fn(&CallObj) -> Result<Array<Var>> + 'static,
    {
        let _ = body_params;
        let hook: Hook = region_body_params_abi::<F>;
        self.attr("FRegionGetBodyParams", Any::from(hook as *mut c_void), false)
    }

    /// `FRegionValidate`: a check of a complete region statement, run by the `RegionStmt`
    /// constructor after the body parameters. `validate` is a function or a closure that
    /// captures nothing.
    pub fn set_region_validate<F>(self, validate: F) -> Self
    where
        F: Fn(&RegionStmtObj) -> Result<()> + 'static,
    {
        let _ = validate;
        let hook: RegionHook = region_validate_abi::<F>;
        self.attr("FRegionValidate", Any::from(hook as *mut c_void), false)
    }

    fn attr(self, key: &str, value: Any, replace: bool) -> Self {
        let key = String::from(key);
        self.invoke(
            "_set_attr",
            &[AnyView::from(&key), AnyView::from(&value), AnyView::from(&replace)],
        )
    }

    /// Call the `ir.Op` method `method` on the operator with `args`.
    fn invoke(self, method: &str, args: &[AnyView]) -> Self {
        self.then(|op| {
            let packed: Vec<AnyView> =
                std::iter::once(AnyView::from(op)).chain(args.iter().copied()).collect();
            tvm_ffi::Function::from_type_key_method("ir.Op", method)?.call_packed(&packed)?;
            Ok(())
        })
    }

    fn then(self, step: impl FnOnce(&Op) -> Result<()>) -> Self {
        Self(self.0.and_then(|op| step(&op).map(|_| op)))
    }
}

/// A handle to an operator, kept in a `static` as C++ keeps one in `const Op& xxx_op()`. The
/// operator is looked up by name once per thread; [`OpHandle::get`] returns it and
/// [`OpHandle::is`] compares with it. The [`register_ops!`](crate::register_ops) blocks run when
/// their program or library loads; where the loader does not run them, the first lookup of an
/// operator that is not registered runs them.
///
/// ```no_run
/// use tvm_tirx_ext::sig::{self, IntExpr};
/// use tvm_tirx_ext::ir::{Call, CallObj, Expr, PrimType, Type, Var, UNKNOWN_LOC};
/// use tvm_tirx_ext::{OpDef, OpHandle};
/// use tvm_tirx_ext::tvm_ffi::Result;
///
/// fn first_arg_type(call: &CallObj) -> Result<Type> {
///     Ok(call.args.get(0)?.ty.clone())
/// }
///
/// tvm_tirx_ext::register_ops! {
///     OpDef::new("my_ext.clamp", "Clamp an index to a bound.")
///         .signature((sig::arg::<IntExpr>("index", ""), sig::arg::<IntExpr>("bound", "")))
///         .set_infer_type(first_arg_type);
/// }
///
/// static CLAMP: OpHandle = OpHandle::new("my_ext.clamp");
/// static LAUNCH_THREAD: OpHandle = OpHandle::new("tirx.launch_thread");
///
/// let index = Var::new("i", PrimType::int(32), UNKNOWN_LOC);
/// let bound = Var::new("n", PrimType::int(32), UNKNOWN_LOC);
/// let args = [Expr::from(&index), Expr::from(&bound)];
/// let clamped = Call::new(None, CLAMP.get()?, args, None, Vec::<Type>::new(), UNKNOWN_LOC)?;
/// assert!(CLAMP.is(&clamped.op) && !LAUNCH_THREAD.is(&clamped.op));
/// # Ok::<(), tvm_tirx_ext::tvm_ffi::Error>(())
/// ```
pub struct OpHandle {
    name: &'static str,
    /// This handle's slot in each thread's cache of fetched operators.
    slot: OnceLock<usize>,
    /// The operator once fetched, for identity tests: operators are registry singletons that
    /// live for the process, and `is` is the hot path of every pass that looks at `Call::op`.
    object: AtomicPtr<TVMFFIObject>,
}

/// The number of slots handed out to handles.
static SLOTS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// The operators this thread fetched, by handle slot.
    static FETCHED: RefCell<Vec<Option<Op>>> = const { RefCell::new(Vec::new()) };
}

impl OpHandle {
    /// The operator `name`, registered in C++, in Python or by a
    /// [`register_ops!`](crate::register_ops) block.
    pub const fn new(name: &'static str) -> Self {
        Self { name, slot: OnceLock::new(), object: AtomicPtr::new(std::ptr::null_mut()) }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    /// The operator. One that is not registered runs the blocks first; if it is still missing, the
    /// error says so, followed by the failures of the blocks, if any.
    pub fn get(&self) -> Result<Op> {
        self.with(Op::clone)
    }

    /// Whether `value`, such as a Call's `op`, is this operator (C++ `value.same_as(op)`). No
    /// value is an operator that cannot be fetched, such as one this build does not register.
    pub fn is(&self, value: &impl ObjectRefCore) -> bool {
        let cached = self.object.load(Ordering::Acquire);
        if !cached.is_null() {
            // SAFETY: the pointer is only compared, never dereferenced.
            let pointer = unsafe { ObjectArc::as_raw(ObjectRefCore::data(value)) };
            return pointer.cast::<TVMFFIObject>().cast_mut() == cached;
        }
        self.with(|op| {
            // Keep one reference so the pointer stays valid for the process (a second thread
            // racing here leaks one more, which is harmless for a registry singleton).
            // SAFETY: the raw pointer is never turned back into an `ObjectArc`.
            let pointer = unsafe { ObjectArc::into_raw(Op::into_data(op.clone())) };
            self.object.store(pointer.cast::<TVMFFIObject>().cast_mut(), Ordering::Release);
            value.same_as(op)
        })
        .unwrap_or(false)
    }

    /// `use_op` applied to the operator, fetched into this thread's slot the first time.
    fn with<R>(&self, use_op: impl FnOnce(&Op) -> R) -> Result<R> {
        let slot = *self.slot.get_or_init(|| SLOTS.fetch_add(1, Ordering::Relaxed));
        let cached = FETCHED.with(|fetched| fetched.borrow().get(slot).cloned().flatten());
        let op = match cached {
            Some(op) => op,
            None => {
                let op = Op::get(self.name).or_else(|missing| {
                    let registered = register_defined_ir();
                    Op::get(self.name).map_err(|_| match registered {
                        Ok(()) => missing,
                        Err(failed) => {
                            let message = format!("{}\n{}", missing.message(), failed.message());
                            Error::new(missing.kind(), &message, "")
                        }
                    })
                })?;
                FETCHED.with(|fetched| {
                    let mut fetched = fetched.borrow_mut();
                    if fetched.len() <= slot {
                        fetched.resize(slot + 1, None);
                    }
                    fetched[slot] = Some(op.clone());
                });
                op
            }
        };
        Ok(use_op(&op))
    }
}

/// The text of `doc` without the indentation its lines share and without the blank lines around
/// it. Text that is not indented is unchanged.
fn doc_text(doc: &str) -> std::string::String {
    let indent = doc
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.len() - line.trim_start().len())
        .min()
        .unwrap_or(0);
    let lines: Vec<&str> =
        doc.lines().map(|line| line.get(indent..).unwrap_or("").trim_end()).collect();
    lines.join("\n").trim().to_string()
}

/// Define operators as C++ does in a `TVM_FFI_STATIC_INIT_BLOCK`: each item is an [`OpDef`]
/// chain, which the block runs in order, stopping at the first that fails.
///
/// ```no_run
/// use tvm_tirx_ext::ir::op_attr::{CALL_EFFECT_KIND, FIXED_RETURN_TYPE};
/// use tvm_tirx_ext::sig::{self, IntExpr};
/// use tvm_tirx_ext::ir::{CallEffectKind, CallObj, Op, PrimType, Type};
/// use tvm_tirx_ext::OpDef;
/// use tvm_tirx_ext::tvm_ffi::Result;
///
/// fn first_arg_type(call: &CallObj) -> Result<Type> {
///     Ok(call.args.get(0)?.ty.clone())
/// }
///
/// tvm_tirx_ext::register_ops! {
///     OpDef::new("my_ext.clamp", "Clamp an index to a bound.")
///         .signature((sig::arg::<IntExpr>("index", ""), sig::arg::<IntExpr>("bound", "")))
///         .set_infer_type(first_arg_type)
///         .set_attr(CALL_EFFECT_KIND, CallEffectKind::Pure);
///
///     OpDef::new("my_ext.mark", "
///         Mark a point in the program.
///
///         The documentation is dedented.
///     ")
///         .set_attr(FIXED_RETURN_TYPE, PrimType::void())
///         .set_attr(CALL_EFFECT_KIND, CallEffectKind::Opaque);
/// }
///
/// // Registered when the program loaded.
/// assert_eq!(Op::get("my_ext.clamp")?.doc.as_str(), "Clamp an index to a bound.");
/// # Ok::<(), tvm_tirx_ext::tvm_ffi::Error>(())
/// ```
///
/// As a C++ static init block, a block runs when its program or library loads: it adds a
/// constructor, which the loader runs, and the first constructor runs every block of the program
/// or library with [`register_defined_ir`]. Python sees the operators of a library once it loads
/// it, by any means, and the library stays loaded for the process, as the operators keep its
/// hooks. An error cannot reach a caller when the library loads: it is printed, and
/// [`register_defined_ir`] (`tirx_register_ir`) and the handles report it again. Where the
/// loader does not run the constructor, an [`OpHandle`] runs the blocks on the first lookup of an
/// operator that is not registered.
///
/// A block runs once per process. Every library built on the crate links its own copy of the
/// crates it uses, so a block of a crate that several libraries link is in each of them; the first
/// library that runs it registers its operators, and the others skip it. A block of another
/// version of the crate, or whose text differs, is another block, and registering its operators
/// again is an error. A change to a hook's body alone does not make another block: rebuild every
/// library that links a crate after changing it.
#[macro_export]
macro_rules! register_ops {
    ($($def:expr);* $(;)?) => {
        $crate::__register_block!("register_ops!", stringify!($($def);*), || {
            $($crate::OpDef::op($def)?;)*
            Ok(())
        });
    };
}

/// The C ABI of a native operator hook, `NativeFunctionView<R(const CallNode*)>`: it returns
/// the result, or the error, in an owned `TVMFFIAny`.
type Hook = extern "C" fn(*const CallObj) -> TVMFFIAny;

/// The C ABI of a native hook on a region statement, `NativeFunctionView<R(const
/// RegionStmtNode*)>`.
type RegionHook = extern "C" fn(*const RegionStmtObj) -> TVMFFIAny;

extern "C" fn validate_abi<F: Fn(&CallObj) -> Result<()>>(call: *const CallObj) -> TVMFFIAny {
    call_hook(call, zero_sized::<F>())
}

extern "C" fn infer_type_abi<F: Fn(&CallObj) -> Result<Type>>(call: *const CallObj) -> TVMFFIAny {
    call_hook(call, zero_sized::<F>())
}

extern "C" fn validate_signature<L: Checks>(call: *const CallObj) -> TVMFFIAny {
    call_hook(call, sig::validate::<L>)
}

extern "C" fn region_body_params_abi<F: Fn(&CallObj) -> Result<Array<Var>>>(
    call: *const CallObj,
) -> TVMFFIAny {
    call_hook(call, zero_sized::<F>())
}

extern "C" fn region_validate_abi<F: Fn(&RegionStmtObj) -> Result<()>>(
    region: *const RegionStmtObj,
) -> TVMFFIAny {
    call_hook(region, zero_sized::<F>())
}

/// The only value of `F`, a function or a closure that captures nothing.
fn zero_sized<F>() -> F {
    const {
        assert!(
            std::mem::size_of::<F>() == 0,
            "a native operator hook is a function or a closure that captures nothing"
        )
    };
    // SAFETY: a zero-sized type has no bytes to initialize.
    unsafe { std::mem::zeroed() }
}

/// Run `hook` on the node C++ passes (the Call or region being typed or checked) and return its
/// result as the hook ABI does. A panic, which must not unwind into C++, becomes an error.
fn call_hook<N, R: Into<Any>>(node: *const N, hook: impl Fn(&N) -> Result<R>) -> TVMFFIAny {
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: C++ passes the node being typed or checked, alive for the call.
        match unsafe { node.as_ref() } {
            Some(node) => hook(node).map(Into::into),
            None => Err(Error::new(TYPE_ERROR, "Invalid Op Call", "")),
        }
    }));
    let value = match result {
        Ok(Ok(value)) => value,
        Ok(Err(error)) => Any::from(error),
        Err(payload) => Any::from(tvm_ffi::function_internal::panic_to_error(payload)),
    };
    // SAFETY: the C++ caller takes ownership of the value.
    unsafe { Any::into_raw_ffi_any(value) }
}
