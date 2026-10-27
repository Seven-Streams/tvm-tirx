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

//! Extending TIRx from Rust: operators and IR types defined in [`register_ops!`] and
//! [`register_types!`] blocks, as C++ defines them in a `TVM_FFI_STATIC_INIT_BLOCK`, over the
//! bindings of the `tvm-tirx` crate.
//!
//! The crate re-exports `tvm_tirx`, so that a crate built on it names everything through it:
//! [`ir`] is `tvm_tirx::ir`, [`tirx`] is `tvm_tirx::tirx`, and [`tvm_ffi`] is the TVM FFI crate
//! the bindings are built on. What this crate adds is at its root and in [`sig`].
//!
//! # Defining operators
//!
//! Each item of a [`register_ops!`] block is an [`OpDef`] chain, the C++ `OpDef`, with the
//! signature ([`sig`], whose descriptors generate the validator), the attributes, by the keys of
//! [`ir::op_attr`] and [`tirx::op_attr`], and the type inference, validator and region hooks as
//! Rust functions, which C++ calls natively:
//!
//! ```no_run
//! use tvm_tirx_ext::ir::op_attr::CALL_EFFECT_KIND;
//! use tvm_tirx_ext::ir::{CallEffectKind, CallObj, Type};
//! use tvm_tirx_ext::sig::{self, IntExpr};
//! use tvm_tirx_ext::tvm_ffi::Result;
//! use tvm_tirx_ext::{OpDef, OpHandle};
//!
//! fn index_type(call: &CallObj) -> Result<Type> {
//!     Ok(call.args.get(0)?.ty.clone())
//! }
//!
//! tvm_tirx_ext::register_ops! {
//!     OpDef::new("my_ext.clamp", "Clamp an index to the range `[0, bound)`.")
//!         .signature((
//!             sig::arg::<IntExpr>("index", "The index."),
//!             sig::arg::<IntExpr>("bound", "The exclusive upper bound."),
//!         ))
//!         .set_infer_type(index_type)
//!         .set_attr(CALL_EFFECT_KIND, CallEffectKind::Pure);
//! }
//!
//! static CLAMP: OpHandle = OpHandle::new("my_ext.clamp");
//! ```
//!
//! An [`OpHandle`] in a `static` keeps an operator as C++ keeps one in `const Op& xxx_op()`.
//! A block runs when its program or library loads, as a C++ one does, wherever in it the block
//! is. Every library built on the crate exports [`register_defined_ir`], which reports the
//! errors of the blocks that failed, as the TVM FFI function `tirx_register_ir`.
//!
//! # Defining a type
//!
//! [`tvm_ffi::rs_class`] defines a new IR type, a type of expressions as [`ir::PrimType`] and
//! [`ir::TupleType`] are, from its object struct, whose first field `base` is the [`ir::TypeObj`]
//! it extends, which holds the location. It registers the type under `ir.Type` with the
//! reflection of every other field, so that the C++ structural equality and hashing, deep copy,
//! `repr`, the TVMScript printer and Python see the fields as they see a C++ type's.
//! [`tvm_ffi::def_object_ref`] defines the reference, which converts into [`ir::Type`]. The
//! object's `new` takes the `TypeObj`, then the fields in order:
//!
//! ```no_run
//! use tvm_tirx_ext::ir::{Location, PrimType, Type, TypeObj};
//! use tvm_tirx_ext::tvm_ffi::{self, def_object_ref, rs_class, ObjectRefCore};
//!
//! /// The type of a pair of values.
//! #[rs_class("my_ext.PairType", final)]
//! pub struct PairTypeObj {
//!     base: TypeObj,
//!     pub first: Type,
//!     pub second: Type,
//!     /// Not part of structural equality or hashing.
//!     #[reflect(seq_hash_ignore)]
//!     pub note: tvm_ffi::String,
//! }
//!
//! def_object_ref!(pub PairType, Type, PairTypeObj);
//!
//! impl PairType {
//!     pub fn new(first: impl Into<Type>, second: impl Into<Type>) -> Self {
//!         PairTypeObj::new(TypeObj::new(Location::unknown()), first, second, "").into()
//!     }
//! }
//!
//! fn main() {
//!     let pair = PairType::new(PrimType::int(32), PrimType::float(32));
//!     let ty: Type = pair.into(); // an `ir.Type` that C++ sees as `my_ext.PairType`
//!     assert!(ty.as_node::<PairTypeObj>().is_some());
//! }
//! ```
//!
//! The generated code names `tvm_ffi`, which `use tvm_tirx_ext::tvm_ffi` brings into scope in a
//! crate that depends only on `tvm-tirx-ext`. The parent's object must have a public `new` and
//! implement `ShallowCopy`; the bindings generate both for a complete type that is not final, as
//! `ir.Type`. Written with its `def_object_ref!` in a [`register_types!`] block, the type is also
//! registered with the operators, so that Python constructs it from its fields and TVMScript
//! parses it, as a C++ type.
//!
//! # The library Python loads
//!
//! The crate is also built as a shared library, which the TIRx build installs as
//! `libtvm_tirx_rust` next to `libtvm_tirx_compiler` when it finds a cargo as new as the crate's
//! `rust-version` (`TVM_TIRX_USE_RUST`, `AUTO` by default). `import tvm_tirx` loads it through
//! the backend autoload (`tvm_tirx.backend.rust`), which registers the operators and types the
//! crate defines by the time the import returns, as the C++ ones are, and calls `tirx_register_ir`
//! to raise the error of a block that failed. Python uses the operators as it uses the C++ ones:
//! `Op.get` finds them, a `Call` built in Python runs their hooks, and the TVMScript printer prints
//! their calls. The TVMScript builder of an operator (`T.<name>` for `tirx.<name>`, `my_ext.<name>`
//! for `my_ext.<name>`) is generated by `tvm_tirx.tirx.script.expose_script_op`, unless Python has
//! one written for it, as for a C++ operator, and a type of a [`register_types!`] block gets the
//! class its TVMScript constructor names (`T.<Name>` for `tirx.<Name>Type`) from
//! `expose_script_type`. Both completion hooks use the canonical `script.expose_script_op` and
//! `script.expose_script_type` FFI receivers. Python installs these before loading native
//! libraries and defers exposure until its script builders are ready. Without receivers, native
//! registration proceeds without script exposure.
//!
//! A library of operators and types of its own, a `cdylib` built on the crate as
//! `examples/define_ir.rs` is, registers them when Python loads it, by `tvm_ffi.load_module` or
//! `ctypes`, as a C++ library does, and stays loaded for the process. Each library links its own
//! copy of the crates it is built from, so several libraries can have the same block, as those of a
//! crate they share: the first library that runs it registers what it defines and the others skip
//! it, so the libraries load together. A crate that a library depends on but never names is not
//! linked into it, nor are its blocks: name it, as `use my_ops as _;` does.
//!
//! # Tests
//!
//! Build `libtvm_tirx_compiler` first, as for `tvm-tirx`, whose build script finds it; this
//! crate's build script takes the loader path from it. The tests are in `tests/ext/`, laid out
//! like the modules:
//!
//! ```bash
//! uv run --no-sync cargo test --manifest-path rust/Cargo.toml -p tvm-tirx-ext
//! ```

mod op;
mod registry;
pub mod sig;
mod ty;

pub use tvm_tirx::*;

pub use op::{OpDef, OpHandle};
#[doc(hidden)]
pub use registry::__register_at_load;
pub use registry::{register_defined_ir, Registration, REGISTRATIONS};
#[doc(hidden)]
pub use ty::__register_type;
pub use ty::Placeholder;

// The errors of the blocks of every library built on the crate, which `import tvm_tirx` raises for
// `libtvm_tirx_rust`: `__tvm_ffi_tirx_register_ir`, the TVM FFI function `tirx_register_ir` of
// the library as `tvm_ffi.load_module` loads it. The module keeps the symbol, which `#[no_mangle]`
// exports from wherever it is, out of the crate's documentation.
#[doc(hidden)]
pub mod exports {
    tvm_ffi::tvm_ffi_dll_export_typed_func!(tirx_register_ir, crate::register_defined_ir);
}

/// What the exported macros expand to.
#[doc(hidden)]
pub mod __private {
    pub use linkme;
}
