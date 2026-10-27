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

//! Signature descriptors, for [`OpDef::signature`](crate::OpDef::signature) (C++ `sig::arg<T>` and friends): the arguments
//! an operator takes, with their documentation and the types a generated validator checks.
//!
//! [`arg`] and [`ty_arg`] are the fixed value and type arguments, in order, [`var_args`] and
//! [`var_ty_args`] a variadic tail of each, and [`call_attrs`] the type of `Call.attrs`. As an
//! argument type, a node type, such as [`StringImm`](tvm_tirx::ir::StringImm) or
//! [`Var`](tvm_tirx::ir::Var), accepts an instance of it, as C++ `TypeTraits<T>::CheckAnyStrict`
//! does, and [`Expr`] or [`Type`] accepts any. [`PrimExpr`], [`IntExpr`] and [`PrimVar`] are the
//! C++ view types, which also check the expression's type, as is [`TensorVar`]. Implement
//! [`ArgType`] for another check.

use std::any::TypeId;
use std::marker::PhantomData;

use tvm_ffi::tvm_ffi_sys::{TVMFFIAny, TVMFFITypeIndex};
use tvm_ffi::type_traits::AnyCompatible;
use tvm_ffi::{
    AnyView, DLDataType, DLDataTypeCode, Error, ObjectCore, ObjectRefCore, Result, String,
    TYPE_ERROR,
};
use tvm_tirx::ir::support;
use tvm_tirx::ir::{Attrs, CallObj, Expr, ExprObj, OpObj, PrimTypeObj, Type, VarObj};
use tvm_tirx::tirx::TensorTypeObj;

const INT: u8 = DLDataTypeCode::kDLInt as u8;
const UINT: u8 = DLDataTypeCode::kDLUInt as u8;

/// What a signature checks an argument against (C++ `TypeTraits<T>`).
pub trait ArgType: 'static {
    /// Whether `value`, which is not None, is accepted (C++ `CheckAnyStrict`).
    fn check(value: &TVMFFIAny) -> bool;
    /// The expected type, for an error (C++ `TypeStr`).
    fn type_str() -> std::string::String;
    /// The rejected value, for an error (C++ `GetMismatchTypeInfo`): its type key.
    fn mismatch(value: &TVMFFIAny) -> std::string::String {
        support::type_key(value.type_index)
    }
}

impl<T: ObjectRefCore + AnyCompatible + 'static> ArgType for T {
    fn check(value: &TVMFFIAny) -> bool {
        // SAFETY: `value` is a live FFI value.
        unsafe { T::check_any_strict(value) }
    }

    fn type_str() -> std::string::String {
        T::type_str()
    }
}

/// C++ `PrimExpr`: an expression whose type is a `PrimType`.
pub struct PrimExpr;

impl ArgType for PrimExpr {
    fn check(value: &TVMFFIAny) -> bool {
        prim_dtype(value).is_some()
    }

    fn type_str() -> std::string::String {
        "ir.PrimExpr".into()
    }

    fn mismatch(value: &TVMFFIAny) -> std::string::String {
        typed_expr_mismatch(value)
    }
}

/// C++ `IntExpr`: a `PrimExpr` of a scalar signed or unsigned integer type.
pub struct IntExpr;

impl ArgType for IntExpr {
    fn check(value: &TVMFFIAny) -> bool {
        prim_dtype(value).is_some_and(|dtype| dtype.lanes == 1 && matches!(dtype.code, INT | UINT))
    }

    fn type_str() -> std::string::String {
        "ir.IntExpr".into()
    }

    fn mismatch(value: &TVMFFIAny) -> std::string::String {
        typed_expr_mismatch(value)
    }
}

/// C++ `PrimVar`: a `Var`, not a subclass, whose type is a `PrimType`.
pub struct PrimVar;

impl ArgType for PrimVar {
    fn check(value: &TVMFFIAny) -> bool {
        var_of_type::<PrimTypeObj>(value)
    }

    fn type_str() -> std::string::String {
        VarObj::TYPE_KEY.into()
    }
}

/// C++ `tirx::TensorVar`: a `Var`, not a subclass, whose type is a
/// [`TensorType`](tvm_tirx::tirx::TensorType).
pub struct TensorVar;

impl ArgType for TensorVar {
    fn check(value: &TVMFFIAny) -> bool {
        var_of_type::<TensorTypeObj>(value)
    }

    fn type_str() -> std::string::String {
        VarObj::TYPE_KEY.into()
    }
}

/// Whether `value` is a `Var`, not a subclass, whose type is an `N`.
pub fn var_of_type<N: ObjectCore>(value: &TVMFFIAny) -> bool {
    value.type_index == VarObj::type_index()
        && expr(value).is_some_and(|var| var.ty.as_node::<N>().is_some())
}

/// The expression `value` holds, if it holds one.
fn expr(value: &TVMFFIAny) -> Option<&ExprObj> {
    let is_object = value.type_index >= TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32;
    // SAFETY: an object value of an `ir.Expr` type points to an `ExprObj`.
    unsafe {
        (is_object && Expr::check_any_strict(value))
            .then(|| &*(value.data_union.v_obj as *const ExprObj))
    }
}

fn prim_dtype(value: &TVMFFIAny) -> Option<DLDataType> {
    Some(expr(value)?.ty.as_node::<PrimTypeObj>()?.dtype)
}

/// C++ `TypeTraits<TypedExpr<T>>::GetMismatchTypeInfo`: an expression with its type.
fn typed_expr_mismatch(value: &TVMFFIAny) -> std::string::String {
    let Some(expr) = expr(value) else { return support::type_key(value.type_index) };
    let ty = match expr.ty.as_node::<PrimTypeObj>() {
        Some(prim) => support::dtype_name(prim.dtype),
        None => support::type_key(AnyView::from(&expr.ty).type_index()),
    };
    format!("{}[ty={ty}]", support::type_key(value.type_index))
}

/// A required value argument of type `T` (C++ `sig::arg<T>`); [`Expr`] accepts any.
pub fn arg<T: ArgType>(name: &str, doc: &str) -> Arg<T> {
    Arg { name: String::from(name), doc: String::from(doc), ty: PhantomData }
}

/// Zero or more further value arguments of type `T` (C++ `sig::var_args<T>`).
pub fn var_args<T: ArgType>(name: &str, doc: &str) -> VarArgs<T> {
    VarArgs { name: String::from(name), doc: String::from(doc), ty: PhantomData }
}

/// A required type argument of type `T` (C++ `sig::ty_arg<T>`); [`Type`] accepts any.
pub fn ty_arg<T: ArgType>(name: &str, doc: &str) -> TyArg<T> {
    TyArg { name: String::from(name), doc: String::from(doc), ty: PhantomData }
}

/// Zero or more further type arguments of type `T` (C++ `sig::var_ty_args<T>`).
pub fn var_ty_args<T: ArgType>(name: &str, doc: &str) -> VarTyArgs<T> {
    VarTyArgs { name: String::from(name), doc: String::from(doc), ty: PhantomData }
}

/// `Call.attrs` holds a `T`, which is also the operator's attrs type (C++
/// `sig::call_attrs<T>`).
pub fn call_attrs<T>() -> CallAttrs<T>
where
    T: ObjectRefCore + Into<Attrs> + 'static,
{
    CallAttrs { ty: PhantomData }
}

/// A required value argument of type `T`, from [`arg`].
pub struct Arg<T> {
    name: String,
    doc: String,
    ty: PhantomData<fn() -> T>,
}

/// The variadic value tail, of type `T`, from [`var_args`].
pub struct VarArgs<T> {
    name: String,
    doc: String,
    ty: PhantomData<fn() -> T>,
}

/// A required type argument of type `T`, from [`ty_arg`].
pub struct TyArg<T> {
    name: String,
    doc: String,
    ty: PhantomData<fn() -> T>,
}

/// The variadic type-argument tail, of type `T`, from [`var_ty_args`].
pub struct VarTyArgs<T> {
    name: String,
    doc: String,
    ty: PhantomData<fn() -> T>,
}

/// `Call.attrs`, of type `T`, from [`call_attrs`].
pub struct CallAttrs<T> {
    ty: PhantomData<fn() -> T>,
}

/// The argument descriptions a signature gives its operator.
#[doc(hidden)]
#[derive(Default)]
pub struct Description {
    arg_names: Vec<String>,
    arg_docs: Vec<String>,
    ty_arg_names: Vec<String>,
    ty_arg_docs: Vec<String>,
    /// Empty, or the tail's name and documentation.
    var_args: Vec<String>,
    var_ty_args: Vec<String>,
}

impl Description {
    /// The lists `ir.Op._set_signature` takes.
    pub(crate) fn into_lists(self) -> [Vec<String>; 6] {
        let Self { arg_names, arg_docs, ty_arg_names, ty_arg_docs, var_args, var_ty_args } = self;
        [arg_names, arg_docs, ty_arg_names, ty_arg_docs, var_args, var_ty_args]
    }
}

/// One descriptor of a signature: what it adds to the description, and what it checks.
#[doc(hidden)]
pub trait Descriptor: 'static {
    const ARGS: usize = 0;
    const TY_ARGS: usize = 0;
    const VAR_ARGS: usize = 0;
    const VAR_TY_ARGS: usize = 0;
    const CALL_ATTRS: usize = 0;
    /// The type key of the attrs type, if the descriptor names one.
    const ATTRS_TYPE_KEY: Option<&'static str> = None;

    fn describe(self, description: &mut Description);

    /// Check what this describes of `call`. `fixed` counts the signature's fixed value and
    /// type arguments, where its tails start, and `at` those declared before this one.
    fn check(call: &CallObj, op: &OpObj, fixed: (usize, usize), at: (usize, usize)) -> Result<()>;
}

impl<T: ArgType> Descriptor for Arg<T> {
    const ARGS: usize = 1;

    fn describe(self, description: &mut Description) {
        description.arg_names.push(self.name);
        description.arg_docs.push(self.doc);
    }

    fn check(call: &CallObj, op: &OpObj, _: (usize, usize), at: (usize, usize)) -> Result<()> {
        check_value::<T, Expr>(op, false, support::array_cells(&call.args), at.0)
    }
}

impl<T: ArgType> Descriptor for VarArgs<T> {
    const VAR_ARGS: usize = 1;

    fn describe(self, description: &mut Description) {
        description.var_args = vec![self.name, self.doc];
    }

    fn check(call: &CallObj, op: &OpObj, fixed: (usize, usize), _: (usize, usize)) -> Result<()> {
        let values = support::array_cells(&call.args);
        (fixed.0..values.len()).try_for_each(|i| check_value::<T, Expr>(op, false, values, i))
    }
}

impl<T: ArgType> Descriptor for TyArg<T> {
    const TY_ARGS: usize = 1;

    fn describe(self, description: &mut Description) {
        description.ty_arg_names.push(self.name);
        description.ty_arg_docs.push(self.doc);
    }

    fn check(call: &CallObj, op: &OpObj, _: (usize, usize), at: (usize, usize)) -> Result<()> {
        check_value::<T, Type>(op, true, support::array_cells(&call.ty_args), at.1)
    }
}

impl<T: ArgType> Descriptor for VarTyArgs<T> {
    const VAR_TY_ARGS: usize = 1;

    fn describe(self, description: &mut Description) {
        description.var_ty_args = vec![self.name, self.doc];
    }

    fn check(call: &CallObj, op: &OpObj, fixed: (usize, usize), _: (usize, usize)) -> Result<()> {
        let values = support::array_cells(&call.ty_args);
        (fixed.1..values.len()).try_for_each(|i| check_value::<T, Type>(op, true, values, i))
    }
}

impl<T: ObjectRefCore + 'static> Descriptor for CallAttrs<T> {
    const CALL_ATTRS: usize = 1;
    const ATTRS_TYPE_KEY: Option<&'static str> = Some(T::ContainerType::TYPE_KEY);

    fn describe(self, _: &mut Description) {}

    fn check(call: &CallObj, op: &OpObj, _: (usize, usize), _: (usize, usize)) -> Result<()> {
        let actual = match &call.attrs {
            Some(attrs) if attrs.as_node::<T::ContainerType>().is_some() => return Ok(()),
            Some(attrs) => {
                format!("`{}`", support::type_key(AnyView::from(attrs).type_index()))
            }
            None => "None".into(),
        };
        let message = format!(
            "Op `{}`: Call.attrs expected `{}`, got {actual}",
            op.name.as_str(),
            T::ContainerType::TYPE_KEY,
        );
        Err(Error::new(TYPE_ERROR, &message, ""))
    }
}

/// The checks of a signature, as a list of descriptors in declaration order.
#[doc(hidden)]
pub trait Checks: 'static {
    const ARGS: usize;
    const TY_ARGS: usize;
    const VAR_ARGS: usize;
    const VAR_TY_ARGS: usize;
    const CALL_ATTRS: usize;
    const ATTRS_TYPE_KEY: Option<&'static str>;

    /// Check the arguments of `call` these descriptors describe, in declaration order, as
    /// [`Descriptor::check`] does.
    fn check(call: &CallObj, op: &OpObj, fixed: (usize, usize), at: (usize, usize)) -> Result<()>;
}

impl Checks for () {
    const ARGS: usize = 0;
    const TY_ARGS: usize = 0;
    const VAR_ARGS: usize = 0;
    const VAR_TY_ARGS: usize = 0;
    const CALL_ATTRS: usize = 0;
    const ATTRS_TYPE_KEY: Option<&'static str> = None;

    fn check(_: &CallObj, _: &OpObj, _: (usize, usize), _: (usize, usize)) -> Result<()> {
        Ok(())
    }
}

impl<D: Descriptor, L: Checks> Checks for (D, L) {
    const ARGS: usize = D::ARGS + L::ARGS;
    const TY_ARGS: usize = D::TY_ARGS + L::TY_ARGS;
    const VAR_ARGS: usize = D::VAR_ARGS + L::VAR_ARGS;
    const VAR_TY_ARGS: usize = D::VAR_TY_ARGS + L::VAR_TY_ARGS;
    const CALL_ATTRS: usize = D::CALL_ATTRS + L::CALL_ATTRS;
    const ATTRS_TYPE_KEY: Option<&'static str> = match D::ATTRS_TYPE_KEY {
        Some(key) => Some(key),
        None => L::ATTRS_TYPE_KEY,
    };

    fn check(call: &CallObj, op: &OpObj, fixed: (usize, usize), at: (usize, usize)) -> Result<()> {
        D::check(call, op, fixed, at)?;
        L::check(call, op, fixed, (at.0 + D::ARGS, at.1 + D::TY_ARGS))
    }
}

/// What [`OpDef::signature`](crate::OpDef::signature) takes: a descriptor, or a tuple of them in order; `()` declares
/// no argument.
pub trait Specs {
    #[doc(hidden)]
    type Checks: Checks;

    #[doc(hidden)]
    fn describe(self, description: &mut Description);
}

impl<D: Descriptor> Specs for D {
    type Checks = (D, ());

    fn describe(self, description: &mut Description) {
        Descriptor::describe(self, description);
    }
}

/// `Specs` of the tuples of descriptors, whose checks nest in declaration order.
macro_rules! tuple_specs {
    (@nest) => { () };
    (@nest $head:ident $(, $rest:ident)*) => { ($head, tuple_specs!(@nest $($rest),*)) };
    ($($name:ident),*) => {
        impl<$($name: Descriptor),*> Specs for ($($name,)*) {
            type Checks = tuple_specs!(@nest $($name),*);

            #[allow(non_snake_case, unused_variables)]
            fn describe(self, description: &mut Description) {
                let ($($name,)*) = self;
                $(Descriptor::describe($name, description);)*
            }
        }
    };
}

tuple_specs!();
tuple_specs!(A);
tuple_specs!(A, B);
tuple_specs!(A, B, C);
tuple_specs!(A, B, C, D);
tuple_specs!(A, B, C, D, E);
tuple_specs!(A, B, C, D, E, F);
tuple_specs!(A, B, C, D, E, F, G);
tuple_specs!(A, B, C, D, E, F, G, H);
tuple_specs!(A, B, C, D, E, F, G, H, I);
tuple_specs!(A, B, C, D, E, F, G, H, I, J);
tuple_specs!(A, B, C, D, E, F, G, H, I, J, K);
tuple_specs!(A, B, C, D, E, F, G, H, I, J, K, L);

/// C++ `ValidateSignature<Specs...>`: the argument counts, then each argument's type and the
/// attrs, in declaration order.
pub(crate) fn validate<L: Checks>(call: &CallObj) -> Result<()> {
    const {
        assert!(
            L::VAR_ARGS <= 1 && L::VAR_TY_ARGS <= 1 && L::CALL_ATTRS <= 1,
            "a signature has at most one variadic tail of each kind and one call_attrs"
        )
    };
    let Some(op) = call.op.as_node::<OpObj>() else {
        return Err(Error::new(TYPE_ERROR, "Invalid Op Call", ""));
    };
    let count = |label: &str, expected: usize, tail: bool, unit: &str, got: usize| {
        let short = if tail { got < expected } else { got != expected };
        if !short {
            return Ok(());
        }
        let at_least = if tail { "at least " } else { "" };
        let plural = if expected == 1 { "" } else { "s" };
        let message = format!(
            "Op `{}`: {label} expected {at_least}{expected} {unit}{plural}, got {got}",
            op.name.as_str()
        );
        Err(Error::new(TYPE_ERROR, &message, ""))
    };
    count("Call.args", L::ARGS, L::VAR_ARGS == 1, "argument", call.args.len())?;
    count("Call.ty_args", L::TY_ARGS, L::VAR_TY_ARGS == 1, "type argument", call.ty_args.len())?;
    L::check(call, op, (L::ARGS, L::TY_ARGS), (0, 0))
}

/// Check argument `index` against `T`. `Base` is the category's base type, which C++ does
/// not check.
fn check_value<T: ArgType, Base: 'static>(
    op: &OpObj,
    type_arg: bool,
    values: &[TVMFFIAny],
    index: usize,
) -> Result<()> {
    if TypeId::of::<T>() == TypeId::of::<Base>() {
        return Ok(());
    }
    let value = &values[index];
    if value.type_index != TVMFFITypeIndex::kTVMFFINone as i32 && T::check(value) {
        return Ok(());
    }
    let (fixed, tail) = match type_arg {
        false => (&op.args_info, &op.var_args_info),
        true => (&op.ty_args_info, &op.var_ty_args_info),
    };
    let name = match fixed.get(index) {
        Ok(info) => Some(info.name.clone()),
        Err(_) => tail.as_ref().map(|info| info.name.clone()),
    };
    let name = match name {
        Some(name) if !name.as_str().is_empty() => format!(" (`{}`)", name.as_str()),
        _ => std::string::String::new(),
    };
    let message = format!(
        "Op `{}`: `{}[{index}]`{name} expected `{}`, got `{}`.",
        op.name.as_str(),
        if type_arg { "Call.ty_args" } else { "Call.args" },
        T::type_str(),
        T::mismatch(value),
    );
    Err(Error::new(TYPE_ERROR, &message, ""))
}
