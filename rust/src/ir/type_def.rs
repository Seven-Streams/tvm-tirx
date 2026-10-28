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

//! Registering IR types from Rust: [`register_type`], which a [`tvm_ffi::static_init_block!`]
//! calls for an `rs_class` type, as C++ registers a type with `refl::ObjectDef` in a
//! `TVM_FFI_STATIC_INIT_BLOCK`, and the [`Placeholder`] values of its fields.
//!
//! `rs_class` registers the fields of a type, but not a way to make one. C++ `refl::ObjectDef`
//! also registers a creator of an empty object, which tvm-ffi fills through the reflection of the
//! fields to construct the type from Python (the `__ffi_init__` it generates) and to read it from
//! a JSON graph. A Rust object cannot be empty, as its references are not null, so
//! [`register_type`] registers `__ffi_new__`, which makes an object whose fields hold
//! [`Placeholder`]s, and tvm-ffi then sets each field as it sets those of a C++ object.
//! `#[rs_class(..., placeholder)]` implements [`Placeholder`] for the object, from the
//! placeholders of the parent's object and the fields, and this module implements it for the
//! objects such a type extends and for the IR types its fields hold.
//!
//! The object a type extends (`TypeObj`, `ExprObj`, `VarObj`, ...) holds the placeholders of its
//! fields. Any other placeholder is a node its constructor builds from the simplest arguments, as
//! C++ builds and types it: an expression is `0`, an `int32` constant, and an operation `0 + 0`,
//! `0 == 0` or `false && false`; a type is `MissingType`, and each other type its empty case
//! (`void`, `()`, `void*` in the global memory, the scalar `void` tensor); an abstract reference
//! takes a node of a concrete type (a `Constant` is `0`, a `StagingExpr` `lambda: 0`, a `BaseFunc`
//! the function `() -> ()` without a body).

use tvm_ffi::tvm_ffi_sys::{TVMFFIByteArray, TVMFFITypeRegisterAttr};
use tvm_ffi::{
    Any, AnyCompatible, Array, Function, Map, ObjectCore, ObjectRefCore, Placeholder, Result,
    String,
};

use super::attrs::{Attrs, DictAttrs};
use super::dtype;
use super::expr::{
    Call, Constant, ConstantObj, DataTypeImm, Expr, ExprObj, FloatImm, GenericConst, GlobalVar,
    IntImm, LambdaExpr, OpaqueExpr, OpaqueExprObj, Range, StagingExpr, StagingExprObj, StringImm,
    TensorLoad, TensorRegion, Tuple, TupleGetItem, Var, VarObj,
};
use super::function::{BaseFunc, BaseFuncObj};
use super::location::Location;
use super::prim;
use super::r#type::{
    AnyType, FuncType, MissingType, NoneType, OpaqueType, PrimType, PtrType, PtrTypeObj,
    StringType, TensorRegionType, TupleType, Type, TypeObj,
};
use super::support;
use crate::runtime::MemoryScope;
use crate::tirx;

/// `impl Placeholder for $ty { fn placeholder() -> Self { $value } }` for each `$ty => $value`.
macro_rules! placeholders {
    ($($ty:ty => $value:expr;)*) => {
        $(impl Placeholder for $ty {
            fn placeholder() -> Self {
                $value
            }
        })*
    };
}

/// The placeholder of each binary node `prim::$node`: the node of `$operand` and `$operand`.
macro_rules! binary_placeholders {
    ($operand:expr => $($node:ident),*) => {
        placeholders! {
            $(prim::$node => support::infallible(
                prim::$node::new($operand, $operand, Location::unknown()),
            );)*
        }
    };
}

// The objects a type extends.
placeholders! {
    TypeObj => TypeObj::new(Location::unknown());
    ExprObj => ExprObj::new(Location::unknown(), Type::placeholder());
    ConstantObj => ConstantObj::new(Location::unknown(), Type::placeholder());
    StagingExprObj => StagingExprObj::new(Location::unknown(), Type::placeholder());
    OpaqueExprObj => OpaqueExprObj::new(Location::unknown(), Type::placeholder());
    VarObj => VarObj::new(Location::unknown(), Type::placeholder(), String::placeholder());
    BaseFuncObj => {
        BaseFuncObj::new(Location::unknown(), Type::placeholder(), DictAttrs::placeholder())
    };
}

// The fields of the nodes.
placeholders! {
    Location => Location::unknown();
    DictAttrs => DictAttrs::new(Map::default());
    // The global memory, `MemoryScope::Global()`, which C++ registers as the default scope of a
    // pointer: Rust has no lookup of a memory scope by name.
    MemoryScope => support::field_default(PtrTypeObj::type_index(), "scope")
        .and_then(|scope| MemoryScope::try_from(scope).ok())
        .expect("ir.PtrType registers the global memory as the default of its scope");
}

// Types.
placeholders! {
    Type => MissingType::new().into();
    MissingType => MissingType::new();
    NoneType => NoneType::new();
    AnyType => AnyType::new(Location::unknown());
    OpaqueType => OpaqueType::new(Location::unknown());
    StringType => StringType::new(Location::unknown());
    TensorRegionType => TensorRegionType::new();
    PrimType => PrimType::void();
    // `void*` (C++ `PtrType::VoidPtrType()`).
    PtrType => {
        PtrType::from_fields(Location::unknown(), PrimType::void(), MemoryScope::placeholder())
    };
    // `()`, the result of a function that returns nothing.
    TupleType => TupleType::new(Vec::<Type>::new(), Location::unknown());
    FuncType => FuncType::new(Vec::<Type>::new(), TupleType::placeholder());
    tirx::TensorMapType => tirx::TensorMapType::new(Location::unknown());
    // The scalar `void` tensor in the global memory, of the default alignment.
    tirx::TensorType => support::infallible(tirx::TensorType::new(
        Vec::<Expr>::new(),
        PrimType::void(),
        MemoryScope::placeholder(),
        0,
        None,
        Location::unknown(),
    ));
}

// Expressions.
placeholders! {
    Expr => IntImm::placeholder().into();
    Constant => IntImm::placeholder().into();
    IntImm => IntImm::int32(0);
    FloatImm => FloatImm::float32(0.0);
    StringImm => StringImm::new("", Location::unknown());
    DataTypeImm => DataTypeImm::from_dtype(dtype::void(), Location::unknown());
    // `None`.
    GenericConst => GenericConst::new(Location::unknown(), NoneType::new(), Any::default());
    Var => Var::new("", Type::placeholder(), Location::unknown());
    GlobalVar => GlobalVar::new("", Location::unknown());
    Tuple => Tuple::new(Vec::<Expr>::new(), Location::unknown());
    // `(0,)[0]`.
    TupleGetItem => {
        let tuple = Tuple::new([Expr::placeholder()], Location::unknown());
        support::infallible(TupleGetItem::from_tuple(tuple, 0, Location::unknown()))
    };
    // A call of the placeholder global variable without arguments.
    Call => Call::from_fields(
        Location::unknown(),
        Type::placeholder(),
        GlobalVar::placeholder(),
        Array::<Expr>::default(),
        None::<Attrs>,
        Array::<Type>::default(),
    );
    TensorLoad => {
        support::infallible(TensorLoad::new(tensor_var(), Vec::<Expr>::new(), Location::unknown()))
    };
    TensorRegion => TensorRegion::new(
        Location::unknown(),
        TensorRegionType::new(),
        tensor_var(),
        Array::<Range>::default(),
    );
    LambdaExpr => LambdaExpr::new(Vec::<Var>::new(), Expr::placeholder(), Location::unknown());
    StagingExpr => LambdaExpr::placeholder().into();
    OpaqueExpr => OpaqueExpr::new(Location::unknown(), Type::placeholder());
    BaseFunc => tirx::Function::placeholder().into();
    tirx::Function => tirx::Function::new(
        Vec::<Var>::new(),
        None,
        Type::placeholder(),
        DictAttrs::placeholder(),
        Location::unknown(),
    );
}

binary_placeholders!(0 => Add, Sub, Mul, Div, Mod, FloorDiv, FloorMod, Min, Max);
binary_placeholders!(0 => LShift, RShift, BitwiseAnd, BitwiseOr, BitwiseXor);
binary_placeholders!(0 => EQ, NE, LT, LE, GT, GE);
binary_placeholders!(false => And, Or);

placeholders! {
    prim::Cast => support::infallible(prim::Cast::new(PrimType::int(32), 0, Location::unknown()));
    prim::BitwiseNot => support::infallible(prim::BitwiseNot::new(0, Location::unknown()));
    prim::Not => support::infallible(prim::Not::new(false, Location::unknown()));
    prim::Select => support::infallible(prim::Select::new(false, 0, 0, Location::unknown()));
    // `let x = 0 in 0`.
    prim::Let => {
        let var = Var::new("", PrimType::int(32), Location::unknown());
        support::infallible(prim::Let::new(var, 0, 0, Location::unknown()))
    };
    // `[0, 1]`, `[0, 0]` and `[0]`.
    prim::Ramp => support::infallible(prim::Ramp::new(0, 1, 2, Location::unknown()));
    prim::Broadcast => support::infallible(prim::Broadcast::new(0, 2, Location::unknown()));
    prim::Shuffle => support::infallible(prim::Shuffle::new([0], [0], Location::unknown()));
}

/// The tensor of the `TensorLoad` and `TensorRegion` placeholders: a variable of the placeholder
/// tensor type.
fn tensor_var() -> Var {
    Var::new("", tirx::TensorType::placeholder(), Location::unknown())
}

/// Register the IR type of the reference `R`, an `rs_class` type, as C++ registers one with
/// `refl::ObjectDef`: `__ffi_new__`, from the [`Placeholder`] of the object, then the
/// `__ffi_init__` tvm-ffi generates from the reflected fields, then, when the receiver is
/// installed, a request for its constructor in registered destinations (`ir.expose_type`); an
/// absent receiver is a no-op, and an installed receiver's error is returned. With an explicitly
/// declared `my_ext` type API destination, Python then constructs the type from its fields,
/// `my_ext.Pair(T.int32, second=T.float32)`, as it constructs a C++ type, and a JSON graph holds
/// it. Import `tvm_tirx` before loading the library so exposure requests reach Python. Import the
/// extension module in Python scope, or supply it to text parsing through imports, `extra_vars` or
/// explicit parser namespace registration; type exposure does not add parser defaults.
///
/// Call it in a [`tvm_ffi::static_init_block!`], with the [`OpDef`](crate::ir::OpDef) chains of
/// the operators, so that the type is registered when its program or library loads, as a C++ one
/// is; the type's struct, reference and `impl`s are written outside the block, as any other:
///
/// ```no_run
/// use tvm_tirx::ir::{register_type, Location, Type, TypeObj};
/// use tvm_tirx::tvm_ffi::{self, def_object_ref, rs_class, ObjectRefCore};
///
/// /// The type of a pair of values, printed as `my_ext.Pair(first, second)`.
/// #[rs_class("my_ext.PairType", final, placeholder)]
/// pub struct PairTypeObj {
///     base: TypeObj,
///     pub first: Type,
///     #[reflect(kw_only)]
///     pub second: Type,
/// }
/// def_object_ref!(pub PairType, Type, PairTypeObj);
///
/// impl PairType {
///     pub fn new(first: impl Into<Type>, second: impl Into<Type>) -> Self {
///         PairTypeObj::new(TypeObj::new(Location::unknown()), first, second).into()
///     }
/// }
///
/// tvm_ffi::static_init_block! {
///     register_type::<PairType>()?;
/// }
/// ```
pub fn register_type<R>() -> Result<()>
where
    R: ObjectRefCore + AnyCompatible + From<R::ContainerType> + 'static,
    R::ContainerType: Placeholder,
{
    let type_index = R::ContainerType::type_index();
    let new =
        Function::from_typed(|| -> Result<R> { Ok(R::from(R::ContainerType::placeholder())) });
    register_type_attr(type_index, "__ffi_new__", Any::from(new))?;
    tvm_ffi::cached_global_func!("ffi._RegisterFFIInit").call_tuple((type_index,))?;
    if let Ok(expose) = Function::get_global("ir.expose_type") {
        expose.call_tuple((String::from(support::type_key(type_index).as_str()),))?;
    }
    Ok(())
}

/// Register `value` as the type attribute `name` of the type of `type_index`.
fn register_type_attr(type_index: i32, name: &str, mut value: Any) -> Result<()> {
    // SAFETY: `name` and `value` outlive the call, which copies them.
    let name = unsafe { TVMFFIByteArray::from_str(name) };
    tvm_ffi::check_safe_call!(unsafe {
        TVMFFITypeRegisterAttr(type_index, &name, Any::as_data_ptr(&mut value))
    })
}
