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

use tvm_ffi::tvm_ffi_sys::{TVMFFIByteArray, TVMFFITypeRegisterAttr};
use tvm_ffi::{Any, AnyCompatible, Array, Function, ObjectCore, ObjectRefCore, Result, String};

use super::expr::{Expr, IntImm};
use super::location::Location;
use super::r#type::{MissingType, PrimType, Type, TypeObj};
use super::support;

/// A valid value of a field type, which an object of a type [`register_type`] registers holds
/// until reflection sets the field: `__ffi_new__` makes an object with one in every field, and a
/// field `__ffi_init__` does not take (`#[reflect(init_off)]`) keeps it. The object of a type you
/// register implements it, from the placeholders of its parent's object and its fields; this
/// crate implements it for the common field types, and a field type of your own crate implements
/// it too.
pub trait Placeholder {
    /// The value.
    fn placeholder() -> Self;
}

macro_rules! default_placeholder {
    ($($ty:ty),*) => {
        $(impl Placeholder for $ty {
            fn placeholder() -> Self {
                Self::default()
            }
        })*
    };
}

default_placeholder!(bool, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

impl Placeholder for String {
    fn placeholder() -> Self {
        String::from("")
    }
}

impl<T: AnyCompatible + Clone + 'static> Placeholder for Array<T> {
    fn placeholder() -> Self {
        Array::new(vec![])
    }
}

impl<T> Placeholder for Option<T> {
    fn placeholder() -> Self {
        None
    }
}

impl Placeholder for Location {
    fn placeholder() -> Self {
        Location::unknown()
    }
}

impl Placeholder for TypeObj {
    fn placeholder() -> Self {
        TypeObj::new(Location::unknown())
    }
}

impl Placeholder for Type {
    fn placeholder() -> Self {
        MissingType::new().into()
    }
}

impl Placeholder for PrimType {
    fn placeholder() -> Self {
        PrimType::void()
    }
}

impl Placeholder for Expr {
    fn placeholder() -> Self {
        IntImm::new(Location::unknown(), PrimType::int(32), 0).into()
    }
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
/// use tvm_tirx::ir::{register_type, Location, Placeholder, Type, TypeObj};
/// use tvm_tirx::tvm_ffi::{self, def_object_ref, rs_class, ObjectRefCore};
///
/// /// The type of a pair of values, printed as `my_ext.Pair(first, second)`.
/// #[rs_class("my_ext.PairType", final)]
/// pub struct PairTypeObj {
///     base: TypeObj,
///     pub first: Type,
///     #[reflect(kw_only)]
///     pub second: Type,
/// }
/// def_object_ref!(pub PairType, Type, PairTypeObj);
///
/// impl Placeholder for PairTypeObj {
///     fn placeholder() -> Self {
///         Self::new(TypeObj::placeholder(), Type::placeholder(), Type::placeholder())
///     }
/// }
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
