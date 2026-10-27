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

//! The keys of the operator attributes (C++ `ffi::reflection::AttrKey<T>` and `tvm::op_attr`):
//! the name of an attribute and the type of its values, which `OpDef::set_attr` of the
//! `tvm-tirx-ext` crate takes to register one. [`tirx::op_attr`](crate::tirx::op_attr) has the
//! keys of TIRx.

use std::marker::PhantomData;

use tvm_ffi::Any;

use super::r#type::Type;

/// C++ `ffi::reflection::AttrKey<T>`: the name of an operator attribute and the type of its
/// values. The constants of this module and of [`tirx::op_attr`](crate::tirx::op_attr) are the
/// keys C++ declares, and `AttrKey::new` names another.
pub struct AttrKey<T> {
    name: &'static str,
    value: PhantomData<fn() -> T>,
}

impl<T> AttrKey<T> {
    pub const fn new(name: &'static str) -> Self {
        Self { name, value: PhantomData }
    }

    pub const fn name(&self) -> &'static str {
        self.name
    }
}

impl<T> Clone for AttrKey<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for AttrKey<T> {}

/// The runtime effect of a call (C++ `kCallEffectKind`, `TCallEffectKind`).
pub const CALL_EFFECT_KIND: AttrKey<CallEffectKind> = AttrKey::new("TCallEffectKind");
/// The result type of every call, independent of its operands, attributes and type arguments,
/// which takes precedence over `FInferType` (C++ `kFixedReturnType`, `TFixedReturnType`).
pub const FIXED_RETURN_TYPE: AttrKey<Type> = AttrKey::new("TFixedReturnType");

/// The runtime effect of an operator call (C++ `CallEffectKind`), the value of
/// [`CALL_EFFECT_KIND`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i64)]
pub enum CallEffectKind {
    /// An annotation, such as `likely`, which can be translated to the identity.
    ExprAnnotation = 0,
    /// A pure function, which does not interact with any external state.
    Pure = 1,
    /// A function that may read from states, such as memory.
    ReadState = 2,
    /// A function that may read from and write to states, such as memory; also
    /// [`CallEffectKind::Opaque`].
    UpdateState = 3,
    /// A special intrinsic that annotates call arguments, valid only as a direct argument to a
    /// call.
    SpecialCallArg = 4,
    /// Opaque information embedded in the expression, which cannot be generated as code.
    EmbedInfo = 5,
    /// A function that changes the control flow.
    ControlJump = 6,
}

impl CallEffectKind {
    /// An opaque function, about which nothing can be assumed: C++ `kOpaque`, which is
    /// [`CallEffectKind::UpdateState`].
    #[allow(non_upper_case_globals)]
    pub const Opaque: Self = Self::UpdateState;
}

impl From<CallEffectKind> for Any {
    fn from(kind: CallEffectKind) -> Self {
        Any::from(kind as i64)
    }
}
