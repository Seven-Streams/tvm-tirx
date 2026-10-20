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
#![allow(dead_code, unused_imports)]

//! Vector expressions: [`Ramp`], [`Broadcast`] and [`Shuffle`].
//!
//! The objects `python/tvm_tirx/ir/prim/vector_expr.py` defines. A [`Ramp`] is `lanes` values
//! from `base_` by `stride`, a [`Broadcast`] repeats a value over `lanes`, and a [`Shuffle`]
//! selects lanes from its vectors.

// `custom-new` reserves `new` for a constructor with the native one's semantics (operand checks and
// the derived vector type); the generated `from_complete_fields` allocates the fields as given.
// tvm-ffi-stubgen(custom-new): prim.Ramp
// tvm-ffi-stubgen(custom-new): prim.Broadcast
// tvm-ffi-stubgen(custom-new): prim.Shuffle

// tvm-ffi-stubgen(begin): import-section
use crate::ir::expr::Expr;
use crate::ir::expr::ExprObj;
use crate::ir::location::Location;
use crate::ir::r#type::Type;
use std::ops::Deref;
use tvm_ffi::Array;
use tvm_ffi::ObjectArc;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Ramp
/// Complete: reflected fields fill [40, 64) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Ramp"]
#[type_final]
pub struct RampObj {
    base: ExprObj,
    pub base_: Expr,
    pub stride: Expr,
    pub lanes: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<RampObj>() == 64);
    assert!(::core::mem::align_of::<RampObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Ramp {
    base: ObjectArc<RampObj>,
}

impl Deref for Ramp {
    type Target = RampObj;
    fn deref(&self) -> &RampObj {
        &self.base
    }
}

impl Deref for RampObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl RampObj {
    pub(crate) fn new(loc: Location, ty: Type, base_: Expr, stride: Expr, lanes: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, base_, stride, lanes }
    }
}

impl Ramp {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        loc: Location,
        ty: Type,
        base_: Expr,
        stride: Expr,
        lanes: Expr,
    ) -> Self {
        let obj = RampObj::new(loc, ty, base_, stride, lanes);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Ramp => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Broadcast
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Broadcast"]
#[type_final]
pub struct BroadcastObj {
    base: ExprObj,
    pub value: Expr,
    pub lanes: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<BroadcastObj>() == 56);
    assert!(::core::mem::align_of::<BroadcastObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Broadcast {
    base: ObjectArc<BroadcastObj>,
}

impl Deref for Broadcast {
    type Target = BroadcastObj;
    fn deref(&self) -> &BroadcastObj {
        &self.base
    }
}

impl Deref for BroadcastObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl BroadcastObj {
    pub(crate) fn new(loc: Location, ty: Type, value: Expr, lanes: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, value, lanes }
    }
}

impl Broadcast {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(loc: Location, ty: Type, value: Expr, lanes: Expr) -> Self {
        let obj = BroadcastObj::new(loc, ty, value, lanes);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Broadcast => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Shuffle
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Shuffle"]
#[type_final]
pub struct ShuffleObj {
    base: ExprObj,
    pub vectors: Array<Expr>,
    pub indices: Array<Expr>,
}

const _: () = {
    assert!(::core::mem::size_of::<ShuffleObj>() == 56);
    assert!(::core::mem::align_of::<ShuffleObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Shuffle {
    base: ObjectArc<ShuffleObj>,
}

impl Deref for Shuffle {
    type Target = ShuffleObj;
    fn deref(&self) -> &ShuffleObj {
        &self.base
    }
}

impl Deref for ShuffleObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl ShuffleObj {
    pub(crate) fn new(loc: Location, ty: Type, vectors: Array<Expr>, indices: Array<Expr>) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, vectors, indices }
    }
}

impl Shuffle {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        loc: Location,
        ty: Type,
        vectors: Array<Expr>,
        indices: Array<Expr>,
    ) -> Self {
        let obj = ShuffleObj::new(loc, ty, vectors, indices);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Shuffle => Expr);
// tvm-ffi-stubgen(end)
