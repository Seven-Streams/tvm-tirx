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

//! Scalar expressions: casts, arithmetic, bitwise and logical operations, comparisons, `Select`
//! and `Let`.
//!
//! The objects `python/tvm_tirx/ir/prim/expr.py` defines. A binary operation such as [`Add`]
//! holds its operands `a` and `b`. Their `new` is the native constructor, which checks the operand
//! types and derives the result type; the type-promoting operators are in [`super::op`].

// `custom-new` reserves `new` for a constructor with the native one's semantics (operand type
// checks and the derived result type); the generated `from_complete_fields` allocates the fields as
// given.
// tvm-ffi-stubgen(custom-new): prim.Cast
// tvm-ffi-stubgen(custom-new): prim.LShift
// tvm-ffi-stubgen(custom-new): prim.RShift
// tvm-ffi-stubgen(custom-new): prim.BitwiseAnd
// tvm-ffi-stubgen(custom-new): prim.BitwiseOr
// tvm-ffi-stubgen(custom-new): prim.BitwiseXor
// tvm-ffi-stubgen(custom-new): prim.BitwiseNot
// tvm-ffi-stubgen(custom-new): prim.Add
// tvm-ffi-stubgen(custom-new): prim.Sub
// tvm-ffi-stubgen(custom-new): prim.Mul
// tvm-ffi-stubgen(custom-new): prim.Div
// tvm-ffi-stubgen(custom-new): prim.Mod
// tvm-ffi-stubgen(custom-new): prim.FloorDiv
// tvm-ffi-stubgen(custom-new): prim.FloorMod
// tvm-ffi-stubgen(custom-new): prim.Min
// tvm-ffi-stubgen(custom-new): prim.Max
// tvm-ffi-stubgen(custom-new): prim.EQ
// tvm-ffi-stubgen(custom-new): prim.NE
// tvm-ffi-stubgen(custom-new): prim.LT
// tvm-ffi-stubgen(custom-new): prim.LE
// tvm-ffi-stubgen(custom-new): prim.GT
// tvm-ffi-stubgen(custom-new): prim.GE
// tvm-ffi-stubgen(custom-new): prim.And
// tvm-ffi-stubgen(custom-new): prim.Or
// tvm-ffi-stubgen(custom-new): prim.Not
// tvm-ffi-stubgen(custom-new): prim.Select
// tvm-ffi-stubgen(custom-new): prim.Let
// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): crate::ir::r#type::PrimType

// tvm-ffi-stubgen(begin): import-section
use crate::ir::expr::Expr;
use crate::ir::expr::ExprObj;
use crate::ir::expr::Var;
use crate::ir::location::Location;
use crate::ir::r#type::PrimType;
use crate::ir::r#type::Type;
use std::ops::Deref;
use tvm_ffi::ObjectArc;
use tvm_ffi::Result;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Cast
/// Complete: reflected fields fill [40, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Cast"]
#[type_final]
pub struct CastObj {
    base: ExprObj,
    pub value: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<CastObj>() == 48);
    assert!(::core::mem::align_of::<CastObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Cast {
    base: ObjectArc<CastObj>,
}

impl Deref for Cast {
    type Target = CastObj;
    fn deref(&self) -> &CastObj {
        &self.base
    }
}

impl Deref for CastObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl CastObj {
    pub(crate) fn new(loc: Location, ty: Type, value: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, value }
    }
}

impl Cast {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(loc: Location, ty: Type, value: Expr) -> Self {
        let obj = CastObj::new(loc, ty, value);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Cast => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.LShift
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.LShift"]
#[type_final]
pub struct LShiftObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<LShiftObj>() == 56);
    assert!(::core::mem::align_of::<LShiftObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct LShift {
    base: ObjectArc<LShiftObj>,
}

impl Deref for LShift {
    type Target = LShiftObj;
    fn deref(&self) -> &LShiftObj {
        &self.base
    }
}

impl Deref for LShiftObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl LShiftObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl LShift {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let obj = LShiftObj::new(loc, ty, a, b);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(LShift => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.RShift
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.RShift"]
#[type_final]
pub struct RShiftObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<RShiftObj>() == 56);
    assert!(::core::mem::align_of::<RShiftObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct RShift {
    base: ObjectArc<RShiftObj>,
}

impl Deref for RShift {
    type Target = RShiftObj;
    fn deref(&self) -> &RShiftObj {
        &self.base
    }
}

impl Deref for RShiftObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl RShiftObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl RShift {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let obj = RShiftObj::new(loc, ty, a, b);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(RShift => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.BitwiseAnd
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.BitwiseAnd"]
#[type_final]
pub struct BitwiseAndObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<BitwiseAndObj>() == 56);
    assert!(::core::mem::align_of::<BitwiseAndObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct BitwiseAnd {
    base: ObjectArc<BitwiseAndObj>,
}

impl Deref for BitwiseAnd {
    type Target = BitwiseAndObj;
    fn deref(&self) -> &BitwiseAndObj {
        &self.base
    }
}

impl Deref for BitwiseAndObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl BitwiseAndObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl BitwiseAnd {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let obj = BitwiseAndObj::new(loc, ty, a, b);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(BitwiseAnd => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.BitwiseOr
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.BitwiseOr"]
#[type_final]
pub struct BitwiseOrObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<BitwiseOrObj>() == 56);
    assert!(::core::mem::align_of::<BitwiseOrObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct BitwiseOr {
    base: ObjectArc<BitwiseOrObj>,
}

impl Deref for BitwiseOr {
    type Target = BitwiseOrObj;
    fn deref(&self) -> &BitwiseOrObj {
        &self.base
    }
}

impl Deref for BitwiseOrObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl BitwiseOrObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl BitwiseOr {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let obj = BitwiseOrObj::new(loc, ty, a, b);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(BitwiseOr => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.BitwiseXor
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.BitwiseXor"]
#[type_final]
pub struct BitwiseXorObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<BitwiseXorObj>() == 56);
    assert!(::core::mem::align_of::<BitwiseXorObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct BitwiseXor {
    base: ObjectArc<BitwiseXorObj>,
}

impl Deref for BitwiseXor {
    type Target = BitwiseXorObj;
    fn deref(&self) -> &BitwiseXorObj {
        &self.base
    }
}

impl Deref for BitwiseXorObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl BitwiseXorObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl BitwiseXor {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let obj = BitwiseXorObj::new(loc, ty, a, b);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(BitwiseXor => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.BitwiseNot
/// Complete: reflected fields fill [40, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.BitwiseNot"]
#[type_final]
pub struct BitwiseNotObj {
    base: ExprObj,
    pub a: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<BitwiseNotObj>() == 48);
    assert!(::core::mem::align_of::<BitwiseNotObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct BitwiseNot {
    base: ObjectArc<BitwiseNotObj>,
}

impl Deref for BitwiseNot {
    type Target = BitwiseNotObj;
    fn deref(&self) -> &BitwiseNotObj {
        &self.base
    }
}

impl Deref for BitwiseNotObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl BitwiseNotObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a }
    }
}

impl BitwiseNot {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(loc: Location, ty: Type, a: Expr) -> Self {
        let obj = BitwiseNotObj::new(loc, ty, a);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(BitwiseNot => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Add
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Add"]
#[type_final]
pub struct AddObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<AddObj>() == 56);
    assert!(::core::mem::align_of::<AddObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Add {
    base: ObjectArc<AddObj>,
}

impl Deref for Add {
    type Target = AddObj;
    fn deref(&self) -> &AddObj {
        &self.base
    }
}

impl Deref for AddObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl AddObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl Add {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let obj = AddObj::new(loc, ty, a, b);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Add => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Sub
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Sub"]
#[type_final]
pub struct SubObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<SubObj>() == 56);
    assert!(::core::mem::align_of::<SubObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Sub {
    base: ObjectArc<SubObj>,
}

impl Deref for Sub {
    type Target = SubObj;
    fn deref(&self) -> &SubObj {
        &self.base
    }
}

impl Deref for SubObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl SubObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl Sub {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let obj = SubObj::new(loc, ty, a, b);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Sub => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Mul
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Mul"]
#[type_final]
pub struct MulObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<MulObj>() == 56);
    assert!(::core::mem::align_of::<MulObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Mul {
    base: ObjectArc<MulObj>,
}

impl Deref for Mul {
    type Target = MulObj;
    fn deref(&self) -> &MulObj {
        &self.base
    }
}

impl Deref for MulObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl MulObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl Mul {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let obj = MulObj::new(loc, ty, a, b);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Mul => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Div
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Div"]
#[type_final]
pub struct DivObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<DivObj>() == 56);
    assert!(::core::mem::align_of::<DivObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Div {
    base: ObjectArc<DivObj>,
}

impl Deref for Div {
    type Target = DivObj;
    fn deref(&self) -> &DivObj {
        &self.base
    }
}

impl Deref for DivObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl DivObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl Div {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let obj = DivObj::new(loc, ty, a, b);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Div => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Mod
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Mod"]
#[type_final]
pub struct ModObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<ModObj>() == 56);
    assert!(::core::mem::align_of::<ModObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Mod {
    base: ObjectArc<ModObj>,
}

impl Deref for Mod {
    type Target = ModObj;
    fn deref(&self) -> &ModObj {
        &self.base
    }
}

impl Deref for ModObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl ModObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl Mod {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let obj = ModObj::new(loc, ty, a, b);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Mod => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.FloorDiv
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.FloorDiv"]
#[type_final]
pub struct FloorDivObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<FloorDivObj>() == 56);
    assert!(::core::mem::align_of::<FloorDivObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct FloorDiv {
    base: ObjectArc<FloorDivObj>,
}

impl Deref for FloorDiv {
    type Target = FloorDivObj;
    fn deref(&self) -> &FloorDivObj {
        &self.base
    }
}

impl Deref for FloorDivObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl FloorDivObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl FloorDiv {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let obj = FloorDivObj::new(loc, ty, a, b);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(FloorDiv => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.FloorMod
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.FloorMod"]
#[type_final]
pub struct FloorModObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<FloorModObj>() == 56);
    assert!(::core::mem::align_of::<FloorModObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct FloorMod {
    base: ObjectArc<FloorModObj>,
}

impl Deref for FloorMod {
    type Target = FloorModObj;
    fn deref(&self) -> &FloorModObj {
        &self.base
    }
}

impl Deref for FloorModObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl FloorModObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl FloorMod {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let obj = FloorModObj::new(loc, ty, a, b);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(FloorMod => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Min
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Min"]
#[type_final]
pub struct MinObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<MinObj>() == 56);
    assert!(::core::mem::align_of::<MinObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Min {
    base: ObjectArc<MinObj>,
}

impl Deref for Min {
    type Target = MinObj;
    fn deref(&self) -> &MinObj {
        &self.base
    }
}

impl Deref for MinObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl MinObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl Min {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let obj = MinObj::new(loc, ty, a, b);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Min => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Max
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Max"]
#[type_final]
pub struct MaxObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<MaxObj>() == 56);
    assert!(::core::mem::align_of::<MaxObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Max {
    base: ObjectArc<MaxObj>,
}

impl Deref for Max {
    type Target = MaxObj;
    fn deref(&self) -> &MaxObj {
        &self.base
    }
}

impl Deref for MaxObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl MaxObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl Max {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let obj = MaxObj::new(loc, ty, a, b);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Max => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.EQ
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.EQ"]
#[type_final]
pub struct EQObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<EQObj>() == 56);
    assert!(::core::mem::align_of::<EQObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct EQ {
    base: ObjectArc<EQObj>,
}

impl Deref for EQ {
    type Target = EQObj;
    fn deref(&self) -> &EQObj {
        &self.base
    }
}

impl Deref for EQObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl EQObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl EQ {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let obj = EQObj::new(loc, ty, a, b);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(EQ => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.NE
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.NE"]
#[type_final]
pub struct NEObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<NEObj>() == 56);
    assert!(::core::mem::align_of::<NEObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct NE {
    base: ObjectArc<NEObj>,
}

impl Deref for NE {
    type Target = NEObj;
    fn deref(&self) -> &NEObj {
        &self.base
    }
}

impl Deref for NEObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl NEObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl NE {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let obj = NEObj::new(loc, ty, a, b);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(NE => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.LT
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.LT"]
#[type_final]
pub struct LTObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<LTObj>() == 56);
    assert!(::core::mem::align_of::<LTObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct LT {
    base: ObjectArc<LTObj>,
}

impl Deref for LT {
    type Target = LTObj;
    fn deref(&self) -> &LTObj {
        &self.base
    }
}

impl Deref for LTObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl LTObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl LT {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let obj = LTObj::new(loc, ty, a, b);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(LT => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.LE
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.LE"]
#[type_final]
pub struct LEObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<LEObj>() == 56);
    assert!(::core::mem::align_of::<LEObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct LE {
    base: ObjectArc<LEObj>,
}

impl Deref for LE {
    type Target = LEObj;
    fn deref(&self) -> &LEObj {
        &self.base
    }
}

impl Deref for LEObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl LEObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl LE {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let obj = LEObj::new(loc, ty, a, b);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(LE => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.GT
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.GT"]
#[type_final]
pub struct GTObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<GTObj>() == 56);
    assert!(::core::mem::align_of::<GTObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct GT {
    base: ObjectArc<GTObj>,
}

impl Deref for GT {
    type Target = GTObj;
    fn deref(&self) -> &GTObj {
        &self.base
    }
}

impl Deref for GTObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl GTObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl GT {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let obj = GTObj::new(loc, ty, a, b);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(GT => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.GE
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.GE"]
#[type_final]
pub struct GEObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<GEObj>() == 56);
    assert!(::core::mem::align_of::<GEObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct GE {
    base: ObjectArc<GEObj>,
}

impl Deref for GE {
    type Target = GEObj;
    fn deref(&self) -> &GEObj {
        &self.base
    }
}

impl Deref for GEObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl GEObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl GE {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let obj = GEObj::new(loc, ty, a, b);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(GE => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.And
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.And"]
#[type_final]
pub struct AndObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<AndObj>() == 56);
    assert!(::core::mem::align_of::<AndObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct And {
    base: ObjectArc<AndObj>,
}

impl Deref for And {
    type Target = AndObj;
    fn deref(&self) -> &AndObj {
        &self.base
    }
}

impl Deref for AndObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl AndObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl And {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let obj = AndObj::new(loc, ty, a, b);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(And => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Or
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Or"]
#[type_final]
pub struct OrObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<OrObj>() == 56);
    assert!(::core::mem::align_of::<OrObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Or {
    base: ObjectArc<OrObj>,
}

impl Deref for Or {
    type Target = OrObj;
    fn deref(&self) -> &OrObj {
        &self.base
    }
}

impl Deref for OrObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl OrObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl Or {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let obj = OrObj::new(loc, ty, a, b);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Or => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Not
/// Complete: reflected fields fill [40, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Not"]
#[type_final]
pub struct NotObj {
    base: ExprObj,
    pub a: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<NotObj>() == 48);
    assert!(::core::mem::align_of::<NotObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Not {
    base: ObjectArc<NotObj>,
}

impl Deref for Not {
    type Target = NotObj;
    fn deref(&self) -> &NotObj {
        &self.base
    }
}

impl Deref for NotObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl NotObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a }
    }
}

impl Not {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(loc: Location, ty: Type, a: Expr) -> Self {
        let obj = NotObj::new(loc, ty, a);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Not => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Select
/// Complete: reflected fields fill [40, 64) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Select"]
#[type_final]
pub struct SelectObj {
    base: ExprObj,
    pub condition: Expr,
    pub true_value: Expr,
    pub false_value: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<SelectObj>() == 64);
    assert!(::core::mem::align_of::<SelectObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Select {
    base: ObjectArc<SelectObj>,
}

impl Deref for Select {
    type Target = SelectObj;
    fn deref(&self) -> &SelectObj {
        &self.base
    }
}

impl Deref for SelectObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl SelectObj {
    pub(crate) fn new(
        loc: Location,
        ty: Type,
        condition: Expr,
        true_value: Expr,
        false_value: Expr,
    ) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, condition, true_value, false_value }
    }
}

impl Select {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        loc: Location,
        ty: Type,
        condition: Expr,
        true_value: Expr,
        false_value: Expr,
    ) -> Self {
        let obj = SelectObj::new(loc, ty, condition, true_value, false_value);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Select => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Let
/// Complete: reflected fields fill [40, 64) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Let"]
#[type_final]
pub struct LetObj {
    base: ExprObj,
    pub var: Var,
    pub value: Expr,
    pub body: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<LetObj>() == 64);
    assert!(::core::mem::align_of::<LetObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Let {
    base: ObjectArc<LetObj>,
}

impl Deref for Let {
    type Target = LetObj;
    fn deref(&self) -> &LetObj {
        &self.base
    }
}

impl Deref for LetObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl LetObj {
    pub(crate) fn new(loc: Location, ty: Type, var: Var, value: Expr, body: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, var, value, body }
    }
}

impl Let {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        loc: Location,
        ty: Type,
        var: Var,
        value: Expr,
        body: Expr,
    ) -> Self {
        let obj = LetObj::new(loc, ty, var, value, body);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Let => Expr);
// tvm-ffi-stubgen(end)

impl Cast {
    /// The node converting `value` to `ty`, which has as many lanes as `value` (`prim.Cast`).
    pub fn new(
        ty: impl Into<PrimType>,
        value: impl Into<Expr>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        tvm_ffi::cached_global_func!("prim.Cast")
            .call_tuple((&ty.into(), &value.into(), &loc.into()))?
            .try_into()
    }
}

impl LShift {
    /// The node `a << b` over integer operands of the same type (`prim.LShift`).
    pub fn new(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Self> {
        tvm_ffi::cached_global_func!("prim.LShift")
            .call_tuple((&a.into(), &b.into(), &loc.into()))?
            .try_into()
    }
}

impl RShift {
    /// The node `a >> b` over integer operands of the same type (`prim.RShift`).
    pub fn new(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Self> {
        tvm_ffi::cached_global_func!("prim.RShift")
            .call_tuple((&a.into(), &b.into(), &loc.into()))?
            .try_into()
    }
}

impl BitwiseAnd {
    /// The node `a & b` over integer or boolean operands of the same type (`prim.BitwiseAnd`).
    pub fn new(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Self> {
        tvm_ffi::cached_global_func!("prim.BitwiseAnd")
            .call_tuple((&a.into(), &b.into(), &loc.into()))?
            .try_into()
    }
}

impl BitwiseOr {
    /// The node `a | b` over integer or boolean operands of the same type (`prim.BitwiseOr`).
    pub fn new(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Self> {
        tvm_ffi::cached_global_func!("prim.BitwiseOr")
            .call_tuple((&a.into(), &b.into(), &loc.into()))?
            .try_into()
    }
}

impl BitwiseXor {
    /// The node `a ^ b` over integer or boolean operands of the same type (`prim.BitwiseXor`).
    pub fn new(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Self> {
        tvm_ffi::cached_global_func!("prim.BitwiseXor")
            .call_tuple((&a.into(), &b.into(), &loc.into()))?
            .try_into()
    }
}

impl Add {
    /// The node `a + b` over operands of the same type (`prim.Add`).
    pub fn new(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Self> {
        tvm_ffi::cached_global_func!("prim.Add")
            .call_tuple((&a.into(), &b.into(), &loc.into()))?
            .try_into()
    }
}

impl Sub {
    /// The node `a - b` over operands of the same type (`prim.Sub`).
    pub fn new(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Self> {
        tvm_ffi::cached_global_func!("prim.Sub")
            .call_tuple((&a.into(), &b.into(), &loc.into()))?
            .try_into()
    }
}

impl Mul {
    /// The node `a * b` over operands of the same type (`prim.Mul`).
    pub fn new(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Self> {
        tvm_ffi::cached_global_func!("prim.Mul")
            .call_tuple((&a.into(), &b.into(), &loc.into()))?
            .try_into()
    }
}

impl Div {
    /// The node `a / b` over operands of the same type (`prim.Div`).
    pub fn new(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Self> {
        tvm_ffi::cached_global_func!("prim.Div")
            .call_tuple((&a.into(), &b.into(), &loc.into()))?
            .try_into()
    }
}

impl Mod {
    /// The node `a % b`, truncated, over operands of the same type (`prim.Mod`).
    pub fn new(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Self> {
        tvm_ffi::cached_global_func!("prim.Mod")
            .call_tuple((&a.into(), &b.into(), &loc.into()))?
            .try_into()
    }
}

impl FloorDiv {
    /// The node `floor(a / b)` over operands of the same type (`prim.FloorDiv`).
    pub fn new(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Self> {
        tvm_ffi::cached_global_func!("prim.FloorDiv")
            .call_tuple((&a.into(), &b.into(), &loc.into()))?
            .try_into()
    }
}

impl FloorMod {
    /// The node `a - floor(a / b) * b` over operands of the same type (`prim.FloorMod`).
    pub fn new(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Self> {
        tvm_ffi::cached_global_func!("prim.FloorMod")
            .call_tuple((&a.into(), &b.into(), &loc.into()))?
            .try_into()
    }
}

impl Min {
    /// The node `min(a, b)` over operands of the same type (`prim.Min`).
    pub fn new(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Self> {
        tvm_ffi::cached_global_func!("prim.Min")
            .call_tuple((&a.into(), &b.into(), &loc.into()))?
            .try_into()
    }
}

impl Max {
    /// The node `max(a, b)` over operands of the same type (`prim.Max`).
    pub fn new(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Self> {
        tvm_ffi::cached_global_func!("prim.Max")
            .call_tuple((&a.into(), &b.into(), &loc.into()))?
            .try_into()
    }
}

impl EQ {
    /// The node the boolean `a == b` over operands of the same type (`prim.EQ`).
    pub fn new(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Self> {
        tvm_ffi::cached_global_func!("prim.EQ")
            .call_tuple((&a.into(), &b.into(), &loc.into()))?
            .try_into()
    }
}

impl NE {
    /// The node the boolean `a != b` over operands of the same type (`prim.NE`).
    pub fn new(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Self> {
        tvm_ffi::cached_global_func!("prim.NE")
            .call_tuple((&a.into(), &b.into(), &loc.into()))?
            .try_into()
    }
}

impl LT {
    /// The node the boolean `a < b` over operands of the same type (`prim.LT`).
    pub fn new(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Self> {
        tvm_ffi::cached_global_func!("prim.LT")
            .call_tuple((&a.into(), &b.into(), &loc.into()))?
            .try_into()
    }
}

impl LE {
    /// The node the boolean `a <= b` over operands of the same type (`prim.LE`).
    pub fn new(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Self> {
        tvm_ffi::cached_global_func!("prim.LE")
            .call_tuple((&a.into(), &b.into(), &loc.into()))?
            .try_into()
    }
}

impl GT {
    /// The node the boolean `a > b` over operands of the same type (`prim.GT`).
    pub fn new(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Self> {
        tvm_ffi::cached_global_func!("prim.GT")
            .call_tuple((&a.into(), &b.into(), &loc.into()))?
            .try_into()
    }
}

impl GE {
    /// The node the boolean `a >= b` over operands of the same type (`prim.GE`).
    pub fn new(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Self> {
        tvm_ffi::cached_global_func!("prim.GE")
            .call_tuple((&a.into(), &b.into(), &loc.into()))?
            .try_into()
    }
}

impl And {
    /// The node the boolean `a && b` over boolean operands (`prim.And`).
    pub fn new(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Self> {
        tvm_ffi::cached_global_func!("prim.And")
            .call_tuple((&a.into(), &b.into(), &loc.into()))?
            .try_into()
    }
}

impl Or {
    /// The node the boolean `a || b` over boolean operands (`prim.Or`).
    pub fn new(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Self> {
        tvm_ffi::cached_global_func!("prim.Or")
            .call_tuple((&a.into(), &b.into(), &loc.into()))?
            .try_into()
    }
}

impl BitwiseNot {
    /// The node `~a` over an integer or boolean operand (`prim.BitwiseNot`).
    pub fn new(a: impl Into<Expr>, loc: impl Into<Location>) -> Result<Self> {
        tvm_ffi::cached_global_func!("prim.BitwiseNot")
            .call_tuple((&a.into(), &loc.into()))?
            .try_into()
    }
}

impl Not {
    /// The node the boolean `!a` over a boolean operand (`prim.Not`).
    pub fn new(a: impl Into<Expr>, loc: impl Into<Location>) -> Result<Self> {
        tvm_ffi::cached_global_func!("prim.Not").call_tuple((&a.into(), &loc.into()))?.try_into()
    }
}

impl Select {
    /// The node choosing `true_value` where `condition` holds and `false_value` elsewhere, which
    /// evaluates both (`prim.Select`).
    pub fn new(
        condition: impl Into<Expr>,
        true_value: impl Into<Expr>,
        false_value: impl Into<Expr>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        tvm_ffi::cached_global_func!("prim.Select")
            .call_tuple((&condition.into(), &true_value.into(), &false_value.into(), &loc.into()))?
            .try_into()
    }
}

impl Let {
    /// The node binding `var` to `value` in `body` (`prim.Let`).
    pub fn new(
        var: impl Into<Var>,
        value: impl Into<Expr>,
        body: impl Into<Expr>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        tvm_ffi::cached_global_func!("prim.Let")
            .call_tuple((&var.into(), &value.into(), &body.into(), &loc.into()))?
            .try_into()
    }
}
