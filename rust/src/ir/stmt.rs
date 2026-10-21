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

//! Statements: bindings, assertions, loops, conditionals, tensor stores and sequences.
//!
//! The objects `python/tvm_tirx/ir/stmt.py` defines. [`Stmt`] is the base of every statement and
//! carries its location. The constructors with the native semantics are `new` (`checked` for
//! [`While`]); a body given as one statement is wrapped in a [`SeqStmt`].

// `custom-new` reserves `new` for a constructor with the native one's semantics (defaults, implicit
// conversions, validation); the generated `from_complete_fields` allocates the fields as given.
// tvm-ffi-stubgen(custom-new): ir.Bind
// tvm-ffi-stubgen(custom-new): ir.AssertStmt
// tvm-ffi-stubgen(custom-new): ir.For
// tvm-ffi-stubgen(custom-new): ir.TensorStore
// tvm-ffi-stubgen(custom-new): ir.RegionStmt
// tvm-ffi-stubgen(custom-new): ir.SeqStmt
// tvm-ffi-stubgen(custom-new): ir.If
// tvm-ffi-stubgen(custom-new): ir.Evaluate
// tvm-ffi-stubgen(custom-new): ir.Return
// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::AnyView
// tvm-ffi-stubgen(import-object): tvm_ffi::Result

// tvm-ffi-stubgen(begin): import-section
use crate::ir::attrs::DictAttrs;
use crate::ir::expr::Expr;
use crate::ir::expr::StringImm;
use crate::ir::expr::Var;
use crate::ir::location::Location;
use crate::ir::op::Op;
use std::ops::Deref;
use tvm_ffi::Any;
use tvm_ffi::AnyView;
use tvm_ffi::Array;
use tvm_ffi::Map;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::Result;
use tvm_ffi::String;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.Stmt
/// Complete: reflected fields fill [24, 32) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Stmt"]
pub struct StmtObj {
    base: Object,
    pub loc: Location,
}

const _: () = {
    assert!(::core::mem::size_of::<StmtObj>() == 32);
    assert!(::core::mem::align_of::<StmtObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Stmt {
    base: ObjectArc<StmtObj>,
}

impl Deref for Stmt {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl StmtObj {
    pub(crate) fn new(loc: Location) -> Self {
        let base = Object::new();
        Self { base, loc }
    }
}

impl Stmt {
    /// Lossless complete-field allocation.
    pub fn new(loc: Location) -> Self {
        let obj = StmtObj::new(loc);
        Self { base: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.Bind
/// Complete: reflected fields fill [32, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Bind"]
#[type_final]
pub struct BindObj {
    base: StmtObj,
    pub var: Var,
    pub value: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<BindObj>() == 48);
    assert!(::core::mem::align_of::<BindObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Bind {
    base: ObjectArc<BindObj>,
}

impl Deref for Bind {
    type Target = BindObj;
    fn deref(&self) -> &BindObj {
        &self.base
    }
}

impl Deref for BindObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl BindObj {
    pub(crate) fn new(loc: Location, var: Var, value: Expr) -> Self {
        let base = StmtObj::new(loc);
        Self { base, var, value }
    }
}

impl Bind {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(loc: Location, var: Var, value: Expr) -> Self {
        let obj = BindObj::new(loc, var, value);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Bind => Stmt);
// tvm-ffi-stubgen(end)

impl Bind {
    /// Bind `var` to `value` for the rest of the enclosing scope (`ir.Bind`).
    pub fn new(
        var: impl Into<Var>,
        value: impl Into<Expr>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.Bind")
            .call_tuple((&var.into(), &value.into(), &loc.into()))?
            .try_into()
    }
}

// tvm-ffi-stubgen(begin): object/ir.AssertStmt
/// Complete: reflected fields fill [32, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.AssertStmt"]
#[type_final]
pub struct AssertStmtObj {
    base: StmtObj,
    pub condition: Expr,
    pub error_kind: StringImm,
    pub message_parts: Array<StringImm>,
}

const _: () = {
    assert!(::core::mem::size_of::<AssertStmtObj>() == 56);
    assert!(::core::mem::align_of::<AssertStmtObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct AssertStmt {
    base: ObjectArc<AssertStmtObj>,
}

impl Deref for AssertStmt {
    type Target = AssertStmtObj;
    fn deref(&self) -> &AssertStmtObj {
        &self.base
    }
}

impl Deref for AssertStmtObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl AssertStmtObj {
    pub(crate) fn new(
        loc: Location,
        condition: Expr,
        error_kind: StringImm,
        message_parts: Array<StringImm>,
    ) -> Self {
        let base = StmtObj::new(loc);
        Self { base, condition, error_kind, message_parts }
    }
}

impl AssertStmt {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        loc: Location,
        condition: Expr,
        error_kind: StringImm,
        message_parts: Array<StringImm>,
    ) -> Self {
        let obj = AssertStmtObj::new(loc, condition, error_kind, message_parts);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(AssertStmt => Stmt);
// tvm-ffi-stubgen(end)

impl AssertStmt {
    /// Assert `condition`, raising an `error_kind` error with the message `message_parts` when it
    /// fails (`ir.AssertStmt`).
    pub fn new(
        condition: impl Into<Expr>,
        error_kind: impl Into<StringImm>,
        message_parts: impl IntoIterator<Item = impl Into<StringImm>>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let message_parts = message_parts.into_iter().map(Into::into).collect::<Array<StringImm>>();
        tvm_ffi::cached_global_func!("ir.AssertStmt")
            .call_tuple((&condition.into(), &error_kind.into(), &message_parts, &loc.into()))?
            .try_into()
    }
}

// tvm-ffi-stubgen(begin): object/ir.For
/// Complete: reflected fields fill [32, 88) exactly (alignment padding [68, 72)).
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.For"]
#[type_final]
pub struct ForObj {
    base: StmtObj,
    pub var: Var,
    pub begin: Expr,
    pub end: Expr,
    pub step: Expr,
    pub kind: i32,
    pub body: SeqStmt,
    pub annotations: Map<String, Any>,
}

const _: () = {
    assert!(::core::mem::size_of::<ForObj>() == 88);
    assert!(::core::mem::align_of::<ForObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct For {
    base: ObjectArc<ForObj>,
}

impl Deref for For {
    type Target = ForObj;
    fn deref(&self) -> &ForObj {
        &self.base
    }
}

impl Deref for ForObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl ForObj {
    pub(crate) fn new(
        loc: Location,
        var: Var,
        begin: Expr,
        end: Expr,
        step: Expr,
        kind: i32,
        body: SeqStmt,
        annotations: Map<String, Any>,
    ) -> Self {
        let base = StmtObj::new(loc);
        Self { base, var, begin, end, step, kind, body, annotations }
    }
}

impl For {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        loc: Location,
        var: Var,
        begin: Expr,
        end: Expr,
        step: Expr,
        kind: i32,
        body: SeqStmt,
        annotations: Map<String, Any>,
    ) -> Self {
        let obj = ForObj::new(loc, var, begin, end, step, kind, body, annotations);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(For => Stmt);
// tvm-ffi-stubgen(end)

/// The execution semantics of a [`For`] loop: `ir::ForKind`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum ForKind {
    /// Sequential iterations.
    #[default]
    Default = 0,
    /// Iterations may run in parallel.
    Parallel = 1,
    /// The body is vectorized.
    Vectorized = 2,
    /// The loop is unrolled.
    Unrolled = 3,
}

impl For {
    /// The loop of `var` from `begin` to `end` by `step`, running `body` with the `kind` semantics
    /// (`ir.For`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        var: impl Into<Var>,
        begin: impl Into<Expr>,
        end: impl Into<Expr>,
        step: impl Into<Expr>,
        kind: ForKind,
        body: impl Into<Stmt>,
        annotations: Option<&Map<String, Any>>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let (var, begin, end, step) = (var.into(), begin.into(), end.into(), step.into());
        let (kind, body, annotations, loc) =
            (kind as i32, body.into(), annotations.cloned(), loc.into());
        tvm_ffi::cached_global_func!("ir.For")
            .call_packed(&[
                AnyView::from(&var),
                AnyView::from(&begin),
                AnyView::from(&end),
                AnyView::from(&step),
                AnyView::from(&kind),
                AnyView::from(&body),
                AnyView::from(&annotations),
                AnyView::from(&loc),
            ])?
            .try_into()
    }
}

// tvm-ffi-stubgen(begin): object/ir.While
/// Complete: reflected fields fill [32, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.While"]
#[type_final]
pub struct WhileObj {
    base: StmtObj,
    pub condition: Expr,
    pub body: SeqStmt,
}

const _: () = {
    assert!(::core::mem::size_of::<WhileObj>() == 48);
    assert!(::core::mem::align_of::<WhileObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct While {
    base: ObjectArc<WhileObj>,
}

impl Deref for While {
    type Target = WhileObj;
    fn deref(&self) -> &WhileObj {
        &self.base
    }
}

impl Deref for WhileObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl WhileObj {
    pub(crate) fn new(loc: Location, condition: Expr, body: SeqStmt) -> Self {
        let base = StmtObj::new(loc);
        Self { base, condition, body }
    }
}

impl While {
    /// Lossless complete-field allocation.
    pub fn new(loc: Location, condition: Expr, body: SeqStmt) -> Self {
        let obj = WhileObj::new(loc, condition, body);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(While => Stmt);
// tvm-ffi-stubgen(end)

impl While {
    /// The loop running `body` while the scalar `condition` holds, checked to be well formed
    /// (`ir.While`).
    pub fn checked(
        condition: impl Into<Expr>,
        body: impl Into<Stmt>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.While")
            .call_tuple((&condition.into(), &body.into(), &loc.into()))?
            .try_into()
    }
}

// tvm-ffi-stubgen(begin): object/ir.TensorStore
/// Complete: reflected fields fill [32, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.TensorStore"]
#[type_final]
pub struct TensorStoreObj {
    base: StmtObj,
    pub dest: Expr,
    pub indices: Array<Expr>,
    pub value: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<TensorStoreObj>() == 56);
    assert!(::core::mem::align_of::<TensorStoreObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct TensorStore {
    base: ObjectArc<TensorStoreObj>,
}

impl Deref for TensorStore {
    type Target = TensorStoreObj;
    fn deref(&self) -> &TensorStoreObj {
        &self.base
    }
}

impl Deref for TensorStoreObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl TensorStoreObj {
    pub(crate) fn new(loc: Location, dest: Expr, indices: Array<Expr>, value: Expr) -> Self {
        let base = StmtObj::new(loc);
        Self { base, dest, indices, value }
    }
}

impl TensorStore {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        loc: Location,
        dest: Expr,
        indices: Array<Expr>,
        value: Expr,
    ) -> Self {
        let obj = TensorStoreObj::new(loc, dest, indices, value);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(TensorStore => Stmt);
// tvm-ffi-stubgen(end)

impl TensorStore {
    /// Store `value` into the element of `dest` at `indices` (`ir.TensorStore`).
    pub fn new(
        dest: impl Into<Expr>,
        indices: impl IntoIterator<Item = impl Into<Expr>>,
        value: impl Into<Expr>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let indices = indices.into_iter().map(Into::into).collect::<Array<Expr>>();
        tvm_ffi::cached_global_func!("ir.TensorStore")
            .call_tuple((&dest.into(), &indices, &value.into(), &loc.into()))?
            .try_into()
    }
}

// tvm-ffi-stubgen(begin): object/ir.RegionStmt
/// Complete: reflected fields fill [32, 80) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.RegionStmt"]
#[type_final]
pub struct RegionStmtObj {
    base: StmtObj,
    pub op: Op,
    pub args: Array<Expr>,
    pub body_params: Array<Var>,
    pub attrs: DictAttrs,
    pub body: SeqStmt,
    pub result_vars: Array<Var>,
}

const _: () = {
    assert!(::core::mem::size_of::<RegionStmtObj>() == 80);
    assert!(::core::mem::align_of::<RegionStmtObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct RegionStmt {
    base: ObjectArc<RegionStmtObj>,
}

impl Deref for RegionStmt {
    type Target = RegionStmtObj;
    fn deref(&self) -> &RegionStmtObj {
        &self.base
    }
}

impl Deref for RegionStmtObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl RegionStmtObj {
    pub(crate) fn new(
        loc: Location,
        op: Op,
        args: Array<Expr>,
        body_params: Array<Var>,
        attrs: DictAttrs,
        body: SeqStmt,
        result_vars: Array<Var>,
    ) -> Self {
        let base = StmtObj::new(loc);
        Self { base, op, args, body_params, attrs, body, result_vars }
    }
}

impl RegionStmt {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        loc: Location,
        op: Op,
        args: Array<Expr>,
        body_params: Array<Var>,
        attrs: DictAttrs,
        body: SeqStmt,
        result_vars: Array<Var>,
    ) -> Self {
        let obj = RegionStmtObj::new(loc, op, args, body_params, attrs, body, result_vars);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(RegionStmt => Stmt);
// tvm-ffi-stubgen(end)

impl RegionStmt {
    /// The region of the region operator `op` on `args`, whose `body` binds `body_params` and
    /// defines `result_vars`, checked against the operator (`ir.RegionStmt`).
    pub fn new(
        op: impl Into<Op>,
        args: impl IntoIterator<Item = impl Into<Expr>>,
        body_params: impl IntoIterator<Item = impl Into<Var>>,
        attrs: impl Into<DictAttrs>,
        body: impl Into<Stmt>,
        result_vars: impl IntoIterator<Item = impl Into<Var>>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let (op, attrs, body, loc) = (op.into(), attrs.into(), body.into(), loc.into());
        let args = args.into_iter().map(Into::into).collect::<Array<Expr>>();
        let body_params = body_params.into_iter().map(Into::into).collect::<Array<Var>>();
        let result_vars = result_vars.into_iter().map(Into::into).collect::<Array<Var>>();
        tvm_ffi::cached_global_func!("ir.RegionStmt")
            .call_packed(&[
                AnyView::from(&op),
                AnyView::from(&args),
                AnyView::from(&body_params),
                AnyView::from(&attrs),
                AnyView::from(&body),
                AnyView::from(&result_vars),
                AnyView::from(&loc),
            ])?
            .try_into()
    }
}

// tvm-ffi-stubgen(begin): object/ir.SeqStmt
/// Complete: reflected fields fill [32, 40) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.SeqStmt"]
#[type_final]
pub struct SeqStmtObj {
    base: StmtObj,
    pub seq: Array<Stmt>,
}

const _: () = {
    assert!(::core::mem::size_of::<SeqStmtObj>() == 40);
    assert!(::core::mem::align_of::<SeqStmtObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct SeqStmt {
    base: ObjectArc<SeqStmtObj>,
}

impl Deref for SeqStmt {
    type Target = SeqStmtObj;
    fn deref(&self) -> &SeqStmtObj {
        &self.base
    }
}

impl Deref for SeqStmtObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl SeqStmtObj {
    pub(crate) fn new(loc: Location, seq: Array<Stmt>) -> Self {
        let base = StmtObj::new(loc);
        Self { base, seq }
    }
}

impl SeqStmt {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(loc: Location, seq: Array<Stmt>) -> Self {
        let obj = SeqStmtObj::new(loc, seq);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(SeqStmt => Stmt);
// tvm-ffi-stubgen(end)

impl SeqStmt {
    /// The sequence of `seq`, with nested sequences and no-ops flattened; at the location of its
    /// statements when `loc` is `None` (`ir.SeqStmt`).
    pub fn new(
        seq: impl IntoIterator<Item = impl Into<Stmt>>,
        loc: Option<&Location>,
    ) -> Result<Self> {
        let seq = seq.into_iter().map(Into::into).collect::<Array<Stmt>>();
        tvm_ffi::cached_global_func!("ir.SeqStmt").call_tuple((&seq, loc.cloned()))?.try_into()
    }
}

// tvm-ffi-stubgen(begin): object/ir.If
/// Complete: reflected fields fill [32, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.If"]
#[type_final]
pub struct IfObj {
    base: StmtObj,
    pub condition: Expr,
    pub then_case: SeqStmt,
    pub else_case: Option<SeqStmt>,
}

const _: () = {
    assert!(::core::mem::size_of::<IfObj>() == 56);
    assert!(::core::mem::align_of::<IfObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct If {
    base: ObjectArc<IfObj>,
}

impl Deref for If {
    type Target = IfObj;
    fn deref(&self) -> &IfObj {
        &self.base
    }
}

impl Deref for IfObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl IfObj {
    pub(crate) fn new(
        loc: Location,
        condition: Expr,
        then_case: SeqStmt,
        else_case: Option<SeqStmt>,
    ) -> Self {
        let base = StmtObj::new(loc);
        Self { base, condition, then_case, else_case }
    }
}

impl If {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        loc: Location,
        condition: Expr,
        then_case: SeqStmt,
        else_case: Option<SeqStmt>,
    ) -> Self {
        let obj = IfObj::new(loc, condition, then_case, else_case);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(If => Stmt);
// tvm-ffi-stubgen(end)

impl If {
    /// Run `then_case` if `condition` holds, else `else_case` if given (`ir.If`).
    pub fn new(
        condition: impl Into<Expr>,
        then_case: impl Into<Stmt>,
        else_case: Option<&Stmt>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.If")
            .call_tuple((&condition.into(), &then_case.into(), else_case.cloned(), &loc.into()))?
            .try_into()
    }
}

// tvm-ffi-stubgen(begin): object/ir.Evaluate
/// Complete: reflected fields fill [32, 40) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Evaluate"]
#[type_final]
pub struct EvaluateObj {
    base: StmtObj,
    pub value: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<EvaluateObj>() == 40);
    assert!(::core::mem::align_of::<EvaluateObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Evaluate {
    base: ObjectArc<EvaluateObj>,
}

impl Deref for Evaluate {
    type Target = EvaluateObj;
    fn deref(&self) -> &EvaluateObj {
        &self.base
    }
}

impl Deref for EvaluateObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl EvaluateObj {
    pub(crate) fn new(loc: Location, value: Expr) -> Self {
        let base = StmtObj::new(loc);
        Self { base, value }
    }
}

impl Evaluate {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(loc: Location, value: Expr) -> Self {
        let obj = EvaluateObj::new(loc, value);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Evaluate => Stmt);
// tvm-ffi-stubgen(end)

impl Evaluate {
    /// Evaluate `value` for its side effects (`ir.Evaluate`).
    pub fn new(value: impl Into<Expr>, loc: impl Into<Location>) -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.Evaluate")
            .call_tuple((&value.into(), &loc.into()))?
            .try_into()
    }
}

// tvm-ffi-stubgen(begin): object/ir.Break
/// Complete: reflected fields fill [32, 32) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Break"]
#[type_final]
pub struct BreakObj {
    base: StmtObj,
}

const _: () = {
    assert!(::core::mem::size_of::<BreakObj>() == 32);
    assert!(::core::mem::align_of::<BreakObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Break {
    base: ObjectArc<BreakObj>,
}

impl Deref for Break {
    type Target = BreakObj;
    fn deref(&self) -> &BreakObj {
        &self.base
    }
}

impl Deref for BreakObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl BreakObj {
    pub(crate) fn new(loc: Location) -> Self {
        let base = StmtObj::new(loc);
        Self { base }
    }
}

impl Break {
    /// Lossless complete-field allocation.
    pub fn new(loc: Location) -> Self {
        let obj = BreakObj::new(loc);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Break => Stmt);
// tvm-ffi-stubgen(end)

// `ir.Break` stores its location as given: the generated `Break::new` is the same.

// tvm-ffi-stubgen(begin): object/ir.Return
/// Complete: reflected fields fill [32, 40) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Return"]
#[type_final]
pub struct ReturnObj {
    base: StmtObj,
    pub value: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<ReturnObj>() == 40);
    assert!(::core::mem::align_of::<ReturnObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Return {
    base: ObjectArc<ReturnObj>,
}

impl Deref for Return {
    type Target = ReturnObj;
    fn deref(&self) -> &ReturnObj {
        &self.base
    }
}

impl Deref for ReturnObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl ReturnObj {
    pub(crate) fn new(loc: Location, value: Expr) -> Self {
        let base = StmtObj::new(loc);
        Self { base, value }
    }
}

impl Return {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(loc: Location, value: Expr) -> Self {
        let obj = ReturnObj::new(loc, value);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Return => Stmt);
// tvm-ffi-stubgen(end)

impl Return {
    /// Return `value` from the enclosing function (`ir.Return`).
    pub fn new(value: impl Into<Expr>, loc: impl Into<Location>) -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.Return")
            .call_tuple((&value.into(), &loc.into()))?
            .try_into()
    }
}

// tvm-ffi-stubgen(begin): object/ir.Continue
/// Complete: reflected fields fill [32, 32) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Continue"]
#[type_final]
pub struct ContinueObj {
    base: StmtObj,
}

const _: () = {
    assert!(::core::mem::size_of::<ContinueObj>() == 32);
    assert!(::core::mem::align_of::<ContinueObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Continue {
    base: ObjectArc<ContinueObj>,
}

impl Deref for Continue {
    type Target = ContinueObj;
    fn deref(&self) -> &ContinueObj {
        &self.base
    }
}

impl Deref for ContinueObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl ContinueObj {
    pub(crate) fn new(loc: Location) -> Self {
        let base = StmtObj::new(loc);
        Self { base }
    }
}

impl Continue {
    /// Lossless complete-field allocation.
    pub fn new(loc: Location) -> Self {
        let obj = ContinueObj::new(loc);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Continue => Stmt);
// tvm-ffi-stubgen(end)

// `ir.Continue` stores its location as given: the generated `Continue::new` is the same.
