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

//! The analyzer, its bound and modular results, and the forms of its simplifiers.
//!
//! The objects `python/tvm_tirx/sym/analyzer.py` defines. An [`Analyzer`] keeps the facts the
//! native analyses share. A [`ConstIntBound`] is an inclusive range `[min_value, max_value]`, and a
//! [`ModularSet`] is the set `coeff * x + base_` for any integer `x`.

// Python has no class for `CanonicalExpr`, `SumExpr` and `SplitExpr`, the forms of the canonical
// simplifier, or for the `RewriteSimplifierStats` that `Analyzer.rewrite_simplify_stats()`
// returns; they are here, with the analyzer they belong to.

// tvm-ffi-stubgen(begin): import-section
use crate::ir::expr::Expr;
use crate::ir::expr::ExprObj;
use std::ops::Deref;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/sym.Analyzer
/// Opaque: bytes [24, 80) of [24, 80) are not accounted for by reflected fields. Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.Analyzer"]
#[type_final]
pub struct AnalyzerObj {
    base: Object,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Analyzer {
    base: ObjectArc<AnalyzerObj>,
}

impl Deref for Analyzer {
    type Target = AnalyzerObj;
    fn deref(&self) -> &AnalyzerObj {
        &self.base
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/sym.CanonicalExpr
/// Opaque: no metadata of its own: total_size is unknown. Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.CanonicalExpr"]
pub struct CanonicalExprObj {
    base: ExprObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct CanonicalExpr {
    base: ObjectArc<CanonicalExprObj>,
}

impl Deref for CanonicalExpr {
    type Target = CanonicalExprObj;
    fn deref(&self) -> &CanonicalExprObj {
        &self.base
    }
}

impl Deref for CanonicalExprObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

tvm_ffi::impl_object_upcast!(CanonicalExpr => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/sym.ConstIntBound
/// Complete: reflected fields fill [24, 40) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.ConstIntBound"]
#[type_final]
pub struct ConstIntBoundObj {
    base: Object,
    pub min_value: i64,
    pub max_value: i64,
}

const _: () = {
    assert!(::core::mem::size_of::<ConstIntBoundObj>() == 40);
    assert!(::core::mem::align_of::<ConstIntBoundObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct ConstIntBound {
    base: ObjectArc<ConstIntBoundObj>,
}

impl Deref for ConstIntBound {
    type Target = ConstIntBoundObj;
    fn deref(&self) -> &ConstIntBoundObj {
        &self.base
    }
}

impl ConstIntBoundObj {
    pub(crate) fn new(min_value: i64, max_value: i64) -> Self {
        let base = Object::new();
        Self { base, min_value, max_value }
    }
}

impl ConstIntBound {
    /// Lossless complete-field allocation.
    pub fn new(min_value: i64, max_value: i64) -> Self {
        let obj = ConstIntBoundObj::new(min_value, max_value);
        Self { base: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/sym.ModularSet
/// Complete: reflected fields fill [24, 40) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.ModularSet"]
#[type_final]
pub struct ModularSetObj {
    base: Object,
    pub coeff: i64,
    pub base_: i64,
}

const _: () = {
    assert!(::core::mem::size_of::<ModularSetObj>() == 40);
    assert!(::core::mem::align_of::<ModularSetObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct ModularSet {
    base: ObjectArc<ModularSetObj>,
}

impl Deref for ModularSet {
    type Target = ModularSetObj;
    fn deref(&self) -> &ModularSetObj {
        &self.base
    }
}

impl ModularSetObj {
    pub(crate) fn new(coeff: i64, base_: i64) -> Self {
        let base = Object::new();
        Self { base, coeff, base_ }
    }
}

impl ModularSet {
    /// Lossless complete-field allocation.
    pub fn new(coeff: i64, base_: i64) -> Self {
        let obj = ModularSetObj::new(coeff, base_);
        Self { base: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/sym.RewriteSimplifierStats
/// Complete: reflected fields fill [24, 72) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.RewriteSimplifierStats"]
#[type_final]
pub struct RewriteSimplifierStatsObj {
    base: Object,
    pub nodes_visited: i64,
    pub constraints_entered: i64,
    pub rewrites_attempted: i64,
    pub rewrites_performed: i64,
    pub max_recursive_depth: i64,
    pub num_recursive_rewrites: i64,
}

const _: () = {
    assert!(::core::mem::size_of::<RewriteSimplifierStatsObj>() == 72);
    assert!(::core::mem::align_of::<RewriteSimplifierStatsObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct RewriteSimplifierStats {
    base: ObjectArc<RewriteSimplifierStatsObj>,
}

impl Deref for RewriteSimplifierStats {
    type Target = RewriteSimplifierStatsObj;
    fn deref(&self) -> &RewriteSimplifierStatsObj {
        &self.base
    }
}

impl RewriteSimplifierStatsObj {
    pub(crate) fn new(
        nodes_visited: i64,
        constraints_entered: i64,
        rewrites_attempted: i64,
        rewrites_performed: i64,
        max_recursive_depth: i64,
        num_recursive_rewrites: i64,
    ) -> Self {
        let base = Object::new();
        Self {
            base,
            nodes_visited,
            constraints_entered,
            rewrites_attempted,
            rewrites_performed,
            max_recursive_depth,
            num_recursive_rewrites,
        }
    }
}

impl RewriteSimplifierStats {
    /// Lossless complete-field allocation.
    pub fn new(
        nodes_visited: i64,
        constraints_entered: i64,
        rewrites_attempted: i64,
        rewrites_performed: i64,
        max_recursive_depth: i64,
        num_recursive_rewrites: i64,
    ) -> Self {
        let obj = RewriteSimplifierStatsObj::new(
            nodes_visited,
            constraints_entered,
            rewrites_attempted,
            rewrites_performed,
            max_recursive_depth,
            num_recursive_rewrites,
        );
        Self { base: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/sym.SplitExpr
/// Opaque: no metadata of its own: total_size is unknown. Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.SplitExpr"]
pub struct SplitExprObj {
    base: CanonicalExprObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct SplitExpr {
    base: ObjectArc<SplitExprObj>,
}

impl Deref for SplitExpr {
    type Target = SplitExprObj;
    fn deref(&self) -> &SplitExprObj {
        &self.base
    }
}

impl Deref for SplitExprObj {
    type Target = CanonicalExprObj;
    fn deref(&self) -> &CanonicalExprObj {
        &self.base
    }
}

tvm_ffi::impl_object_upcast!(SplitExpr => Expr, SplitExpr => CanonicalExpr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/sym.SumExpr
/// Opaque: no metadata of its own: total_size is unknown. Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.SumExpr"]
pub struct SumExprObj {
    base: CanonicalExprObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct SumExpr {
    base: ObjectArc<SumExprObj>,
}

impl Deref for SumExpr {
    type Target = SumExprObj;
    fn deref(&self) -> &SumExprObj {
        &self.base
    }
}

impl Deref for SumExprObj {
    type Target = CanonicalExprObj;
    fn deref(&self) -> &CanonicalExprObj {
        &self.base
    }
}

tvm_ffi::impl_object_upcast!(SumExpr => Expr, SumExpr => CanonicalExpr);
// tvm-ffi-stubgen(end)
