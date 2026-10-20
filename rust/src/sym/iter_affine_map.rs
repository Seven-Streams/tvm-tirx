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

//! Iterator maps: index expressions decomposed into splits of iterators.
//!
//! The objects `python/tvm_tirx/sym/iter_affine_map.py` defines. An [`IterMark`] is an iterator
//! `source` with its `extent`; an [`IterSplitExpr`] is `(source // lower_factor) % extent * scale`;
//! an [`IterSumExpr`] is a sum of splits plus `base_`. An [`IterMapResult`] holds the detected
//! `indices`, the `errors` and the `padding_predicate`.

// tvm-ffi-stubgen(begin): import-section
use crate::ir::expr::Expr;
use crate::ir::expr::ExprObj;
use std::ops::Deref;
use tvm_ffi::Array;
use tvm_ffi::FieldGetter;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::ObjectCore;
use tvm_ffi::Result;
use tvm_ffi::String;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/sym.IterMapExpr
/// Opaque: no metadata of its own: total_size is unknown. Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.IterMapExpr"]
pub struct IterMapExprObj {
    base: ExprObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IterMapExpr {
    base: ObjectArc<IterMapExprObj>,
}

impl Deref for IterMapExpr {
    type Target = IterMapExprObj;
    fn deref(&self) -> &IterMapExprObj {
        &self.base
    }
}

impl Deref for IterMapExprObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

tvm_ffi::impl_object_upcast!(IterMapExpr => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/sym.IterMapResult
/// Complete: reflected fields fill [24, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.IterMapResult"]
#[type_final]
pub struct IterMapResultObj {
    base: Object,
    pub indices: Array<IterSumExpr>,
    pub errors: Array<String>,
    pub padding_predicate: Option<Expr>,
}

const _: () = {
    assert!(::core::mem::size_of::<IterMapResultObj>() == 48);
    assert!(::core::mem::align_of::<IterMapResultObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IterMapResult {
    base: ObjectArc<IterMapResultObj>,
}

impl Deref for IterMapResult {
    type Target = IterMapResultObj;
    fn deref(&self) -> &IterMapResultObj {
        &self.base
    }
}

impl IterMapResultObj {
    pub(crate) fn new(
        indices: Array<IterSumExpr>,
        errors: Array<String>,
        padding_predicate: Option<Expr>,
    ) -> Self {
        let base = Object::new();
        Self { base, indices, errors, padding_predicate }
    }
}

impl IterMapResult {
    /// Lossless complete-field allocation.
    pub fn new(
        indices: Array<IterSumExpr>,
        errors: Array<String>,
        padding_predicate: Option<Expr>,
    ) -> Self {
        let obj = IterMapResultObj::new(indices, errors, padding_predicate);
        Self { base: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/sym.IterMark
/// Complete: reflected fields fill [24, 40) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.IterMark"]
#[type_final]
pub struct IterMarkObj {
    base: Object,
    pub source: Expr,
    pub extent: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<IterMarkObj>() == 40);
    assert!(::core::mem::align_of::<IterMarkObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IterMark {
    base: ObjectArc<IterMarkObj>,
}

impl Deref for IterMark {
    type Target = IterMarkObj;
    fn deref(&self) -> &IterMarkObj {
        &self.base
    }
}

impl IterMarkObj {
    pub(crate) fn new(source: Expr, extent: Expr) -> Self {
        let base = Object::new();
        Self { base, source, extent }
    }
}

impl IterMark {
    /// Lossless complete-field allocation.
    pub fn new(source: Expr, extent: Expr) -> Self {
        let obj = IterMarkObj::new(source, extent);
        Self { base: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/sym.IterSplitExpr
/// Opaque: parent 'sym.IterMapExpr' is opaque (layout-unknown). Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.IterSplitExpr"]
#[type_final]
pub struct IterSplitExprObj {
    base: IterMapExprObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IterSplitExpr {
    base: ObjectArc<IterSplitExprObj>,
}

impl Deref for IterSplitExpr {
    type Target = IterSplitExprObj;
    fn deref(&self) -> &IterSplitExprObj {
        &self.base
    }
}

impl Deref for IterSplitExprObj {
    type Target = IterMapExprObj;
    fn deref(&self) -> &IterMapExprObj {
        &self.base
    }
}

impl IterSplitExprObj {
    pub fn source(&self) -> Result<IterMark> {
        FieldGetter::new(Self::type_index(), "source")?.get(self)
    }

    pub fn lower_factor(&self) -> Result<Expr> {
        FieldGetter::new(Self::type_index(), "lower_factor")?.get(self)
    }

    pub fn extent(&self) -> Result<Expr> {
        FieldGetter::new(Self::type_index(), "extent")?.get(self)
    }

    pub fn scale(&self) -> Result<Expr> {
        FieldGetter::new(Self::type_index(), "scale")?.get(self)
    }
}

tvm_ffi::impl_object_upcast!(IterSplitExpr => Expr, IterSplitExpr => IterMapExpr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/sym.IterSumExpr
/// Opaque: parent 'sym.IterMapExpr' is opaque (layout-unknown). Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.IterSumExpr"]
#[type_final]
pub struct IterSumExprObj {
    base: IterMapExprObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IterSumExpr {
    base: ObjectArc<IterSumExprObj>,
}

impl Deref for IterSumExpr {
    type Target = IterSumExprObj;
    fn deref(&self) -> &IterSumExprObj {
        &self.base
    }
}

impl Deref for IterSumExprObj {
    type Target = IterMapExprObj;
    fn deref(&self) -> &IterMapExprObj {
        &self.base
    }
}

impl IterSumExprObj {
    pub fn args(&self) -> Result<Array<IterSplitExpr>> {
        FieldGetter::new(Self::type_index(), "args")?.get(self)
    }

    pub fn base_(&self) -> Result<Expr> {
        FieldGetter::new(Self::type_index(), "base")?.get(self)
    }
}

tvm_ffi::impl_object_upcast!(IterSumExpr => Expr, IterSumExpr => IterMapExpr);
// tvm-ffi-stubgen(end)
