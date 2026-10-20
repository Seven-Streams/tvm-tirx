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

//! Layouts: how the indices of a tensor map to memory and to threads.
//!
//! The objects `python/tvm_tirx/tirx/layout.py` defines. [`Layout`] is the base. A [`TileLayout`]
//! maps indices through the [`Iter`]s of its `shard` (distinct elements) and `replica` (copies),
//! each an extent and a stride over an [`Axis`]. A [`ComposeLayout`] applies an XOR swizzle to the
//! memory coordinate of a tile layout.

// tvm-ffi-stubgen(begin): import-section
use crate::ir::expr::Expr;
use std::ops::Deref;
use tvm_ffi::Array;
use tvm_ffi::FieldGetter;
use tvm_ffi::Map;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::ObjectCore;
use tvm_ffi::Result;

/// Header-only stand-in for the builtin `ffi.Enum`; it only carries the ancestor depth.
#[allow(dead_code)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ffi.Enum"]
struct FfiEnumObj {
    base: Object,
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.Axis
/// Opaque: parent 'ffi.Enum' is opaque (no-mirror). Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.Axis"]
#[type_final]
pub struct AxisObj {
    base: FfiEnumObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Axis {
    base: ObjectArc<AxisObj>,
}

impl Deref for Axis {
    type Target = AxisObj;
    fn deref(&self) -> &AxisObj {
        &self.base
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.Iter
/// Complete: reflected fields fill [24, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.Iter"]
#[type_final]
pub struct IterObj {
    base: Object,
    pub extent: Expr,
    pub stride: Expr,
    pub axis: Axis,
}

const _: () = {
    assert!(::core::mem::size_of::<IterObj>() == 48);
    assert!(::core::mem::align_of::<IterObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Iter {
    base: ObjectArc<IterObj>,
}

impl Deref for Iter {
    type Target = IterObj;
    fn deref(&self) -> &IterObj {
        &self.base
    }
}

impl IterObj {
    pub(crate) fn new(extent: Expr, stride: Expr, axis: Axis) -> Self {
        let base = Object::new();
        Self { base, extent, stride, axis }
    }
}

impl Iter {
    /// Lossless complete-field allocation.
    pub fn new(extent: Expr, stride: Expr, axis: Axis) -> Self {
        let obj = IterObj::new(extent, stride, axis);
        Self { base: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.Layout
/// Opaque: no metadata of its own: total_size is unknown. Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.Layout"]
pub struct LayoutObj {
    base: Object,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Layout {
    base: ObjectArc<LayoutObj>,
}

impl Deref for Layout {
    type Target = LayoutObj;
    fn deref(&self) -> &LayoutObj {
        &self.base
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.ComposeLayout
/// Opaque: parent 'tirx.Layout' is opaque (layout-unknown). Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.ComposeLayout"]
#[type_final]
pub struct ComposeLayoutObj {
    base: LayoutObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct ComposeLayout {
    base: ObjectArc<ComposeLayoutObj>,
}

impl Deref for ComposeLayout {
    type Target = ComposeLayoutObj;
    fn deref(&self) -> &ComposeLayoutObj {
        &self.base
    }
}

impl Deref for ComposeLayoutObj {
    type Target = LayoutObj;
    fn deref(&self) -> &LayoutObj {
        &self.base
    }
}

impl ComposeLayoutObj {
    pub fn per_element(&self) -> Result<i64> {
        FieldGetter::new(Self::type_index(), "per_element")?.get(self)
    }

    pub fn swizzle_len(&self) -> Result<i64> {
        FieldGetter::new(Self::type_index(), "swizzle_len")?.get(self)
    }

    pub fn atom_len(&self) -> Result<i64> {
        FieldGetter::new(Self::type_index(), "atom_len")?.get(self)
    }

    pub fn swizzle_inner(&self) -> Result<bool> {
        FieldGetter::new(Self::type_index(), "swizzle_inner")?.get(self)
    }

    pub fn inner_mask(&self) -> Result<i64> {
        FieldGetter::new(Self::type_index(), "inner_mask")?.get(self)
    }

    pub fn outer_mask(&self) -> Result<i64> {
        FieldGetter::new(Self::type_index(), "outer_mask")?.get(self)
    }

    pub fn tile_layout(&self) -> Result<TileLayout> {
        FieldGetter::new(Self::type_index(), "tile_layout")?.get(self)
    }
}

tvm_ffi::impl_object_upcast!(ComposeLayout => Layout);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.TileLayout
/// Opaque: parent 'tirx.Layout' is opaque (layout-unknown). Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.TileLayout"]
#[type_final]
pub struct TileLayoutObj {
    base: LayoutObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct TileLayout {
    base: ObjectArc<TileLayoutObj>,
}

impl Deref for TileLayout {
    type Target = TileLayoutObj;
    fn deref(&self) -> &TileLayoutObj {
        &self.base
    }
}

impl Deref for TileLayoutObj {
    type Target = LayoutObj;
    fn deref(&self) -> &LayoutObj {
        &self.base
    }
}

impl TileLayoutObj {
    pub fn shard(&self) -> Result<Array<Iter>> {
        FieldGetter::new(Self::type_index(), "shard")?.get(self)
    }

    pub fn replica(&self) -> Result<Array<Iter>> {
        FieldGetter::new(Self::type_index(), "replica")?.get(self)
    }

    pub fn offset(&self) -> Result<Map<Axis, Expr>> {
        FieldGetter::new(Self::type_index(), "offset")?.get(self)
    }
}

tvm_ffi::impl_object_upcast!(TileLayout => Layout);
// tvm-ffi-stubgen(end)
