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

//! Tensor types.
//!
//! The objects `python/tvm_tirx/tirx/tensor.py` defines. A [`TensorType`] is the element type,
//! memory scope, shape, data alignment and optional layout of a tensor.

// `custom-new` reserves `new` for a constructor with the native one's semantics (defaults); the
// generated `from_complete_fields` allocates the fields as given.
// tvm-ffi-stubgen(custom-new): tirx.TensorType

// tvm-ffi-stubgen(begin): import-section
use crate::ir::expr::Expr;
use crate::ir::location::Location;
use crate::ir::r#type::PrimType;
use crate::ir::r#type::Type;
use crate::ir::r#type::TypeObj;
use crate::runtime::MemoryScope;
use crate::tirx::layout::Layout;
use std::ops::Deref;
use tvm_ffi::Array;
use tvm_ffi::ObjectArc;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.TensorType
/// Complete: reflected fields fill [32, 72) exactly (alignment padding [60, 64)).
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.TensorType"]
#[type_final]
pub struct TensorTypeObj {
    base: TypeObj,
    pub dtype: PrimType,
    pub scope: MemoryScope,
    pub shape: Array<Expr>,
    pub data_alignment: i32,
    pub layout: Option<Layout>,
}

const _: () = {
    assert!(::core::mem::size_of::<TensorTypeObj>() == 72);
    assert!(::core::mem::align_of::<TensorTypeObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct TensorType {
    base: ObjectArc<TensorTypeObj>,
}

impl Deref for TensorType {
    type Target = TensorTypeObj;
    fn deref(&self) -> &TensorTypeObj {
        &self.base
    }
}

impl Deref for TensorTypeObj {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base
    }
}

impl TensorTypeObj {
    pub(crate) fn new(
        loc: Location,
        dtype: PrimType,
        scope: MemoryScope,
        shape: Array<Expr>,
        data_alignment: i32,
        layout: Option<Layout>,
    ) -> Self {
        let base = TypeObj::new(loc);
        Self { base, dtype, scope, shape, data_alignment, layout }
    }
}

impl TensorType {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        loc: Location,
        dtype: PrimType,
        scope: MemoryScope,
        shape: Array<Expr>,
        data_alignment: i32,
        layout: Option<Layout>,
    ) -> Self {
        let obj = TensorTypeObj::new(loc, dtype, scope, shape, data_alignment, layout);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(TensorType => Type);
// tvm-ffi-stubgen(end)
