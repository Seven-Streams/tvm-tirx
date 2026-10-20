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

//! Index maps.
//!
//! The objects `python/tvm_tirx/tirx/index_map.py` defines. An [`IndexMap`] maps its
//! `initial_indices` to its `final_indices`, as `(i, j) -> (j, i)` does, and may hold its inverse.

// tvm-ffi-stubgen(begin): import-section
use crate::ir::expr::Expr;
use crate::ir::expr::Var;
use std::ops::Deref;
use tvm_ffi::Array;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::object::ObjectRef;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.IndexMap
/// Complete: reflected fields fill [24, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.IndexMap"]
#[type_final]
pub struct IndexMapObj {
    base: Object,
    pub initial_indices: Array<Var>,
    pub final_indices: Array<Expr>,
    pub inverse_index_map: Option<ObjectRef>,
}

const _: () = {
    assert!(::core::mem::size_of::<IndexMapObj>() == 48);
    assert!(::core::mem::align_of::<IndexMapObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IndexMap {
    base: ObjectArc<IndexMapObj>,
}

impl Deref for IndexMap {
    type Target = IndexMapObj;
    fn deref(&self) -> &IndexMapObj {
        &self.base
    }
}

impl IndexMapObj {
    pub(crate) fn new(
        initial_indices: Array<Var>,
        final_indices: Array<Expr>,
        inverse_index_map: Option<ObjectRef>,
    ) -> Self {
        let base = Object::new();
        Self { base, initial_indices, final_indices, inverse_index_map }
    }
}

impl IndexMap {
    /// Lossless complete-field allocation.
    pub fn new(
        initial_indices: Array<Var>,
        final_indices: Array<Expr>,
        inverse_index_map: Option<ObjectRef>,
    ) -> Self {
        let obj = IndexMapObj::new(initial_indices, final_indices, inverse_index_map);
        Self { base: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)
