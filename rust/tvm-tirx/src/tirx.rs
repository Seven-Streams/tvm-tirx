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

//! Tensor programs: functions, tensor types, index maps, layouts and thread scopes.
//!
//! Laid out like `python/tvm_tirx/tirx/`, on top of [`ir`](crate::ir). [`Function`] is a tensor
//! function, [`TensorType`] the type of a tensor, and [`IndexMap`] and the [`layout`] types map
//! indices. [`tile`] holds the tile instructions, [`transform`] the TIRx passes and [`analysis`]
//! the analyses of tensor-level IR.

// An object registered under `tirx` that no file has a block for yet is generated here, to be moved
// to the file of the Python module that defines it.
// tvm-ffi-stubgen(prefix): tirx

pub mod analysis;
pub mod function;
pub mod index_map;
pub mod layout;
pub mod op;
pub mod op_attr;
pub mod tensor;
pub mod thread_scope;
pub mod tile;
pub mod transform;
pub mod r#type;

pub use function::*;
pub use index_map::*;
pub use layout::*;
pub use op::*;
pub use tensor::*;
pub use thread_scope::*;
pub use r#type::*;

// tvm-ffi-stubgen(begin): import-section
use crate::sym::ConstraintContext;
use std::ops::Deref;
use tvm_ffi::FieldGetter;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::ObjectCore;
use tvm_ffi::Result;

/// Header-only stand-in for the builtin `ffi.StructuralMutator`; it only carries the ancestor depth.
#[allow(dead_code)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ffi.StructuralMutator"]
struct FfiStructuralMutatorObj {
    base: Object,
}

/// Header-only stand-in for the builtin `ffi.StructuralVisitor`; it only carries the ancestor depth.
#[allow(dead_code)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ffi.StructuralVisitor"]
struct FfiStructuralVisitorObj {
    base: Object,
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.StructuralMutatorWithConstraintContext
/// Opaque: parent 'ffi.StructuralMutator' is opaque (no-mirror). Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.StructuralMutatorWithConstraintContext"]
pub struct StructuralMutatorWithConstraintContextObj {
    base: FfiStructuralMutatorObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct StructuralMutatorWithConstraintContext {
    base: ObjectArc<StructuralMutatorWithConstraintContextObj>,
}

impl Deref for StructuralMutatorWithConstraintContext {
    type Target = StructuralMutatorWithConstraintContextObj;
    fn deref(&self) -> &StructuralMutatorWithConstraintContextObj {
        &self.base
    }
}

impl StructuralMutatorWithConstraintContextObj {
    pub fn constraints(&self) -> Result<ConstraintContext> {
        FieldGetter::new(Self::type_index(), "constraints")?.get(self)
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.StructuralVisitorWithConstraintContext
/// Opaque: parent 'ffi.StructuralVisitor' is opaque (no-mirror). Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.StructuralVisitorWithConstraintContext"]
pub struct StructuralVisitorWithConstraintContextObj {
    base: FfiStructuralVisitorObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct StructuralVisitorWithConstraintContext {
    base: ObjectArc<StructuralVisitorWithConstraintContextObj>,
}

impl Deref for StructuralVisitorWithConstraintContext {
    type Target = StructuralVisitorWithConstraintContextObj;
    fn deref(&self) -> &StructuralVisitorWithConstraintContextObj {
        &self.base
    }
}

impl StructuralVisitorWithConstraintContextObj {
    pub fn constraints(&self) -> Result<ConstraintContext> {
        FieldGetter::new(Self::type_index(), "constraints")?.get(self)
    }
}
// tvm-ffi-stubgen(end)
