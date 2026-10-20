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

//! The context of tile instruction dispatch.
//!
//! The objects `python/tvm_tirx/tirx/tile/dispatch.py` defines. A [`DispatchContext`] gives a
//! lowering the target, the thread scope, the launch parameters, the ranges of the variables, the
//! callbacks and the state shared between dispatches.

// tvm-ffi-stubgen(begin): import-section
use crate::ir::expr::Expr;
use crate::ir::expr::Range;
use crate::ir::expr::Var;
use crate::target::Target;
use crate::tirx::thread_scope::ThreadScope;
use std::ops::Deref;
use tvm_ffi::Any;
use tvm_ffi::Array;
use tvm_ffi::FieldGetter;
use tvm_ffi::Map;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::ObjectCore;
use tvm_ffi::Result;
use tvm_ffi::String;
use tvm_ffi::object::ObjectRef;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.DispatchContext
/// Opaque: field 'launch_params' (Map[str, tuple[ir.Var, ir.Expr]]) has no native mirror. Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.DispatchContext"]
#[type_final]
pub struct DispatchContextObj {
    base: Object,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct DispatchContext {
    base: ObjectArc<DispatchContextObj>,
}

impl Deref for DispatchContext {
    type Target = DispatchContextObj;
    fn deref(&self) -> &DispatchContextObj {
        &self.base
    }
}

impl DispatchContextObj {
    pub fn target(&self) -> Result<Target> {
        FieldGetter::new(Self::type_index(), "target")?.get(self)
    }

    pub fn thread_scope(&self) -> Result<ThreadScope> {
        FieldGetter::new(Self::type_index(), "thread_scope")?.get(self)
    }

    pub fn launch_params(&self) -> Result<Any> {
        FieldGetter::new(Self::type_index(), "launch_params")?.get_any(self)
    }

    pub fn var_range_map(&self) -> Result<Map<Var, Range>> {
        FieldGetter::new(Self::type_index(), "var_range_map")?.get(self)
    }

    pub fn alloc_only(&self) -> Result<bool> {
        FieldGetter::new(Self::type_index(), "alloc_only")?.get(self)
    }

    pub fn callbacks(&self) -> Result<Map<String, ObjectRef>> {
        FieldGetter::new(Self::type_index(), "callbacks")?.get(self)
    }

    pub fn shared_state(&self) -> Result<Map<String, ObjectRef>> {
        FieldGetter::new(Self::type_index(), "shared_state")?.get(self)
    }

    pub fn inter(&self) -> Result<Map<String, Array<Expr>>> {
        FieldGetter::new(Self::type_index(), "inter")?.get(self)
    }

    pub fn intra(&self) -> Result<Map<String, Array<Expr>>> {
        FieldGetter::new(Self::type_index(), "intra")?.get(self)
    }

    pub fn scope_kind(&self) -> Result<String> {
        FieldGetter::new(Self::type_index(), "scope_kind")?.get(self)
    }
}
// tvm-ffi-stubgen(end)
