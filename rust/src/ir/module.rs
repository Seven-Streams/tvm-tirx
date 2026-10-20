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

//! The module: the unit of compilation.
//!
//! The objects `python/tvm_tirx/ir/module.py` defines. An [`IRModule`] maps global variables to
//! functions, and holds the module attributes, the module-level information and the names of its
//! global variables.

// tvm-ffi-stubgen(begin): import-section
use crate::ir::attrs::DictAttrs;
use crate::ir::expr::GlobalVar;
use crate::ir::function::BaseFunc;
use crate::ir::global_info::GlobalInfo;
use std::ops::Deref;
use tvm_ffi::Array;
use tvm_ffi::Map;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::String;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.IRModule
/// Complete: reflected fields fill [24, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.IRModule"]
#[type_final]
pub struct IRModuleObj {
    base: Object,
    pub functions: Map<GlobalVar, BaseFunc>,
    pub attrs: DictAttrs,
    pub global_infos: Map<String, Array<GlobalInfo>>,
    pub global_var_map: Map<String, GlobalVar>,
}

const _: () = {
    assert!(::core::mem::size_of::<IRModuleObj>() == 56);
    assert!(::core::mem::align_of::<IRModuleObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IRModule {
    base: ObjectArc<IRModuleObj>,
}

impl Deref for IRModule {
    type Target = IRModuleObj;
    fn deref(&self) -> &IRModuleObj {
        &self.base
    }
}

impl IRModuleObj {
    pub(crate) fn new(
        functions: Map<GlobalVar, BaseFunc>,
        attrs: DictAttrs,
        global_infos: Map<String, Array<GlobalInfo>>,
        global_var_map: Map<String, GlobalVar>,
    ) -> Self {
        let base = Object::new();
        Self { base, functions, attrs, global_infos, global_var_map }
    }
}

impl IRModule {
    /// Lossless complete-field allocation.
    pub fn new(
        functions: Map<GlobalVar, BaseFunc>,
        attrs: DictAttrs,
        global_infos: Map<String, Array<GlobalInfo>>,
        global_var_map: Map<String, GlobalVar>,
    ) -> Self {
        let obj = IRModuleObj::new(functions, attrs, global_infos, global_var_map);
        Self { base: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)
