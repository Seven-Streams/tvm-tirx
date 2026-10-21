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

//! Code generation: IR modules compiled into runtime modules.
//!
//! Laid out like `python/tvm_tirx/codegen/`: the functions of its `__init__.py` are here, and the
//! queries of the LLVM backend are in [`llvm`]. The code generator of one target kind is
//! [`crate::target::build`].

// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::Module
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): tvm_ffi::String
// tvm-ffi-stubgen(import-object): crate::ir::module::IRModule
// tvm-ffi-stubgen(import-object): crate::target::Target

pub mod llvm;

// tvm-ffi-stubgen(begin): import-section
use crate::ir::module::IRModule;
use crate::target::Target;
use tvm_ffi::Module;
use tvm_ffi::Result;
use tvm_ffi::String;
// tvm-ffi-stubgen(end)

/// Compile `module` into a runtime module with the code generator of the kind of `target`
/// (`codegen.Build`).
pub fn build_module(module: impl Into<IRModule>, target: impl Into<Target>) -> Result<Module> {
    tvm_ffi::cached_global_func!("codegen.Build")
        .call_tuple((&module.into(), &target.into()))?
        .try_into()
}

/// Whether the LLVM target `target`, or the current target, has the CPU feature `feature`;
/// `false` for a target of another kind; an error when TIRx is built without LLVM, whose backend
/// registers it (`codegen.target_has_feature`).
pub fn target_has_feature(feature: &str, target: Option<&Target>) -> Result<bool> {
    tvm_ffi::Function::get_global("codegen.target_has_feature")?
        .call_tuple((String::from(feature), target.cloned()))?
        .try_into()
}
