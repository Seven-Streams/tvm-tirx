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

//! The LLVM backend: intrinsics, the host system and the CPUs and features of LLVM targets.
//!
//! The functions of `python/tvm_tirx/codegen/llvm.py`, and the host queries
//! `python/tvm_tirx/target/detect_target.py` makes. A query that takes an optional target falls
//! back to the current one, and answers nothing for a target that is not `llvm`.

// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::Array
// tvm-ffi-stubgen(import-object): tvm_ffi::Map
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): tvm_ffi::String
// tvm-ffi-stubgen(import-object): crate::ir::expr::IntImm
// tvm-ffi-stubgen(import-object): crate::target::Target

// tvm-ffi-stubgen(begin): import-section
use crate::ir::expr::IntImm;
use crate::target::Target;
use tvm_ffi::Array;
use tvm_ffi::Map;
use tvm_ffi::Result;
use tvm_ffi::String;
// tvm-ffi-stubgen(end)

// The LLVM backend registers these functions only when TIRx is built with LLVM: they are looked up
// when called, and their absence is an error.

/// The id of the LLVM intrinsic `name`, 0 when there is none
/// (`codegen.llvm_lookup_intrinsic_id`).
pub fn lookup_intrinsic_id(name: &str) -> Result<i64> {
    tvm_ffi::Function::get_global("codegen.llvm_lookup_intrinsic_id")?
        .call_tuple((String::from(name),))?
        .try_into()
}

/// The name of the LLVM intrinsic `intrin_id` (`codegen.llvm_get_intrinsic_name`).
pub fn get_intrinsic_name(intrin_id: i64) -> Result<String> {
    tvm_ffi::Function::get_global("codegen.llvm_get_intrinsic_name")?
        .call_tuple((intrin_id,))?
        .try_into()
}

/// The vendor of the host x86 CPU: `intel`, `amd`, `unknown`, or `unimplemented` on other CPUs
/// (`codegen.llvm_get_system_x86_vendor`).
pub fn get_system_x86_vendor() -> Result<String> {
    tvm_ffi::Function::get_global("codegen.llvm_get_system_x86_vendor")?.call_tuple(())?.try_into()
}

/// The default LLVM target triple of the host (`codegen.llvm_get_system_triple`).
pub fn get_system_triple() -> Result<String> {
    tvm_ffi::Function::get_global("codegen.llvm_get_system_triple")?.call_tuple(())?.try_into()
}

/// The LLVM name of the host CPU (`codegen.llvm_get_system_cpu`).
pub fn get_system_cpu() -> Result<String> {
    tvm_ffi::Function::get_global("codegen.llvm_get_system_cpu")?.call_tuple(())?.try_into()
}

/// The LLVM targets LLVM was built with (`codegen.llvm_get_targets`).
pub fn get_targets() -> Result<Array<String>> {
    tvm_ffi::Function::get_global("codegen.llvm_get_targets")?.call_tuple(())?.try_into()
}

/// The CPUs of the `-mtriple` of `target` (`codegen.llvm_get_cpu_archlist`).
pub fn get_cpu_archlist(target: Option<&Target>) -> Result<Array<String>> {
    tvm_ffi::Function::get_global("codegen.llvm_get_cpu_archlist")?
        .call_tuple((target.cloned(),))?
        .try_into()
}

/// The CPU features of the `-mtriple`, `-mcpu` and `-mattr` of `target`
/// (`codegen.llvm_get_cpu_features`).
pub fn get_cpu_features(target: Option<&Target>) -> Result<Map<String, String>> {
    tvm_ffi::Function::get_global("codegen.llvm_get_cpu_features")?
        .call_tuple((target.cloned(),))?
        .try_into()
}

/// Whether the CPU of `target` has the feature `feature` (`codegen.llvm_cpu_has_feature`).
pub fn cpu_has_feature(feature: &str, target: Option<&Target>) -> Result<bool> {
    tvm_ffi::Function::get_global("codegen.llvm_cpu_has_feature")?
        .call_tuple((String::from(feature), target.cloned()))?
        .try_into()
}

/// The vector width of the CPU of `target`, in bits; -1 for a target that is not `llvm`
/// (`codegen.llvm_get_vector_width`).
pub fn get_vector_width(target: Option<&Target>) -> Result<i32> {
    tvm_ffi::Function::get_global("codegen.llvm_get_vector_width")?
        .call_tuple((target.cloned(),))?
        .try_into()
}

/// Whether LLVM knows the CPU `cpu` for the target triple `triple` (`codegen.llvm_is_valid_cpu`).
pub fn is_valid_cpu(cpu: &str, triple: &str) -> Result<bool> {
    tvm_ffi::Function::get_global("codegen.llvm_is_valid_cpu")?
        .call_tuple((String::from(cpu), String::from(triple)))?
        .try_into()
}

/// The major version of LLVM (`codegen.llvm_version_major`).
pub fn version_major() -> Result<i32> {
    tvm_ffi::Function::get_global("codegen.llvm_version_major")?.call_tuple(())?.try_into()
}

/// Whether LLVM can create a target machine for the target string `target`, such as
/// `"llvm -mtriple=aarch64-linux-gnu"` (`codegen.llvm_target_enabled`).
pub fn target_enabled(target: &str) -> Result<bool> {
    tvm_ffi::Function::get_global("codegen.llvm_target_enabled")?
        .call_tuple((String::from(target),))?
        .try_into()
}

/// The default LLVM target triple of the host (`tvm.codegen.llvm.GetDefaultTargetTriple`).
pub fn get_default_target_triple() -> Result<String> {
    tvm_ffi::Function::get_global("tvm.codegen.llvm.GetDefaultTargetTriple")?
        .call_tuple(())?
        .try_into()
}

/// The LLVM target triple of the running process (`tvm.codegen.llvm.GetProcessTriple`).
pub fn get_process_triple() -> Result<String> {
    tvm_ffi::Function::get_global("tvm.codegen.llvm.GetProcessTriple")?.call_tuple(())?.try_into()
}

/// The LLVM name of the host CPU (`tvm.codegen.llvm.GetHostCPUName`).
pub fn get_host_cpu_name() -> Result<String> {
    tvm_ffi::Function::get_global("tvm.codegen.llvm.GetHostCPUName")?.call_tuple(())?.try_into()
}

/// The features of the host CPU, each with a boolean saying whether it is enabled; empty when
/// LLVM cannot tell (`tvm.codegen.llvm.GetHostCPUFeatures`).
pub fn get_host_cpu_features() -> Result<Map<String, IntImm>> {
    tvm_ffi::Function::get_global("tvm.codegen.llvm.GetHostCPUFeatures")?.call_tuple(())?.try_into()
}
