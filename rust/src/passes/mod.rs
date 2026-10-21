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

//! TIRx passes reimplemented in Rust.
//!
//! Each submodule mirrors the C++ pass of the same name in `src/tirx/transform/`. The pass
//! drivers stay native: `function_pass` hands the per-function rewrite to
//! `tirx.transform.CreateFunctionPass`, so which functions are visited and how a removed
//! function is dropped are exactly the C++ behavior. `tests/test_pass_parity.py` runs every
//! Rust pass next to its C++ counterpart on recorded inputs and requires the same results.

use tvm_ffi::{Any, AnyView, Function as FfiFunction, Result};

use crate::ir::transform::Pass;

pub mod analyzer;
pub mod bind_target;
pub mod common_subexpr_elim;
pub mod convert_ssa;
pub mod data_type_rewriter;
pub mod decorate_device_scope;
pub mod dtype_conversion;
pub mod force_narrow_index_to_i32;
pub mod function_utils;
pub mod hoist_if;
pub mod lower_intrin;
pub mod narrow_datatype;
pub mod remove_no_op;
pub mod renormalize_split_pattern;
pub mod side_effect;
pub mod skip_assert;
pub mod stmt_simplify;
pub mod substitute_dtype;
pub mod unroll_loop;
pub mod unsupported_dtype_legalize;
pub mod util;
pub mod verify_memory;

pub use util::{function_pass, module_pass, native_pass, sequential};

/// Every Rust pass factory, by the name of the C++ factory it mirrors (`tirx.transform.<Name>`).
pub fn factories() -> Vec<(&'static str, FfiFunction)> {
    vec![
        ("AnnotateEntryFunc", nullary(function_utils::annotate_entry_func)),
        ("BF16ComputeLegalize", nullary(unsupported_dtype_legalize::bf16_compute_legalize)),
        ("BF16StorageLegalize", nullary(unsupported_dtype_legalize::bf16_storage_legalize)),
        (
            "BindTarget",
            FfiFunction::from_packed(|args: &[AnyView]| -> Result<Any> {
                Ok(Any::from(bind_target::bind_target(crate::target::Target::try_from(args[0])?)?))
            }),
        ),
        ("CommonSubexprElim", nullary(common_subexpr_elim::common_subexpr_elim)),
        ("ConvertSSA", nullary(convert_ssa::convert_ssa)),
        ("DecorateDeviceScope", nullary(decorate_device_scope::decorate_device_scope)),
        (
            "FP8ComputeLegalize",
            FfiFunction::from_packed(|args: &[AnyView]| -> Result<Any> {
                let promote = tvm_ffi::String::try_from(args[0])?;
                Ok(Any::from(unsupported_dtype_legalize::fp8_compute_legalize(promote.as_str())?))
            }),
        ),
        ("FP8StorageLegalize", nullary(unsupported_dtype_legalize::fp8_storage_legalize)),
        (
            "Filter",
            FfiFunction::from_packed(|args: &[AnyView]| -> Result<Any> {
                Ok(Any::from(function_utils::filter(FfiFunction::try_from(args[0])?)?))
            }),
        ),
        ("ForceNarrowIndexToInt32", nullary(force_narrow_index_to_i32::force_narrow_index_to_i32)),
        ("HoistIf", nullary(hoist_if::hoist_if)),
        ("InsertHoistedIf", nullary(hoist_if::insert_hoisted_if)),
        ("LowerIntrin", nullary(lower_intrin::lower_intrin)),
        (
            "NarrowDataType",
            FfiFunction::from_packed(|args: &[AnyView]| -> Result<Any> {
                let target_bits = i64::try_from(args[0])? as i32;
                Ok(Any::from(narrow_datatype::narrow_datatype(target_bits)?))
            }),
        ),
        ("RemoveNoOp", nullary(remove_no_op::remove_no_op)),
        ("RenormalizeSplitPattern", nullary(renormalize_split_pattern::renormalize_split_pattern)),
        ("SkipAssert", nullary(skip_assert::skip_assert)),
        ("StmtSimplify", nullary(stmt_simplify::stmt_simplify)),
        ("UnrollLoop", nullary(unroll_loop::unroll_loop)),
        ("VerifyMemory", nullary(verify_memory::verify_memory)),
    ]
}

fn nullary(factory: fn() -> Result<Pass>) -> FfiFunction {
    FfiFunction::from_packed(move |_args: &[AnyView]| -> Result<Any> { Ok(Any::from(factory()?)) })
}

/// Register each Rust pass factory as the global function `rust.tirx.transform.<Name>`.
pub fn register_globals() -> Result<()> {
    for (name, factory) in factories() {
        FfiFunction::register_global(&format!("rust.tirx.transform.{name}"), factory)?;
    }
    Ok(())
}
