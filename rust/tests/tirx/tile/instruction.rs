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

//! Tests of `tirx::tile::instruction`: the tensor regions and validators of tile instructions.

use tvm_tirx::ir::{AttrsObj, Expr, Op, PrimType, Location, UNKNOWN_LOC};
use tvm_tirx::tirx::tile::{
    as_tensor_region, configure_tensor_instruction, ScopeAttrs, ScopeAttrsObj, TrnAttrs,
    TrnAttrsObj,
};
use tvm_tirx::tirx::{self, TensorType};
use tvm_tirx::tvm_ffi::{Array, Optional};
use crate::common::{assert_fields, assert_type_contracts, global, serial, structural_equal};
use crate::{int32, memory_scope, s, var};

#[test]
fn generated_types_match_the_registry() {
    let _serial = serial();
    assert_type_contracts!(true; ScopeAttrsObj: AttrsObj, TrnAttrsObj: AttrsObj);
}

#[test]
fn tensor_regions() {
    let _serial = serial();
    let scope = memory_scope();
    let ty =
        TensorType::new([int32(4)], PrimType::float(32), &scope, 0, None, UNKNOWN_LOC).unwrap();
    let tensor = var("T", &ty);
    let load: Expr = global("ir.TensorLoad")
        .call_tuple((Expr::from(&tensor), Array::new(vec![int32(1)]), Location::unknown()))
        .unwrap()
        .try_into()
        .unwrap();
    let region = as_tensor_region(&load).unwrap();
    assert!(structural_equal(&as_tensor_region(&region).unwrap(), &region));
}

#[test]
fn tensor_instructions() {
    let _serial = serial();
    Op::register("rust.tests.tirx.instruction", "").unwrap();
    let instruction = Op::get("rust.tests.tirx.instruction").unwrap();
    configure_tensor_instruction(&instruction, |_call| Ok(())).unwrap();
    assert!(configure_tensor_instruction(&instruction, |_call| Ok(())).is_err());
}

#[test]
fn rust_allocated_data_objects_are_read_by_native_getters() {
    let _serial = serial();
    let kernel = tirx::ThreadScope::get("cuda.kernel").unwrap();
    let scope_attrs = ScopeAttrs::new(kernel.clone());
    assert_fields!(scope_attrs; scope = kernel);

    let axes: Array<i64> = Array::new(vec![0, 2]);
    let thread = tirx::ThreadScope::get("cuda.thread").unwrap();
    let trn = TrnAttrs::new(
        thread.clone(),
        Optional::some(8i64),
        s("op"),
        s("add"),
        s("x"),
        s("y"),
        axes.clone(),
        true,
        false,
    );
    assert_fields!(trn;
        scope = thread, opcode = s("op"), reduce_op = s("add"), op0 = s("x"), op1 = s("y"),
        axes = axes, negate = true, reverse1 = false,
    );
    assert_fields!(trn; optional; max_inst_size = Some(8i64));
}
