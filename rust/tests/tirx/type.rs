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

//! Tests of `tirx::type`.

use tvm_tirx::ir::TypeObj;
use tvm_tirx::tirx;

use crate::common::{assert_type_contracts, global, serial, structural_equal};
use crate::ir;

#[test]
fn generated_types_match_the_registry() {
    let _serial = serial();
    assert_type_contracts!(true; tirx::TensorMapTypeObj: TypeObj);
}

#[test]
fn native_ir_objects_agree_with_getters_and_rust_allocation() {
    let _serial = serial();
    let ir = ir();

    let map_type: tirx::TensorMapType =
        global("tirx.TensorMapType").call_tuple((ir.loc.clone(),)).unwrap().try_into().unwrap();
    assert!(structural_equal(&tirx::TensorMapType::new(ir.loc.clone()), &map_type));
}
