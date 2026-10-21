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

//! Tests of `tirx::op`.

use tvm_tirx::tirx::{gpu_thread_return, logaddexp, reinterpret};

use super::{loc, prim_type, var};
use crate::common::{load_tirx_compiler, serial, structural_equal};

#[test]
fn ops() {
    load_tirx_compiler();
    let _serial = serial();
    let (a, b) = (var("a", prim_type("float32")), var("b", prim_type("float32")));
    let sum = logaddexp(&a, &b, loc()).unwrap();
    assert!(structural_equal(&sum.ty, &prim_type("float32")));
    let bits = reinterpret(prim_type("int32"), &a, loc()).unwrap();
    assert!(structural_equal(&bits.ty, &prim_type("int32")));
    let ret = gpu_thread_return(loc()).unwrap();
    assert!(structural_equal(&ret.ty, &prim_type("void")));
}
