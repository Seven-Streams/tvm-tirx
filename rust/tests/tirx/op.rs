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

use tvm_tirx::ir::PrimType;
use tvm_tirx::tirx::{gpu_thread_return, logaddexp, reinterpret};

use super::{loc, var};
use crate::common::{serial, structural_equal};

#[test]
fn ops() {
    let _serial = serial();
    let (a, b) = (var("a", PrimType::float(32)), var("b", PrimType::float(32)));
    let sum = logaddexp(&a, &b, loc()).unwrap();
    assert!(structural_equal(&sum.ty, &PrimType::float(32)));
    let bits = reinterpret(PrimType::int(32), &a, loc()).unwrap();
    assert!(structural_equal(&bits.ty, &PrimType::int(32)));
    let ret = gpu_thread_return(loc()).unwrap();
    assert!(structural_equal(&ret.ty, &PrimType::void()));
}
