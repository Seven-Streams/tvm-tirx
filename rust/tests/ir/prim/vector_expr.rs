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

//! Tests of `ir::prim::vector_expr`.

use tvm_tirx::ir::prim;
use tvm_tirx::ir::Expr;

use super::{dtype, int, var};
use crate::common::{load_tirx_compiler, serial};
use crate::loc;

#[test]
fn vector_constructors() {
    load_tirx_compiler();
    let _serial = serial();
    let x = var("x", "int32");
    let ramp = prim::Ramp::new(&x, int("int32", 1), int("int32", 4), loc()).unwrap();
    assert_eq!(dtype(&ramp.clone().into()), "int32x4");
    let broadcast = prim::Broadcast::new(&x, int("int32", 4), loc()).unwrap();
    assert_eq!(dtype(&broadcast.clone().into()), "int32x4");
    let indices = [3, 2, 1, 0].map(|i| int("int32", i));
    let shuffle = prim::Shuffle::new([&ramp], indices, loc()).unwrap();
    assert_eq!(dtype(&shuffle.clone().into()), "int32x4");
    assert_eq!(shuffle.indices.len(), 4);
    assert!(prim::Shuffle::new(Vec::<Expr>::new(), [int("int32", 0)], loc()).is_err());
}
