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

//! Tests of `tirx::tensor`: tensor types.

use tvm_tirx::tirx::{tensor_is_scalar, Layout, TensorType};

use super::{int32, loc, memory_scope, prim_type, tile_8x8, var};
use crate::common::{load_tirx_compiler, serial};

#[test]
fn tensor_types() {
    load_tirx_compiler();
    let _serial = serial();
    let scope = memory_scope();
    let ty = TensorType::new([int32(1)], prim_type("float32"), &scope, 0, None, loc()).unwrap();
    assert_eq!(ty.data_alignment, 64);
    assert!(ty.has_default_layout(None).unwrap());
    let tile = tile_8x8();
    let laid_out = TensorType::new(
        [int32(8), int32(8)],
        prim_type("float32"),
        &scope,
        16,
        Some(&Layout::from(&tile)),
        loc(),
    )
    .unwrap();
    assert!(laid_out.has_default_layout(None).unwrap());
    assert!(tensor_is_scalar(var("A", &ty), true).unwrap());
    assert!(!tensor_is_scalar(var("B", &laid_out), true).unwrap());
}
