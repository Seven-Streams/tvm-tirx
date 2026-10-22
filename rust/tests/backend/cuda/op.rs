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

//! Tests of `backend::cuda::op`: the attributes of the CUDA instructions.

use tvm_tirx::backend::cuda::*;
use tvm_tirx::ir::AttrsObj;
use tvm_tirx::tvm_ffi::{DLDataType, DLDataTypeExt, String as FfiString};

use crate::common::{assert_fields, assert_type_contracts, serial};

#[test]
fn generated_types_match_the_registry() {
    let _serial = serial();
    assert_type_contracts!(true;
        TCGen05InstrDescriptorAttrsObj: AttrsObj,
        TCGen05InstrDescriptorBlockScaledAttrsObj: AttrsObj, TensorMapEncodeTiledAttrObj: AttrsObj,
    );
}

fn s(text: &str) -> FfiString {
    FfiString::from(text)
}

#[test]
fn rust_allocated_data_objects_are_read_by_native_getters() {
    let _serial = serial();
    let descriptor = TCGen05InstrDescriptorAttrs::new(
        s("f32"),
        s("f16"),
        s("bf16"),
        1,
        2,
        3,
        true,
        false,
        4,
        false,
        true,
        true,
        false,
    );
    assert_fields!(descriptor;
        d_dtype = s("f32"), a_dtype = s("f16"), b_dtype = s("bf16"), M = 1i64, N = 2i64, K = 3i64,
        trans_a = true, trans_b = false, n_cta_groups = 4i64, neg_a = false, neg_b = true,
        sat_d = true, is_sparse = false,
    );

    let block_scaled = TCGen05InstrDescriptorBlockScaledAttrs::new(
        s("f32"),
        s("e4m3"),
        s("e5m2"),
        s("ue8"),
        s("ue4"),
        1,
        2,
        3,
        false,
        true,
        4,
        true,
        false,
        true,
    );
    assert_fields!(block_scaled;
        d_dtype = s("f32"), a_dtype = s("e4m3"), b_dtype = s("e5m2"), sfa_dtype = s("ue8"),
        sfb_dtype = s("ue4"), M = 1i64, N = 2i64, K = 3i64, trans_a = false, trans_b = true,
        n_cta_groups = 4i64, neg_a = true, neg_b = false, is_sparse = true,
    );

    let uint8 = DLDataType::try_from_str("uint8").unwrap();
    let tiled = TensorMapEncodeTiledAttr::new(uint8, 1, 2, 3, 4, 5, 6);
    assert_fields!(tiled;
        descriptor_dtype = uint8, rank = 1i64, interleave = 2i64, swizzle = 3i64,
        l2_promotion = 4i64, oob_fill = 5i64, force_cu_dtype = 6i64,
    );
}
