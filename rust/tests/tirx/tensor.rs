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

use tvm_tirx::ir::{Expr, PrimType, StringImm, Type, TypeObj, UNKNOWN_LOC, Location};
use tvm_tirx::tirx::{self, tensor_is_scalar, Layout, TensorType};
use tvm_tirx::tvm_ffi::{AnyView, Array, ObjectRefCast};
use super::{int32, int_imm, ir, memory_scope, s, tile_8x8, var};
use crate::common::{
    agree, agree_kind, assert_fields, assert_type_contracts, global, native, serial,
    structural_equal,
};

#[test]
fn generated_types_match_the_registry() {
    let _serial = serial();
    assert_type_contracts!(true; tirx::TensorTypeObj: TypeObj);
}

#[test]
fn tensor_types() {
    let _serial = serial();
    let scope = memory_scope();
    let ty =
        TensorType::new([int32(1)], PrimType::float(32), &scope, 0, None, UNKNOWN_LOC).unwrap();
    assert_eq!(ty.data_alignment, 64);
    assert!(ty.has_default_layout(None).unwrap());
    let tile = tile_8x8();
    let laid_out = TensorType::new(
        [int32(8), int32(8)],
        PrimType::float(32),
        &scope,
        16,
        Some(&Layout::from(&tile)),
        UNKNOWN_LOC,
    )
    .unwrap();
    assert!(laid_out.has_default_layout(None).unwrap());
    assert!(tensor_is_scalar(var("A", &ty), true).unwrap());
    assert!(!tensor_is_scalar(var("B", &laid_out), true).unwrap());
}

#[test]
fn tensor_types_agree_with_cpp() {
    let _serial = serial();
    let scope = memory_scope();
    let float32 = PrimType::float(32);
    let layout: Layout = tile_8x8().into();
    let text: Expr = StringImm::new("s", UNKNOWN_LOC).into();
    let cases: Vec<(&str, Vec<Expr>, i32, Option<Layout>)> = vec![
        ("TensorType(default alignment)", vec![int32(1)], 0, None),
        ("TensorType(negative alignment)", vec![int32(4), int32(4)], -8, None),
        ("TensorType(alignment)", vec![int32(8), int32(8)], 128, Some(layout)),
    ];
    for (what, shape, alignment, layout) in cases {
        let array: Array<Expr> = shape.iter().cloned().collect();
        let native_type = native(
            "tirx.TensorType",
            &[
                (&array).into(),
                (&float32).into(),
                (&scope).into(),
                (&alignment).into(),
                (&layout).into(),
                (&Location::unknown()).into(),
            ],
        );
        let rust =
            TensorType::new(shape, &float32, &scope, alignment, layout.as_ref(), UNKNOWN_LOC);
        agree(what, rust, native_type);
    }
    let array: Array<Expr> = [text.clone()].into_iter().collect();
    let (alignment, layout) = (0i32, Option::<Layout>::None);
    let native_type = native(
        "tirx.TensorType",
        &[
            (&array).into(),
            (&float32).into(),
            (&scope).into(),
            (&alignment).into(),
            (&layout).into(),
            (&Location::unknown()).into(),
        ],
    );
    let rust = TensorType::new([text], &float32, &scope, 0, None, UNKNOWN_LOC);
    agree_kind("TensorType(shape)", rust, native_type);
}

#[test]
fn native_ir_objects_agree_with_getters_and_rust_allocation() {
    let _serial = serial();
    let ir = ir();

    // tirx.TensorType
    let shape = Array::<Expr>::new(vec![int_imm(&ir, 4).into(), int_imm(&ir, 8).into()]);
    let tensor: tirx::TensorType = global("tirx.TensorType")
        .call_packed(&[
            AnyView::from(&shape),
            AnyView::from(&ir.int32),
            AnyView::from(&s("global")),
            AnyView::from(&64i64),
            AnyView::new(),
            AnyView::from(&ir.loc),
        ])
        .unwrap()
        .try_into()
        .unwrap();
    assert_fields!(tensor;
        dtype = ir.int32, scope, shape = shape, data_alignment = 64i64,
        layout = None::<tirx::Layout>,
    );
    let rust = tirx::TensorType::from_complete_fields(
        tensor.loc.clone(),
        tensor.dtype.clone(),
        tensor.scope.clone(),
        tensor.shape.clone(),
        tensor.data_alignment,
        tensor.layout.clone(),
    );
    assert!(structural_equal(&rust, &tensor));
    let as_type: Type = tensor.clone().into();
    assert!(as_type.clone().try_cast::<tirx::TensorType>().is_ok());
    assert!(as_type.try_cast::<tirx::TensorMapType>().is_err());
}
