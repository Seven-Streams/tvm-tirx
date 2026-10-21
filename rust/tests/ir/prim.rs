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

//! Tests of `ir::prim`, laid out like `src/ir/prim/`: the node constructors and the
//! type-promoting operators.

mod expr;
mod op;
mod vector_expr;

use tvm_tirx::ir::prim;
use tvm_tirx::ir::{Expr, FloatImm, IntImm, PrimType, Var};
use tvm_tirx::tvm_ffi::{BigInt, DLDataType, DLDataTypeExt, ObjectRefCast, String as FfiString};

use super::loc;
use crate::common::{global, serial};

fn ty(dtype: &str) -> PrimType {
    PrimType::from_dtype(DLDataType::try_from_str(dtype).unwrap()).unwrap()
}

fn int(dtype: &str, value: i64) -> Expr {
    let imm: IntImm = global("ir.IntImm")
        .call_tuple((DLDataType::try_from_str(dtype).unwrap(), BigInt::from(value), loc()))
        .unwrap()
        .try_into()
        .unwrap();
    imm.into()
}

fn float(dtype: &str, value: f64) -> Expr {
    let imm: FloatImm = global("ir.FloatImm")
        .call_tuple((DLDataType::try_from_str(dtype).unwrap(), value, loc()))
        .unwrap()
        .try_into()
        .unwrap();
    imm.into()
}

fn var(name: &str, dtype: &str) -> Var {
    global("ir.Var")
        .call_tuple((FfiString::from(name), ty(dtype), loc()))
        .unwrap()
        .try_into()
        .unwrap()
}

fn dtype(expr: &Expr) -> String {
    let ty: PrimType = expr.ty.clone().try_cast().unwrap();
    ty.dtype.to_string().as_str().to_owned()
}

fn int_value(expr: Expr) -> i64 {
    let imm: IntImm = expr.try_cast().unwrap();
    imm.value.to_i64().unwrap()
}

#[test]
fn expr_deep_equal() {
    let _serial = serial();
    let x = var("x", "int32");
    let a = prim::op::add(&x, int("int32", 1), loc()).unwrap();
    let b = prim::op::add(&x, int("int32", 1), loc()).unwrap();
    assert!(prim::expr_deep_equal(&a, &b).unwrap());
    assert!(!prim::expr_deep_equal(&a, var("x", "int32")).unwrap());
}
