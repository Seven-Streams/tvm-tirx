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

//! Tests of `tvm_tirx::ops`, laid out like `src/ops/`: the tests of `src/ops/<path>.rs` are in
//! `tests/ops/<path>.rs`. Most cases are built both by the Rust code and by the native function
//! it follows, which must agree.

#[path = "../common/mod.rs"]
mod common;

mod build;
mod pattern;

use tvm_tirx::ir::{Expr, Location, PrimType, Type, Var};
use tvm_tirx::ops::int_imm;
use tvm_tirx::tvm_ffi::{AnyView, DLDataType, DLDataTypeExt, String as FfiString};

use common::global;

fn dtype(name: &str) -> DLDataType {
    DLDataType::try_from_str(name).unwrap()
}

fn loc() -> Location {
    Location::unknown()
}

fn prim(name: &str) -> PrimType {
    global("ir.PrimType").call_tuple((dtype(name),)).unwrap().try_into().unwrap()
}

fn native_var(name: &str, ty: impl Into<Type>) -> Var {
    let ty: Type = ty.into();
    global("ir.Var").call_tuple((FfiString::from(name), ty, loc())).unwrap().try_into().unwrap()
}

fn var(name: &str, ty: impl Into<Type>) -> Expr {
    native_var(name, ty).into()
}

fn prim_var(ty: &str) -> Expr {
    var(&format!("v_{}", ty.replace('x', "_")), prim(ty))
}

fn imm(ty: &str, value: i64) -> Expr {
    int_imm(dtype(ty), value as i128).unwrap().into()
}

fn repr<T>(value: &T) -> std::string::String
where
    for<'a> AnyView<'a>: From<&'a T>,
{
    let text: FfiString =
        global("ffi.ReprPrint").call_packed(&[AnyView::from(value)]).unwrap().try_into().unwrap();
    text.as_str().to_string()
}
