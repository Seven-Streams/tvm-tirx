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

//! Tests of `tvm_tirx::sym`, laid out like `src/sym/`: the tests of `src/sym/<path>.rs` are in
//! `tests/sym/<path>.rs`.

#[path = "../common/mod.rs"]
mod common;

mod analyzer;
mod int_set;
mod iter_affine_map;

use tvm_tirx::ir::{Expr, IntImm, Location, PrimType, Range, Var};
use tvm_tirx::sym::Analyzer;
use tvm_tirx::tvm_ffi::{BigInt, DLDataType, DLDataTypeExt, String as FfiString};

use common::global;

fn loc() -> Location {
    global("ir.UnknownLoc").call_packed(&[]).unwrap().try_into().unwrap()
}

fn imm(dtype: &str, value: i64) -> Expr {
    let dtype = DLDataType::try_from_str(dtype).unwrap();
    let imm: IntImm = global("ir.IntImm")
        .call_tuple((dtype, BigInt::from(value), loc()))
        .unwrap()
        .try_into()
        .unwrap();
    imm.into()
}

fn int32(value: i64) -> Expr {
    imm("int32", value)
}

fn op(name: &str, a: impl Into<Expr>, b: impl Into<Expr>) -> Expr {
    global(name).call_tuple((a.into(), b.into(), loc())).unwrap().try_into().unwrap()
}

fn var(name: &str) -> Var {
    global("ir.Var")
        .call_tuple((FfiString::from(name), PrimType::int(32), loc()))
        .unwrap()
        .try_into()
        .unwrap()
}

fn range(begin: i64, extent: i64) -> Range {
    Range::from_complete_fields(int32(begin), int32(extent), loc())
}

struct Fixture {
    analyzer: Analyzer,
    x: Var,
}

/// An analyzer with `x` bound to `[0, 16)`, through the native functions only.
fn fixture() -> Fixture {
    let x = var("x");
    // A typed `Analyzer` handle goes through `call_tuple` like any other object.
    let analyzer: Analyzer = global("sym.Analyzer").call_tuple(()).unwrap().try_into().unwrap();
    global("sym.AnalyzerBind").call_tuple((analyzer.clone(), x.clone(), range(0, 16))).unwrap();
    Fixture { analyzer, x }
}
