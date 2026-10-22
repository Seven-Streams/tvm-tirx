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

//! Tests of `tvm_tirx::tirx`, laid out like `src/tirx/`: the tests of `src/tirx/<path>.rs` are in
//! `tests/tirx/<path>.rs`.

#[path = "../common/mod.rs"]
mod common;

mod analysis;
mod function;
mod index_map;
mod layout;
mod op;
mod tensor;
mod thread_scope;
mod tile;
mod transform;
mod r#type;

use std::sync::atomic::{AtomicUsize, Ordering};

use tvm_tirx::ir::{
    dtype, BaseFunc, DictAttrs, Evaluate, Expr, GlobalInfo, GlobalVar, IRModule, IntImm, Location,
    PrimType, Range, SeqStmt, Stmt, Type, Var, UNKNOWN_LOC,
};
use tvm_tirx::runtime::MemoryScope;
use tvm_tirx::tirx::{self, Axis, Function, Iter, TileLayout};
use tvm_tirx::tvm_ffi::{
    Any, Array, BigInt, DLDataType, DLDataTypeExt, Map, ObjectRefCast, String as FfiString,
};

use common::global;

fn int32(value: i64) -> Expr {
    IntImm::from_dtype(dtype::int(32), value, UNKNOWN_LOC).unwrap().into()
}

fn var(name: &str, ty: impl Into<Type>) -> Var {
    global("ir.Var")
        .call_tuple((FfiString::from(name), Some(ty.into()), Location::unknown()))
        .unwrap()
        .try_into()
        .unwrap()
}

fn op(name: &str, a: impl Into<Expr>, b: impl Into<Expr>) -> Expr {
    global(name).call_tuple((a.into(), b.into(), Location::unknown())).unwrap().try_into().unwrap()
}

fn range(extent: i64) -> Range {
    global("ir.Range")
        .call_tuple((int32(0), Some(int32(extent)), Location::unknown()))
        .unwrap()
        .try_into()
        .unwrap()
}

fn as_i64(expr: &Expr) -> i64 {
    expr.clone().try_cast::<IntImm>().unwrap().value.to_i64().unwrap()
}

/// A new scope of the test's own, since the builtin ones cannot be registered again.
fn memory_scope() -> MemoryScope {
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    let index = COUNT.fetch_add(1, Ordering::Relaxed);
    MemoryScope::register(&format!("rust.tests.tirx.{index}")).unwrap()
}

/// The row-major layout of an 8 x 8 tile.
fn tile_8x8() -> TileLayout {
    let m = Axis::get("m").unwrap();
    TileLayout::new(
        [Iter::new(int32(8), int32(8), m.clone()), Iter::new(int32(8), int32(1), m)],
        Vec::<Iter>::new(),
        &Map::new(),
    )
}

fn void_function(params: Vec<Var>) -> Function {
    let zero = global("ir.Evaluate").call_tuple((int32(0), Location::unknown())).unwrap();
    let body: SeqStmt = global("ir.SeqStmt")
        .call_tuple((Array::<Stmt>::new(vec![zero.try_into().unwrap()]), Some(Location::unknown())))
        .unwrap()
        .try_into()
        .unwrap();
    let missing: Type = global("ir.MissingType").call_tuple(()).unwrap().try_into().unwrap();
    Function::new(params, Some(&body), missing, DictAttrs::new(Map::new()), UNKNOWN_LOC)
}

fn module_of(name: &str, func: &Function) -> IRModule {
    let gv: GlobalVar = global("ir.GlobalVar")
        .call_tuple((FfiString::from(name), Location::unknown()))
        .unwrap()
        .try_into()
        .unwrap();
    let functions: Map<GlobalVar, BaseFunc> = [(gv, BaseFunc::from(func))].into_iter().collect();
    IRModule::from_functions(&functions, None, &Map::<FfiString, Array<GlobalInfo>>::new()).unwrap()
}

fn s(text: &str) -> FfiString {
    FfiString::from(text)
}

/// The pieces the IR-bearing tests share.
struct Ir {
    loc: Location,
    int32: PrimType,
    x: Var,
    y: Var,
}

fn ir() -> Ir {
    let loc: Location = global("ir.UnknownLoc").call_packed(&[]).unwrap().try_into().unwrap();
    let int32: PrimType = global("ir.PrimType")
        .call_tuple((DLDataType::try_from_str("int32").unwrap(),))
        .unwrap()
        .try_into()
        .unwrap();
    let var = |name: &str| -> Var {
        global("ir.Var")
            .call_tuple((s(name), int32.clone(), loc.clone()))
            .unwrap()
            .try_into()
            .unwrap()
    };
    let (x, y) = (var("x"), var("y"));
    Ir { loc, int32, x, y }
}

fn int_imm(ir: &Ir, value: i64) -> IntImm {
    global("ir.IntImm")
        .call_tuple((ir.int32.dtype, BigInt::from(value), ir.loc.clone()))
        .unwrap()
        .try_into()
        .unwrap()
}

/// `{ evaluate(value) }`, with the statements allocated in Rust.
fn evaluate_body(ir: &Ir, value: i64) -> SeqStmt {
    let stmt = Evaluate::from_complete_fields(ir.loc.clone(), int_imm(ir, value).into());
    SeqStmt::from_complete_fields(ir.loc.clone(), Array::<Stmt>::new(vec![stmt.into()]))
}

fn function(ir: &Ir, body: &SeqStmt) -> tirx::Function {
    let params = Array::new(vec![ir.x.clone(), ir.y.clone()]);
    let attrs = DictAttrs::new(Map::<FfiString, Any>::new());
    global("tirx.Function")
        .call_tuple((params, body.clone(), Type::from(ir.int32.clone()), attrs, ir.loc.clone()))
        .unwrap()
        .try_into()
        .unwrap()
}
