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

//! Tests of `ops::build`: the helpers over the IR constructors, against the native functions they
//! follow. Each case is built by both, which must give the same node or an error of the same kind
//! with the same message.

use tvm_tirx::ir::prim::Shuffle;
use tvm_tirx::ir::{
    Attrs, Evaluate, Expr, FloatImm, For, ForKind, IntoExpr, MissingType, PtrType, SeqStmt, Stmt,
    Type, UniqueNameSupply, Var, UNKNOWN_LOC,
};
use tvm_tirx::ops::{float_imm, int32, int32_dtype, int_imm, prim_type};
use tvm_tirx::runtime::MemoryScope;
use tvm_tirx::tirx::{TensorType, ThreadScope};
use tvm_tirx::tvm_ffi::{
    Any, AnyCompatible, AnyView, Array, BigInt, DLDataType, DLDataTypeExt, ObjectRefCast,
    ObjectRefCore, Result, String as FfiString,
};

use super::{dtype, imm, loc, native_var, prim, prim_var, repr};
use crate::common::{global, serial, structural_equal};

fn scope(name: &str) -> MemoryScope {
    MemoryScope::get(name).unwrap()
}

fn native_tensor_type(shape: &[i64], element: &str) -> TensorType {
    let shape: Array<Expr> = shape.iter().map(|&extent| imm("int32", extent)).collect();
    let layout = None::<tvm_tirx::tirx::Layout>;
    global("tirx.TensorType")
        .call_packed(&[
            AnyView::from(&shape),
            AnyView::from(&prim(element)),
            AnyView::from(&scope("global")),
            AnyView::from(&0i32),
            AnyView::from(&layout),
            AnyView::from(&loc()),
        ])
        .unwrap()
        .try_into()
        .unwrap()
}

/// `ffi.StructuralEqual` with free variables mapped, so fresh variables of one shape are equal.
fn equal<L: AnyCompatible, R: AnyCompatible>(lhs: &L, rhs: &R) -> bool {
    let equal = global("ffi.StructuralEqual")
        .call_packed(&[
            AnyView::from(lhs),
            AnyView::from(rhs),
            AnyView::from(&true),
            AnyView::from(&false),
        ])
        .unwrap();
    bool::try_from(equal).unwrap()
}

/// The Rust helper and the native function agree: the same node, or an error of the same kind
/// with the same message.
fn agree<T>(what: &str, rust: Result<T>, native: Result<Any>)
where
    T: AnyCompatible + TryFrom<Any>,
    for<'a> AnyView<'a>: From<&'a T>,
    <T as TryFrom<Any>>::Error: std::fmt::Debug,
{
    match (rust, native) {
        (Ok(rust), Ok(native)) => {
            let native: T = native.try_into().unwrap();
            assert!(equal(&rust, &native), "{what}: {} vs {}", repr(&rust), repr(&native));
        }
        (Err(rust), Err(native)) => {
            assert_eq!(rust.kind(), native.kind(), "{what}: {}", native.message());
            assert_eq!(rust.message(), native.message(), "{what}");
        }
        (Ok(rust), Err(native)) => {
            panic!("{what}: Rust gave {}, C++ failed with {}", repr(&rust), native.message())
        }
        (Err(rust), Ok(native)) => {
            panic!("{what}: Rust failed with {}, C++ gave {}", rust.message(), repr::<Any>(&native))
        }
    }
}

#[test]
fn constants_and_types() {
    let _serial = serial();
    // The shared nodes are the native ones.
    assert_eq!(int32_dtype(), DLDataType::try_from_str("int32").unwrap());
    let int32_ty: Type =
        global("ir.PrimType").call_tuple((int32_dtype(),)).unwrap().try_into().unwrap();
    assert!(prim_type(int32_dtype()).unwrap().same_as(&int32_ty));

    // The constants of a bare literal, as C++ types them.
    let native_int = |dtype: DLDataType, value: i128| {
        global("ir.IntImm").call_tuple((dtype, BigInt::from(value), loc()))
    };
    agree("int32(7)", int32(7), native_int(int32_dtype(), 7));
    agree("int32(1 << 40)", int32(1 << 40), native_int(int32_dtype(), 1 << 40));
    for (name, value) in [("int64", 1i128 << 40), ("uint8", 255), ("uint8", 256), ("bool", 1)] {
        let what = format!("int_imm({name}, {value})");
        agree(&what, int_imm(dtype(name), value), native_int(dtype(name), value));
    }
    for (name, value) in [("float32", 2.5), ("float16", 1e9), ("int32", 1.0)] {
        let what = format!("float_imm({name}, {value})");
        let native = global("ir.FloatImm").call_tuple((dtype(name), value, loc()));
        agree(&what, float_imm(dtype(name), value), native);
    }
    // The implicit conversions of a literal argument.
    assert!(equal(&3i32.into_expr().unwrap(), &imm("int32", 3)));
    let float: FloatImm = 2.5f32.into_expr().unwrap().try_cast().unwrap();
    assert_eq!(float.value, 2.5);
    assert!(equal(&float.ty, &prim("float32")));

    // The `PrimExpr` queries of an expression.
    let x = prim_var("int32");
    assert_eq!(x.prim_dtype().unwrap(), int32_dtype());
    assert!(x.prim_ty().unwrap().same_as(&prim("int32")));
    let text: Expr = global("ir.StringImm")
        .call_tuple((FfiString::from("s"), loc()))
        .unwrap()
        .try_into()
        .unwrap();
    assert!(text.prim_dtype().is_err());
}

#[test]
fn shuffle() {
    let _serial = serial();
    let (v4, w4) = (prim_var("int32x4"), prim_var("int32x4"));
    let index = |i: i64| imm("int32", i);
    let native = |vectors: Vec<Expr>, indices: Vec<i64>| -> Expr {
        let vectors: Array<Expr> = vectors.into_iter().collect();
        let indices: Array<Expr> = indices.into_iter().map(index).collect();
        global("prim.Shuffle").call_tuple((vectors, indices, loc())).unwrap().try_into().unwrap()
    };
    let concat = Shuffle::concat([v4.clone(), w4.clone()]).unwrap();
    assert!(equal(&concat, &native(vec![v4.clone(), w4.clone()], (0..8).collect())));
    assert!(Shuffle::concat([v4.clone()]).unwrap().same_as(&v4), "one vector is itself");
    assert!(Shuffle::concat(Vec::<Expr>::new()).is_err());
    let element = Shuffle::extract_element(v4.clone(), 3).unwrap();
    assert!(equal(&element, &native(vec![v4], vec![3])));
}

#[test]
fn variables_and_pointers() {
    let _serial = serial();
    let prim_var = Var::prim("p", dtype("int64")).unwrap();
    assert!(equal(&prim_var, &native_var("p", prim("int64"))));
    assert!(prim_var.ty.same_as(&prim("int64")));

    let missing: Type = global("ir.MissingType").call_packed(&[]).unwrap().try_into().unwrap();
    assert!(MissingType::new().same_as(&missing), "MissingType() is a singleton");

    for name in ["global", "shared", "local"] {
        let scope = scope(name);
        assert_eq!(scope.name().unwrap().as_str(), name);
        let native: MemoryScope = global("ir.PtrType")
            .call_tuple((Type::from(prim("int8")), FfiString::from(name)))
            .and_then(|ptr| Ok(PtrType::try_from(ptr)?.scope.clone()))
            .unwrap();
        assert!(scope.same_as(&native), "{name}");
    }
    assert!(MemoryScope::get("no.such.scope").is_err());
    let void: PtrType = global("ir.PtrType")
        .call_tuple((prim("void"), FfiString::from("global")))
        .unwrap()
        .try_into()
        .unwrap();
    let rust_void = PtrType::void().unwrap();
    assert!(equal(&rust_void, &void));
    assert!(rust_void.scope.same_as(&MemoryScope::global()));
    assert!(MemoryScope::global().same_as(&scope("global")));
}

#[test]
fn tensor_data() {
    let _serial = serial();
    let ty = native_tensor_type(&[4, 8], "float16");
    let tensor = native_var("buf", ty.clone());
    let native_ptr: Type = global("ir.PtrType")
        .call_tuple((Type::from(ty.dtype.clone()), ty.scope.clone()))
        .unwrap()
        .try_into()
        .unwrap();
    assert!(equal(&ty.data_ptr_type().unwrap(), &native_ptr));
    let tensor_data_ptr = global("ir.GetOp").call_tuple((FfiString::from("tirx.tensor_data_ptr"),));
    let native_data: Expr = global("ir.Call")
        .call_tuple((
            None::<Type>,
            Expr::try_from(tensor_data_ptr.unwrap()).unwrap(),
            Array::new(vec![Expr::from(&tensor)]),
            None::<Attrs>,
            Array::<Type>::new(vec![]),
            loc(),
        ))
        .unwrap()
        .try_into()
        .unwrap();
    assert!(equal(&tensor.tensor_data().unwrap(), &native_data));
    assert!(native_var("x", prim("int32")).tensor_data().is_err());
}

#[test]
fn native_objects() {
    let _serial = serial();
    let what = |name: &str| format!("ThreadScope::Get({name})");
    for name in ["cuda.kernel", "cuda.warp", "cuda.no_such_scope", "kernel"] {
        let rust = ThreadScope::get(name);
        let native = global("tirx.ThreadScopeGet").call_tuple((FfiString::from(name),));
        match (rust, native) {
            (Ok(rust), Ok(native)) => {
                let native: ThreadScope = native.try_into().unwrap();
                assert!(rust.same_as(&native), "{}", what(name));
                assert_eq!(rust.name_str().unwrap().as_str(), name, "{}", what(name));
                assert_eq!(rust.local_name().unwrap().as_str(), &name[5..], "{}", what(name));
                assert_eq!(rust.backend().unwrap().as_str(), "cuda", "{}", what(name));
            }
            (Err(rust), Err(native)) => assert_eq!(rust.message(), native.message()),
            _ => panic!("{}: one side failed", what(name)),
        }
    }
    UniqueNameSupply::new("");
}

#[test]
fn sequences_and_loops() {
    let _serial = serial();
    let evaluate = |value: Expr| -> Stmt { Evaluate::new(value, UNKNOWN_LOC).into() };
    let (a, b) = (evaluate(prim_var("int32")), evaluate(imm("int32", 1)));
    let nested: Stmt = SeqStmt::new(vec![a.clone(), b.clone()], UNKNOWN_LOC).into();
    // The implicit `Stmt -> SeqStmt` conversion passes a sequence through and wraps anything else.
    for (what, stmt) in [("a statement", a.clone()), ("a no-op", evaluate(imm("int32", 0)))] {
        let rust = SeqStmt::from_stmt(stmt.clone());
        let native = global("ir.SeqStmt").call_tuple((stmt, None::<tvm_tirx::ir::Location>));
        agree(&format!("SeqStmt of {what}"), rust, native);
    }
    let sequence = SeqStmt::from_stmt(nested.clone()).unwrap();
    assert!(sequence.same_as(&nested), "a sequence is itself");

    // `ForNode::extent()`, as `Range::GetExtentFromBeginEnd` spells it.
    let (i, n) = (native_var("i", prim("int32")), prim_var("int32"));
    let loop_ = |begin: Expr, end: Expr| -> For {
        For::new(&i, begin, end, imm("int32", 1), ForKind::Default, a.clone(), None, loc()).unwrap()
    };
    let native_extent = |begin: Expr, end: Expr| -> Expr {
        global("ir.RangeGetExtentFromBeginEnd")
            .call_tuple((begin, end))
            .unwrap()
            .try_into()
            .unwrap()
    };
    for (begin, end) in [
        (imm("int32", 0), n.clone()),
        (imm("int32", 2), imm("int32", 10)),
        (
            n.clone(),
            global("prim._OpAdd")
                .call_tuple((n.clone(), imm("int32", 4), loc()))
                .unwrap()
                .try_into()
                .unwrap(),
        ),
    ] {
        let extent = loop_(begin.clone(), end.clone()).extent().unwrap();
        assert!(structural_equal(&extent, &native_extent(begin, end)));
    }
}
