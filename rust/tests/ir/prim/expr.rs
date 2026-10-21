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

//! Tests of `ir::prim::expr`.

use tvm_tirx::ir::prim;
use tvm_tirx::ir::{Expr, PrimType, StringImm, Type, Var};
use tvm_tirx::tvm_ffi::{Any, AnyView, Result};

use super::{dtype, var};
use crate::common::{agree, agree_kind, native, serial, structural_equal};
use crate::loc;

#[test]
fn node_constructors() {
    let _serial = serial();
    let (x, y) = (var("x", PrimType::int(32)), var("y", PrimType::int(32)));
    let (p, q) = (var("p", PrimType::bool()), var("q", PrimType::bool()));

    let add = prim::Add::new(&x, &y, loc()).unwrap();
    assert!(structural_equal(&add.a, &Expr::from(&x)));
    assert_eq!(dtype(&add.clone().into()), "int32");
    assert!(prim::Add::new(&x, 1.0f32, loc()).is_err(), "no type promotion");

    let binary: Vec<Expr> = vec![
        prim::Sub::new(&x, &y, loc()).unwrap().into(),
        prim::Mul::new(&x, &y, loc()).unwrap().into(),
        prim::Div::new(&x, &y, loc()).unwrap().into(),
        prim::Mod::new(&x, &y, loc()).unwrap().into(),
        prim::FloorDiv::new(&x, &y, loc()).unwrap().into(),
        prim::FloorMod::new(&x, &y, loc()).unwrap().into(),
        prim::Min::new(&x, &y, loc()).unwrap().into(),
        prim::Max::new(&x, &y, loc()).unwrap().into(),
        prim::LShift::new(&x, &y, loc()).unwrap().into(),
        prim::RShift::new(&x, &y, loc()).unwrap().into(),
        prim::BitwiseAnd::new(&x, &y, loc()).unwrap().into(),
        prim::BitwiseOr::new(&x, &y, loc()).unwrap().into(),
        prim::BitwiseXor::new(&x, &y, loc()).unwrap().into(),
        prim::BitwiseNot::new(&x, loc()).unwrap().into(),
    ];
    for expr in &binary {
        assert_eq!(dtype(expr), "int32");
    }
    let compare: Vec<Expr> = vec![
        prim::EQ::new(&x, &y, loc()).unwrap().into(),
        prim::NE::new(&x, &y, loc()).unwrap().into(),
        prim::LT::new(&x, &y, loc()).unwrap().into(),
        prim::LE::new(&x, &y, loc()).unwrap().into(),
        prim::GT::new(&x, &y, loc()).unwrap().into(),
        prim::GE::new(&x, &y, loc()).unwrap().into(),
        prim::And::new(&p, &q, loc()).unwrap().into(),
        prim::Or::new(&p, &q, loc()).unwrap().into(),
        prim::Not::new(&p, loc()).unwrap().into(),
    ];
    for expr in &compare {
        assert_eq!(dtype(expr), "bool");
    }

    let cast = prim::Cast::new(PrimType::float(32), &x, loc()).unwrap();
    assert_eq!(dtype(&cast.into()), "float32");
    let select = prim::Select::new(&p, &x, &y, loc()).unwrap();
    assert!(structural_equal(&select.true_value, &Expr::from(&x)));
    let let_ = prim::Let::new(&x, 1, &add, loc()).unwrap();
    assert!(structural_equal(&let_.var, &x));
}

/// The native constructor `name` on `args` and the location.
fn native_with_loc(name: &str, args: &[&Expr]) -> Result<Any> {
    let loc = loc();
    let mut views: Vec<AnyView> = args.iter().map(|arg| AnyView::from(*arg)).collect();
    views.push(AnyView::from(&loc));
    native(name, &views)
}

#[test]
fn binary_nodes_agree_with_cpp() {
    let _serial = serial();
    let (x, y) = (Expr::from(var("x", PrimType::int(32))), Expr::from(var("y", PrimType::int(32))));
    let (u, v4) = (
        Expr::from(var("u", PrimType::uint(8))),
        Expr::from(var("v", PrimType::int(32).with_lanes(4))),
    );
    let (p, q) = (Expr::from(var("p", PrimType::bool())), Expr::from(var("q", PrimType::bool())));
    let p4 = Expr::from(var("p4", PrimType::bool().with_lanes(4)));
    let f = Expr::from(var("f", PrimType::float(32)));
    let text: Expr = StringImm::new("s", loc()).unwrap().into();
    let operands: Vec<(&str, &Expr, &Expr)> = vec![
        ("int32", &x, &y),
        ("uint8", &u, &u),
        ("int32x4", &v4, &v4),
        ("bool", &p, &q),
        ("float32", &f, &f),
        ("int32 and float32", &x, &f),
        ("int32 and int32x4", &x, &v4),
    ];
    macro_rules! check {
        ($($node:ident),+) => {$(
            for (what, a, b) in &operands {
                let what = format!("{}({what})", stringify!($node));
                let native = native_with_loc(concat!("prim.", stringify!($node)), &[a, b]);
                agree(&what, prim::$node::new(*a, *b, loc()), native);
            }
            let what = format!("{}(string)", stringify!($node));
            let native = native_with_loc(concat!("prim.", stringify!($node)), &[&text, &x]);
            agree_kind(&what, prim::$node::new(&text, &x, loc()), native);
        )+};
    }
    check!(Add, Sub, Mul, Div, Mod, FloorDiv, FloorMod, Min, Max);
    check!(LShift, RShift, BitwiseAnd, BitwiseOr, BitwiseXor);
    check!(EQ, NE, LT, LE, GT, GE);
    let logical: Vec<(&str, &Expr, &Expr)> = vec![
        ("bool", &p, &q),
        ("boolx4", &p4, &p4),
        ("bool and boolx4", &p, &p4),
        ("int32 and bool", &x, &p),
        ("bool and int32", &p, &x),
    ];
    for (what, a, b) in &logical {
        agree(
            &format!("And({what})"),
            prim::And::new(*a, *b, loc()),
            native_with_loc("prim.And", &[a, b]),
        );
        agree(
            &format!("Or({what})"),
            prim::Or::new(*a, *b, loc()),
            native_with_loc("prim.Or", &[a, b]),
        );
    }
    for (what, a) in [("int32", &x), ("bool", &p), ("boolx4", &p4), ("float32", &f)] {
        let native_not = native_with_loc("prim.Not", &[a]);
        agree(&format!("Not({what})"), prim::Not::new(a, loc()), native_not);
        let native_bitwise_not = native_with_loc("prim.BitwiseNot", &[a]);
        agree(&format!("BitwiseNot({what})"), prim::BitwiseNot::new(a, loc()), native_bitwise_not);
    }
}

#[test]
fn other_nodes_agree_with_cpp() {
    let _serial = serial();
    let (x, y) = (Expr::from(var("x", PrimType::int(32))), Expr::from(var("y", PrimType::int(32))));
    let (p, p4) = (
        Expr::from(var("p", PrimType::bool())),
        Expr::from(var("p4", PrimType::bool().with_lanes(4))),
    );
    let (v4, w4) = (
        Expr::from(var("v", PrimType::int(32).with_lanes(4))),
        Expr::from(var("w", PrimType::int(32).with_lanes(4))),
    );
    let f = Expr::from(var("f", PrimType::float(32)));

    for (what, target, value) in [
        ("Cast(int32 to float32)", PrimType::float(32), &x),
        ("Cast(int32x4 to float32x4)", PrimType::float(32).with_lanes(4), &v4),
        ("Cast(int32 to float32x4)", PrimType::float(32).with_lanes(4), &x),
    ] {
        let target_ty: Type = target.clone().into();
        let native_cast =
            native("prim.Cast", &[(&target_ty).into(), value.into(), (&loc()).into()]);
        agree(what, prim::Cast::new(target, value, loc()), native_cast);
    }

    for (what, condition, t, f_) in [
        ("Select", &p, &x, &y),
        ("Select(vector)", &p4, &v4, &w4),
        ("Select(scalar condition)", &p, &v4, &w4),
        ("Select(lanes)", &p4, &x, &y),
        ("Select(condition type)", &x, &x, &y),
        ("Select(value types)", &p, &x, &f),
    ] {
        let native_select = native_with_loc("prim.Select", &[condition, t, f_]);
        agree(what, prim::Select::new(condition, t, f_, loc()), native_select);
    }

    let z = var("z", PrimType::int(32));
    let tensor_ty: Type = PrimType::float(32).into();
    let tensor = Var::new("t", Some(&tensor_ty), loc()).unwrap();
    let missing = Var::new("m", None, loc()).unwrap();
    for (what, var, value) in [
        ("Let", &z, &y),
        ("Let(value type)", &z, &f),
        ("Let(var type)", &tensor, &y),
        ("Let(untyped var)", &missing, &y),
    ] {
        let body = Expr::from(var.clone());
        let body = if what == "Let" { body } else { x.clone() };
        let native_let =
            native("prim.Let", &[var.into(), value.into(), (&body).into(), (&loc()).into()]);
        agree(what, prim::Let::new(var, value, &body, loc()), native_let);
    }
}
