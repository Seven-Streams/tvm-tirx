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

use tvm_tirx::ir::prim::{self, *};
use tvm_tirx::ir::{
    Expr, ExprObj, IntImm, Location, PrimType, StringImm, Type, Var, UNKNOWN_LOC, MissingType,
};
use tvm_tirx::tvm_ffi::{
    Any, AnyView, Array, BigInt, DLDataType, DLDataTypeExt, ObjectRefCast, Result,
    String as FfiString,
};
use super::{dtype, var};
use crate::common::{
    agree, agree_kind, assert_type_contracts, global, native, serial, structural_equal,
};

#[test]
fn generated_types_match_the_registry() {
    let _serial = serial();
    // Every node derives directly from `ir.Expr`.
    assert_type_contracts!(true;
        AddObj: ExprObj, AndObj: ExprObj, BitwiseAndObj: ExprObj, BitwiseNotObj: ExprObj,
        BitwiseOrObj: ExprObj, BitwiseXorObj: ExprObj, CastObj: ExprObj, DivObj: ExprObj,
        EQObj: ExprObj, FloorDivObj: ExprObj, FloorModObj: ExprObj, GEObj: ExprObj, GTObj: ExprObj,
        LEObj: ExprObj, LShiftObj: ExprObj, LTObj: ExprObj, LetObj: ExprObj, MaxObj: ExprObj,
        MinObj: ExprObj, ModObj: ExprObj, MulObj: ExprObj, NEObj: ExprObj, NotObj: ExprObj,
        OrObj: ExprObj, RShiftObj: ExprObj, SelectObj: ExprObj, SubObj: ExprObj,
    );
}

#[test]
fn node_constructors() {
    let _serial = serial();
    let (x, y) = (var("x", PrimType::int(32)), var("y", PrimType::int(32)));
    let (p, q) = (var("p", PrimType::bool()), var("q", PrimType::bool()));

    let add = prim::Add::new(&x, &y, UNKNOWN_LOC).unwrap();
    assert!(structural_equal(&add.a, &Expr::from(&x)));
    assert_eq!(dtype(&add.clone().into()), "int32");
    assert!(prim::Add::new(&x, 1.0f32, UNKNOWN_LOC).is_err(), "no type promotion");

    let binary: Vec<Expr> = vec![
        prim::Sub::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::Mul::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::Div::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::Mod::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::FloorDiv::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::FloorMod::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::Min::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::Max::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::LShift::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::RShift::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::BitwiseAnd::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::BitwiseOr::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::BitwiseXor::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::BitwiseNot::new(&x, UNKNOWN_LOC).unwrap().into(),
    ];
    for expr in &binary {
        assert_eq!(dtype(expr), "int32");
    }
    let compare: Vec<Expr> = vec![
        prim::EQ::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::NE::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::LT::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::LE::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::GT::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::GE::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::And::new(&p, &q, UNKNOWN_LOC).unwrap().into(),
        prim::Or::new(&p, &q, UNKNOWN_LOC).unwrap().into(),
        prim::Not::new(&p, UNKNOWN_LOC).unwrap().into(),
    ];
    for expr in &compare {
        assert_eq!(dtype(expr), "bool");
    }

    let cast = prim::Cast::new(PrimType::float(32), &x, UNKNOWN_LOC).unwrap();
    assert_eq!(dtype(&cast.into()), "float32");
    let select = prim::Select::new(&p, &x, &y, UNKNOWN_LOC).unwrap();
    assert!(structural_equal(&select.true_value, &Expr::from(&x)));
    let let_ = prim::Let::new(&x, 1, &add, UNKNOWN_LOC).unwrap();
    assert!(structural_equal(&let_.var, &x));
}

/// The native constructor `name` on `args` and the location.
fn native_with_loc(name: &str, args: &[&Expr]) -> Result<Any> {
    let loc = Location::unknown();
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
        Expr::from(var("v", PrimType::int(32).with_lanes(4).unwrap())),
    );
    let (p, q) = (Expr::from(var("p", PrimType::bool())), Expr::from(var("q", PrimType::bool())));
    let p4 = Expr::from(var("p4", PrimType::bool().with_lanes(4).unwrap()));
    let f = Expr::from(var("f", PrimType::float(32)));
    let text: Expr = StringImm::new("s", UNKNOWN_LOC).into();
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
                agree(&what, prim::$node::new(*a, *b, UNKNOWN_LOC), native);
            }
            let what = format!("{}(string)", stringify!($node));
            let native = native_with_loc(concat!("prim.", stringify!($node)), &[&text, &x]);
            agree_kind(&what, prim::$node::new(&text, &x, UNKNOWN_LOC), native);
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
            prim::And::new(*a, *b, UNKNOWN_LOC),
            native_with_loc("prim.And", &[a, b]),
        );
        agree(
            &format!("Or({what})"),
            prim::Or::new(*a, *b, UNKNOWN_LOC),
            native_with_loc("prim.Or", &[a, b]),
        );
    }
    for (what, a) in [("int32", &x), ("bool", &p), ("boolx4", &p4), ("float32", &f)] {
        let native_not = native_with_loc("prim.Not", &[a]);
        agree(&format!("Not({what})"), prim::Not::new(a, UNKNOWN_LOC), native_not);
        let native_bitwise_not = native_with_loc("prim.BitwiseNot", &[a]);
        agree(
            &format!("BitwiseNot({what})"),
            prim::BitwiseNot::new(a, UNKNOWN_LOC),
            native_bitwise_not,
        );
    }
}

#[test]
fn other_nodes_agree_with_cpp() {
    let _serial = serial();
    let (x, y) = (Expr::from(var("x", PrimType::int(32))), Expr::from(var("y", PrimType::int(32))));
    let (p, p4) = (
        Expr::from(var("p", PrimType::bool())),
        Expr::from(var("p4", PrimType::bool().with_lanes(4).unwrap())),
    );
    let (v4, w4) = (
        Expr::from(var("v", PrimType::int(32).with_lanes(4).unwrap())),
        Expr::from(var("w", PrimType::int(32).with_lanes(4).unwrap())),
    );
    let f = Expr::from(var("f", PrimType::float(32)));

    for (what, target, value) in [
        ("Cast(int32 to float32)", PrimType::float(32), &x),
        ("Cast(int32x4 to float32x4)", PrimType::float(32).with_lanes(4).unwrap(), &v4),
        ("Cast(int32 to float32x4)", PrimType::float(32).with_lanes(4).unwrap(), &x),
    ] {
        let target_ty: Type = target.clone().into();
        let native_cast = native(
            "prim.Cast",
            &[(&target_ty).into(), value.into(), (&Location::unknown()).into()],
        );
        agree(what, prim::Cast::new(target, value, UNKNOWN_LOC), native_cast);
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
        agree(what, prim::Select::new(condition, t, f_, UNKNOWN_LOC), native_select);
    }

    let z = var("z", PrimType::int(32));
    let tensor_ty: Type = PrimType::float(32).into();
    let tensor = Var::new("t", tensor_ty, UNKNOWN_LOC);
    let missing = Var::new("m", MissingType::new(), UNKNOWN_LOC);
    for (what, var, value) in [
        ("Let", &z, &y),
        ("Let(value type)", &z, &f),
        ("Let(var type)", &tensor, &y),
        ("Let(untyped var)", &missing, &y),
    ] {
        let body = Expr::from(var.clone());
        let body = if what == "Let" { body } else { x.clone() };
        let native_let = native(
            "prim.Let",
            &[var.into(), value.into(), (&body).into(), (&Location::unknown()).into()],
        );
        agree(what, prim::Let::new(var, value, &body, UNKNOWN_LOC), native_let);
    }
}

/// Operands shared by the tests: two `int32` variables and two boolean expressions over them.
struct Operands {
    loc: Location,
    int32: PrimType,
    x_var: Var,
    x: Expr,
    y: Expr,
    p: Expr,
    q: Expr,
}

fn prim_type(dtype: &str) -> PrimType {
    global("ir.PrimType")
        .call_tuple((DLDataType::try_from_str(dtype).unwrap(),))
        .unwrap()
        .try_into()
        .unwrap()
}

fn operands() -> Operands {
    let loc: Location = global("ir.UnknownLoc").call_packed(&[]).unwrap().try_into().unwrap();
    let int32 = prim_type("int32");
    let var = |name: &str| -> Var {
        global("ir.Var")
            .call_tuple((FfiString::from(name), int32.clone(), loc.clone()))
            .unwrap()
            .try_into()
            .unwrap()
    };
    let x_var = var("x");
    let (x, y): (Expr, Expr) = (x_var.clone().into(), var("y").into());
    let compare = |name: &str| -> Expr {
        global(name).call_tuple((x.clone(), y.clone(), loc.clone())).unwrap().try_into().unwrap()
    };
    let (p, q) = (compare("prim.LT"), compare("prim.GT"));
    Operands { loc, int32, x_var, x, y, p, q }
}

fn int_imm(o: &Operands, value: i64) -> Expr {
    let imm: IntImm = global("ir.IntImm")
        .call_tuple((o.int32.dtype, BigInt::from(value), o.loc.clone()))
        .unwrap()
        .try_into()
        .unwrap();
    imm.into()
}

/// A native binary node reads back its operands, and the Rust-allocated node equals it, as does
/// the one the Rust constructor builds.
macro_rules! check_binary {
    ($o:expr; $($node:ident($lhs:expr, $rhs:expr)),* $(,)?) => {$({
        let name = concat!("prim.", stringify!($node));
        let native: $node = global(name)
            .call_tuple(($lhs.clone(), $rhs.clone(), $o.loc.clone()))
            .unwrap()
            .try_into()
            .unwrap();
        assert!(structural_equal(&native.a, &$lhs), "{name}: a");
        assert!(structural_equal(&native.b, &$rhs), "{name}: b");
        let rust = $node::from_complete_fields($o.loc.clone(), native.ty.clone(), $lhs.clone(), $rhs.clone());
        assert!(structural_equal(&rust, &native), "{name}: Rust-allocated node differs");
        let swapped = $node::from_complete_fields($o.loc.clone(), native.ty.clone(), $rhs.clone(), $lhs.clone());
        assert!(!structural_equal(&swapped, &native), "{name}: operands are not distinguished");
        let built = $node::new($lhs.clone(), $rhs.clone(), $o.loc.clone()).unwrap();
        assert!(structural_equal(&built, &native), "{name}: Rust constructor differs");
    })*};
}

#[test]
fn binary_nodes_round_trip() {
    let _serial = serial();
    let o = operands();
    check_binary!(o;
        Add(o.x, o.y), Sub(o.x, o.y), Mul(o.x, o.y), Div(o.x, o.y), Mod(o.x, o.y),
        FloorDiv(o.x, o.y), FloorMod(o.x, o.y), Min(o.x, o.y), Max(o.x, o.y),
        EQ(o.x, o.y), NE(o.x, o.y), LT(o.x, o.y), LE(o.x, o.y), GT(o.x, o.y), GE(o.x, o.y),
        BitwiseAnd(o.x, o.y), BitwiseOr(o.x, o.y), BitwiseXor(o.x, o.y),
        LShift(o.x, o.y), RShift(o.x, o.y),
        And(o.p, o.q), Or(o.p, o.q),
    );
}

#[test]
fn other_nodes_round_trip() {
    let _serial = serial();
    let o = operands();
    let loc = || o.loc.clone();

    let not: Not = global("prim.Not").call_tuple((o.p.clone(), loc())).unwrap().try_into().unwrap();
    assert!(structural_equal(&Not::from_complete_fields(loc(), not.ty.clone(), o.p.clone()), &not));
    assert!(!structural_equal(
        &Not::from_complete_fields(loc(), not.ty.clone(), o.q.clone()),
        &not
    ));

    let bitwise_not: BitwiseNot =
        global("prim.BitwiseNot").call_tuple((o.x.clone(), loc())).unwrap().try_into().unwrap();
    let rust = BitwiseNot::from_complete_fields(loc(), bitwise_not.ty.clone(), o.x.clone());
    assert!(structural_equal(&rust, &bitwise_not));

    // `Cast` keeps its target type in the inherited `ty`.
    let cast: Cast = global("prim.Cast")
        .call_tuple((prim_type("float32"), o.x.clone(), loc()))
        .unwrap()
        .try_into()
        .unwrap();
    let target: PrimType = cast.ty.clone().try_cast().unwrap();
    assert_eq!(target.dtype, DLDataType::try_from_str("float32").unwrap());
    assert!(structural_equal(
        &Cast::from_complete_fields(loc(), cast.ty.clone(), o.x.clone()),
        &cast
    ));

    let select: Select = global("prim.Select")
        .call_tuple((o.p.clone(), o.x.clone(), o.y.clone(), loc()))
        .unwrap()
        .try_into()
        .unwrap();
    assert!(structural_equal(&select.true_value, &o.x));
    assert!(structural_equal(&select.false_value, &o.y));
    let rust = Select::from_complete_fields(
        loc(),
        select.ty.clone(),
        o.p.clone(),
        o.x.clone(),
        o.y.clone(),
    );
    assert!(structural_equal(&rust, &select));
    let swapped = Select::from_complete_fields(
        loc(),
        select.ty.clone(),
        o.p.clone(),
        o.y.clone(),
        o.x.clone(),
    );
    assert!(!structural_equal(&swapped, &select));

    let let_: Let = global("prim.Let")
        .call_tuple((o.x_var.clone(), int_imm(&o, 1), o.x.clone(), loc()))
        .unwrap()
        .try_into()
        .unwrap();
    assert!(structural_equal(&let_.body, &o.x));
    let rust = Let::from_complete_fields(
        loc(),
        let_.ty.clone(),
        o.x_var.clone(),
        int_imm(&o, 1),
        o.x.clone(),
    );
    assert!(structural_equal(&rust, &let_));

    // The reflected field `base` is spelled `base_`: `base` is the parent slot.
    let ramp: Ramp = global("prim.Ramp")
        .call_tuple((o.x.clone(), int_imm(&o, 2), int_imm(&o, 4), loc()))
        .unwrap()
        .try_into()
        .unwrap();
    assert!(structural_equal(&ramp.base_, &o.x));
    let rust = Ramp::from_complete_fields(
        loc(),
        ramp.ty.clone(),
        o.x.clone(),
        int_imm(&o, 2),
        int_imm(&o, 4),
    );
    assert!(structural_equal(&rust, &ramp));

    let broadcast: Broadcast = global("prim.Broadcast")
        .call_tuple((o.y.clone(), int_imm(&o, 4), loc()))
        .unwrap()
        .try_into()
        .unwrap();
    let rust =
        Broadcast::from_complete_fields(loc(), broadcast.ty.clone(), o.y.clone(), int_imm(&o, 4));
    assert!(structural_equal(&rust, &broadcast));

    let vectors = Array::<Expr>::new(vec![ramp.clone().into(), broadcast.clone().into()]);
    let indices = Array::<Expr>::new(vec![int_imm(&o, 0), int_imm(&o, 5)]);
    let shuffle: Shuffle = global("prim.Shuffle")
        .call_tuple((vectors.clone(), indices.clone(), loc()))
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(shuffle.vectors.len(), 2);
    assert_eq!(shuffle.indices.len(), 2);
    let rust = Shuffle::from_complete_fields(loc(), shuffle.ty.clone(), vectors, indices);
    assert!(structural_equal(&rust, &shuffle));
}

#[test]
fn expressions_are_matched_by_checked_casts() {
    let _serial = serial();
    let o = operands();
    // (x + y) * x, built natively and taken apart in Rust.
    let sum: Expr = global("prim.Add")
        .call_tuple((o.x.clone(), o.y.clone(), o.loc.clone()))
        .unwrap()
        .try_into()
        .unwrap();
    let product: Expr = global("prim.Mul")
        .call_tuple((sum, o.x.clone(), o.loc.clone()))
        .unwrap()
        .try_into()
        .unwrap();

    assert!(product.clone().try_cast::<Add>().is_err());
    let mul: Mul = product.try_cast().unwrap();
    assert!(mul.a.clone().try_cast::<Sub>().is_err());
    let add: Add = mul.a.clone().try_cast().unwrap();
    assert!(structural_equal(&add.b, &o.y));
    let var: Var = mul.b.clone().try_cast().unwrap();
    assert_eq!(var.name.as_str(), "x");
}
