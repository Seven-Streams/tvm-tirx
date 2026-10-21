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

//! Tests of `ir::stmt`.

use tvm_tirx::ir::*;
use tvm_tirx::tvm_ffi::{Any, Array, Map, String as FfiString};

use super::{int32, int32_var, loc, op, parse_dtype, seq, tensor_var};
use crate::common::{agree, native, serial, structural_equal};

#[test]
fn statements() {
    let _serial = serial();
    let x = int32_var("x");
    let y = int32_var("y");
    let bind = Bind::new(&y, op("prim._OpAdd", &x, int32(1)), loc()).unwrap();
    assert!(structural_equal(&bind.var, &y));
    let evaluate = Evaluate::new(&x, loc()).unwrap();
    let ret = Return::new(&x, loc()).unwrap();
    assert!(structural_equal(&ret.value, &x));

    let body = seq([Stmt::from(&bind), Stmt::from(&evaluate)]);
    assert_eq!(body.seq.len(), 2);
    let nested = SeqStmt::new([Stmt::from(&body), Stmt::from(&ret)], Some(&loc())).unwrap();
    assert_eq!(nested.seq.len(), 3);

    let kind = ForKind::Unrolled;
    let annotations: Map<FfiString, Any> =
        [(FfiString::from("note"), Any::from(1i64))].into_iter().collect();
    let for_ =
        For::new(&x, int32(0), int32(16), int32(1), kind, &evaluate, Some(&annotations), loc())
            .unwrap();
    assert_eq!(for_.kind, kind as i32);
    assert_eq!(for_.body.seq.len(), 1);
    let less = op("prim._OpLT", &x, int32(4));
    let while_ = While::checked(&less, &body, loc()).unwrap();
    assert_eq!(while_.body.seq.len(), 2);
    let if_ = If::new(&less, &evaluate, Some(&Stmt::from(&ret)), loc()).unwrap();
    assert!(if_.else_case.is_some());
    assert!(If::new(&less, &evaluate, None, loc()).unwrap().else_case.is_none());

    let message = [StringImm::new("boom", loc()).unwrap()];
    let kind_name = StringImm::new("RuntimeError", loc()).unwrap();
    let assert_ = AssertStmt::new(&less, &kind_name, &message, loc()).unwrap();
    assert_eq!(assert_.message_parts.len(), 1);

    let a = tensor_var("A");
    let one = FloatImm::float32(1.0);
    let store = TensorStore::new(&a, [&x], &one, loc()).unwrap();
    assert_eq!(store.indices.len(), 1);

    let attrs = DictAttrs::new(Map::new());
    let region = RegionStmt::new(
        Op::get("tirx.device_scope").unwrap(),
        Vec::<Expr>::new(),
        Vec::<Var>::new(),
        &attrs,
        &evaluate,
        Vec::<Var>::new(),
        loc(),
    )
    .unwrap();
    assert_eq!(region.body.seq.len(), 1);
}

/// A variable of the primitive type `dtype`.
fn prim_var(name: &str, dtype_name: &str) -> Var {
    let ty: Type = PrimType::from_dtype(parse_dtype(dtype_name)).unwrap().into();
    Var::new(name, Some(&ty), loc()).unwrap()
}

fn int_imm(dtype_name: &str, value: i64) -> Expr {
    IntImm::from_dtype(parse_dtype(dtype_name), value, loc()).unwrap().into()
}

/// A `For` case: its name, variable, begin, end, step, body and annotations.
type ForCase<'a> = (&'a str, &'a Var, Expr, Expr, Expr, Stmt, Option<Map<FfiString, Any>>);

/// A `RegionStmt` case: its name, operator, arguments, body parameters, attributes and results.
type RegionCase<'a> = (&'a str, &'a Op, Vec<Expr>, Vec<Var>, &'a DictAttrs, Vec<Var>);

#[test]
fn statements_agree_with_cpp() {
    let _serial = serial();
    let (x, y) = (prim_var("x", "int32"), prim_var("y", "int32"));
    let (f, p) = (prim_var("f", "float32"), prim_var("p", "bool"));
    let evaluate: Stmt = Evaluate::new(&x, loc()).unwrap().into();
    let no_op: Stmt = Evaluate::new(int32(0), loc()).unwrap().into();

    for (what, var, value) in [("Bind", &y, Expr::from(&x)), ("Bind(type)", &y, Expr::from(&f))] {
        let native_bind = native("ir.Bind", &[var.into(), (&value).into(), (&loc()).into()]);
        agree(what, Bind::new(var, &value, loc()), native_bind);
    }

    let kind = StringImm::new("RuntimeError", loc()).unwrap();
    let parts: Array<StringImm> = [StringImm::new("failed", loc()).unwrap()].into_iter().collect();
    for (what, condition) in
        [("AssertStmt", Expr::from(&p)), ("AssertStmt(condition)", Expr::from(&x))]
    {
        let native_assert = native(
            "ir.AssertStmt",
            &[(&condition).into(), (&kind).into(), (&parts).into(), (&loc()).into()],
        );
        agree(what, AssertStmt::new(&condition, &kind, parts.iter(), loc()), native_assert);
    }

    let (i, i64_var) = (prim_var("i", "int32"), prim_var("j", "int64"));
    let note: Map<FfiString, Any> =
        [(FfiString::from("note"), Any::from(1i64))].into_iter().collect();
    let body_seq: Stmt = SeqStmt::new([evaluate.clone(), evaluate.clone()], None).unwrap().into();
    let cases: Vec<ForCase> = vec![
        ("For", &i, int32(0), int32(8), int32(1), evaluate.clone(), None),
        ("For(annotations)", &i, int32(0), int32(8), int32(1), body_seq.clone(), Some(note)),
        ("For(no-op body)", &i, int32(0), int32(8), int32(1), no_op.clone(), None),
        (
            "For(int64 var, int32 bounds)",
            &i64_var,
            int32(0),
            int32(8),
            int32(2),
            evaluate.clone(),
            None,
        ),
        (
            "For(int64 bound literal)",
            &i,
            int_imm("int64", 0),
            int_imm("int64", 8),
            int32(1),
            evaluate.clone(),
            None,
        ),
        (
            "For(literal out of range)",
            &i,
            int32(0),
            int_imm("int64", 1 << 40),
            int32(1),
            evaluate.clone(),
            None,
        ),
        ("For(uint8 literal)", &i, int_imm("uint8", 3), int32(8), int32(1), evaluate.clone(), None),
        ("For(narrower var)", &i, int32(0), Expr::from(&i64_var), int32(1), evaluate.clone(), None),
        ("For(float var)", &f, int32(0), int32(8), int32(1), evaluate.clone(), None),
        ("For(float bound)", &i, int32(0), Expr::from(&f), int32(1), evaluate.clone(), None),
        (
            "For(mismatched end)",
            &i64_var,
            int_imm("int64", 0),
            Expr::from(&x),
            int_imm("int64", 1),
            evaluate.clone(),
            None,
        ),
    ];
    for (what, var, begin, end, step, body, annotations) in cases {
        let kind = ForKind::Parallel as i32;
        let native_for = native(
            "ir.For",
            &[
                var.into(),
                (&begin).into(),
                (&end).into(),
                (&step).into(),
                (&kind).into(),
                (&body).into(),
                (&annotations).into(),
                (&loc()).into(),
            ],
        );
        let rust = For::new(
            var,
            &begin,
            &end,
            &step,
            ForKind::Parallel,
            &body,
            annotations.as_ref(),
            loc(),
        );
        agree(what, rust, native_for);
    }

    let v4 = prim_var("v", "int32x4");
    for (what, condition) in
        [("While", Expr::from(&p)), ("While(vector condition)", Expr::from(&v4))]
    {
        let native_while =
            native("ir.While", &[(&condition).into(), (&evaluate).into(), (&loc()).into()]);
        agree(what, While::checked(&condition, &evaluate, loc()), native_while);
    }

    let a = tensor_var("A");
    let one: Expr = FloatImm::float32(1.0).into();
    let stores: Vec<(&str, Expr, Vec<Expr>, Expr)> = vec![
        ("TensorStore", Expr::from(&a), vec![Expr::from(&x)], one.clone()),
        ("TensorStore(rank)", Expr::from(&a), vec![Expr::from(&x), Expr::from(&x)], one.clone()),
        ("TensorStore(value type)", Expr::from(&a), vec![Expr::from(&x)], Expr::from(&x)),
        ("TensorStore(no tensor)", Expr::from(&x), vec![Expr::from(&x)], one.clone()),
    ];
    for (what, dest, indices, value) in stores {
        let array: Array<Expr> = indices.iter().cloned().collect();
        let native_store = native(
            "ir.TensorStore",
            &[(&dest).into(), (&array).into(), (&value).into(), (&loc()).into()],
        );
        agree(what, TensorStore::new(&dest, indices, &value, loc()), native_store);
    }

    let nested: Stmt =
        SeqStmt::new([evaluate.clone(), no_op.clone(), evaluate.clone()], None).unwrap().into();
    let sequences: Vec<(&str, Vec<Stmt>, Option<Location>)> = vec![
        ("SeqStmt", vec![evaluate.clone(), evaluate.clone()], None),
        ("SeqStmt(nested and no-ops)", vec![nested.clone(), no_op.clone(), evaluate.clone()], None),
        ("SeqStmt(empty)", vec![], None),
        ("SeqStmt(located)", vec![evaluate.clone()], Some(loc())),
    ];
    for (what, stmts, location) in sequences {
        let array: Array<Stmt> = stmts.iter().cloned().collect();
        let native_seq = native("ir.SeqStmt", &[(&array).into(), (&location).into()]);
        agree(what, SeqStmt::new(stmts, location.as_ref()), native_seq);
    }

    let ifs: Vec<(&str, Expr, Stmt, Option<Stmt>)> = vec![
        ("If", Expr::from(&p), evaluate.clone(), None),
        ("If(else)", Expr::from(&p), nested.clone(), Some(no_op.clone())),
        ("If(int condition)", Expr::from(&x), evaluate.clone(), Some(evaluate.clone())),
    ];
    for (what, condition, then_case, else_case) in ifs {
        let native_if = native(
            "ir.If",
            &[(&condition).into(), (&then_case).into(), (&else_case).into(), (&loc()).into()],
        );
        agree(what, If::new(&condition, &then_case, else_case.as_ref(), loc()), native_if);
    }

    let native_evaluate = native("ir.Evaluate", &[(&Expr::from(&x)).into(), (&loc()).into()]);
    agree("Evaluate", Evaluate::new(&x, loc()), native_evaluate);
    let native_return = native("ir.Return", &[(&Expr::from(&x)).into(), (&loc()).into()]);
    agree("Return", Return::new(&x, loc()), native_return);
}

#[test]
fn regions_agree_with_cpp() {
    let _serial = serial();
    let launch = Op::get("tirx.launch_thread").unwrap();
    let compute = Op::get("tirx.compute_scope").unwrap();
    let exp = Op::get("prim.exp").unwrap();
    let tag = |text: &str| -> Expr { StringImm::new(text, loc()).unwrap().into() };
    let (t, t64, r) = (prim_var("t", "int32"), prim_var("t64", "int64"), prim_var("r", "int32"));
    let body: Stmt = Evaluate::new(&t, loc()).unwrap().into();
    let no_attrs = DictAttrs::new(Map::new());
    let note: Map<FfiString, Any> =
        [(FfiString::from("note"), Any::from(1i64))].into_iter().collect();
    let attrs = DictAttrs::new(note);
    let cases: Vec<RegionCase> = vec![
        (
            "RegionStmt",
            &launch,
            vec![tag("threadIdx.x"), int32(8)],
            vec![t.clone()],
            &no_attrs,
            vec![],
        ),
        (
            "RegionStmt(parameter count)",
            &launch,
            vec![tag("threadIdx.x"), int32(8)],
            vec![],
            &no_attrs,
            vec![],
        ),
        (
            "RegionStmt(parameter type)",
            &launch,
            vec![tag("threadIdx.x"), int32(8)],
            vec![t64.clone()],
            &no_attrs,
            vec![],
        ),
        (
            "RegionStmt(results)",
            &launch,
            vec![tag("threadIdx.x"), int32(8)],
            vec![t.clone()],
            &no_attrs,
            vec![r.clone()],
        ),
        (
            "RegionStmt(repeated definition)",
            &launch,
            vec![tag("threadIdx.x"), int32(8)],
            vec![t.clone()],
            &no_attrs,
            vec![t.clone()],
        ),
        (
            "RegionStmt(empty tag)",
            &launch,
            vec![tag(""), int32(8)],
            vec![t.clone()],
            &no_attrs,
            vec![],
        ),
        ("RegionStmt(signature)", &launch, vec![int32(8)], vec![t.clone()], &no_attrs, vec![]),
        ("RegionStmt(no body parameters)", &compute, vec![tag("work")], vec![], &no_attrs, vec![]),
        ("RegionStmt(attrs)", &compute, vec![tag("work")], vec![], &attrs, vec![]),
        ("RegionStmt(not a region)", &exp, vec![int32(1)], vec![], &no_attrs, vec![]),
    ];
    for (what, op, args, params, attrs, results) in cases {
        let (args_array, params_array, results_array): (Array<Expr>, Array<Var>, Array<Var>) = (
            args.iter().cloned().collect(),
            params.iter().cloned().collect(),
            results.iter().cloned().collect(),
        );
        let native_region = native(
            "ir.RegionStmt",
            &[
                op.into(),
                (&args_array).into(),
                (&params_array).into(),
                attrs.into(),
                (&body).into(),
                (&results_array).into(),
                (&loc()).into(),
            ],
        );
        let rust = RegionStmt::new(op, args, params, attrs, &body, results, loc());
        agree(what, rust, native_region);
    }
}
