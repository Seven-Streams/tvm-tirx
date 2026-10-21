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

//! Tests of `ir::expr`.

use tvm_tirx::ir::*;
use tvm_tirx::tvm_ffi::{
    Any, AnyView, Array, BigInt, Function, ObjectRefCore, Result, String as FfiString,
};

use super::{dtype, int32, int32_var, loc, op, tensor_var};
use crate::common::{agree, agree_fresh, agree_kind, native, serial, structural_equal};

#[test]
fn expressions() {
    let _serial = serial();
    let x = int32_var("x");
    assert_eq!(x.name.as_str(), "x");
    assert_eq!(GlobalVar::new("main", loc()).unwrap().name_hint.as_str(), "main");
    assert_eq!(StringImm::new("hi", loc()).unwrap().value.as_str(), "hi");
    assert_eq!(FloatImm::from_dtype(dtype("float32"), 1.5, loc()).unwrap().value, 1.5);
    assert!(IntImm::from_dtype(dtype("uint8"), 256, loc()).is_err());
    assert_eq!(DataTypeImm::from_dtype(dtype("int8"), loc()).unwrap().value, dtype("int8"));
    let i32_ty = PrimType::from_dtype(dtype("int32")).unwrap();
    let constant = GenericConst::checked(&i32_ty, Any::from(i32_ty.clone()), loc()).unwrap();
    assert!(GenericConst::checked(&i32_ty, Any::from(1i64), loc()).is_err());
    assert!(structural_equal(&constant.ty, &i32_ty));

    let tuple = Tuple::new([Expr::from(&x), int32(1)], loc()).unwrap();
    let item = TupleGetItem::from_tuple(&tuple, 1, loc()).unwrap();
    assert!(structural_equal(&item.ty, &i32_ty));
    assert!(TupleGetItem::from_tuple(&tuple, 2, loc()).is_err());
    let subscripted = Expr::from(&tuple).subscript([Subscript::Index(int32(0))], loc()).unwrap();
    assert!(Expr::try_from(subscripted).is_ok());

    let a = tensor_var("A");
    let load = TensorLoad::new(&a, [&x], loc()).unwrap();
    assert_eq!(load.indices.len(), 1);
    let slice = Expr::from(&a)
        .subscript([Subscript::Slice(Some(int32(0)), Some(int32(4)), None)], loc())
        .unwrap();
    assert!(Expr::try_from(slice).is_ok());

    let exp = Op::get("prim.exp").unwrap();
    let one = FloatImm::from_dtype(dtype("float32"), 1.0, loc()).unwrap();
    let call = Call::new(None, &exp, [&one], None, Vec::<Type>::new(), loc()).unwrap();
    call.validate().unwrap();
    assert!(structural_equal(&call.reinfer_type().unwrap(), &call.ty));

    let lambda = LambdaExpr::new([&x], op("prim._OpAdd", &x, int32(1)), loc()).unwrap();
    let applied = lambda.apply([int32(2)]).unwrap();
    let analyzer = tvm_tirx::sym::Analyzer::new().unwrap();
    assert!(analyzer.can_prove_equal(applied, int32(3)).unwrap());
}

#[test]
fn ranges() {
    let _serial = serial();
    let range = Range::new(int32(8), None, loc()).unwrap();
    assert!(structural_equal(&range.extent, &int32(8)));
    let range = Range::new(int32(2), Some(&int32(8)), loc()).unwrap();
    assert!(structural_equal(&range.extent, &int32(6)));
    let range = Range::from_min_extent(int32(2), int32(8), loc()).unwrap();
    assert!(structural_equal(&range.extent, &int32(8)));
    let extent = Range::get_extent_from_begin_end(int32(2), int32(8)).unwrap();
    assert!(structural_equal(&extent, &int32(6)));
}

#[test]
fn constants_agree_with_cpp() {
    let _serial = serial();
    let float_imm = |name: &str, value: f64| {
        let native =
            native("ir.FloatImm", &[(&dtype(name)).into(), (&value).into(), (&loc()).into()]);
        let what = format!("FloatImm({name}, {value})");
        agree(&what, FloatImm::from_dtype(dtype(name), value, loc()), native);
    };
    for (name, value) in [
        ("float32", 1.5),
        ("float32", f64::INFINITY),
        ("float32", f64::NAN),
        ("float32", 1e39),
        ("float32", -1e39),
        ("float16", 1e9),
        ("float16", -65504.0),
        ("bfloat16", 1e39),
        ("float64", 1e300),
        ("float8_e4m3fn", 500.0),
        ("float8_e4m3fnuz", -1.0),
        ("float8_e5m2", -60000.0),
        ("float8_e8m0fnu", 1e39),
        ("float6_e2m3fn", 8.0),
        ("float4_e2m1fn", -7.0),
        ("int32", 1.0),
        ("float32x4", 1.0),
    ] {
        float_imm(name, value);
    }

    let int_imm = |name: &str, value: BigInt| {
        let native =
            native("ir.IntImm", &[(&dtype(name)).into(), (&value).into(), (&loc()).into()]);
        let what = format!("IntImm({name}, {value})");
        agree(&what, IntImm::from_dtype(dtype(name), value.clone(), loc()), native);
    };
    let big = |text: &str| -> BigInt { text.parse().unwrap() };
    for (name, value) in [
        ("int32", big("7")),
        ("int8", big("300")),
        ("int8", big("-129")),
        ("int64", big("-9223372036854775808")),
        ("int128", big("170141183460469231731687303715884105727")),
        ("int128", big("170141183460469231731687303715884105728")),
        ("uint8", big("256")),
        ("uint8", big("-1")),
        ("uint64", big("18446744073709551615")),
        ("uint128", big("340282366920938463463374607431768211456")),
        ("bool", big("1")),
        ("bool", big("2")),
        ("int1", big("-1")),
        ("int32x4", big("1")),
        ("float32", big("1")),
    ] {
        int_imm(name, value);
    }

    let i32_ty = PrimType::from_dtype(dtype("int32")).unwrap();
    let missing: Type = MissingType::new().unwrap().into();
    let generic = |what: &str, ty: &Type, value: Any| {
        let native = native("ir.GenericConst", &[ty.into(), (&value).into(), (&loc()).into()]);
        agree(what, GenericConst::checked(ty, value.clone(), loc()), native);
    };
    let i32_ty_as_type: Type = i32_ty.clone().into();
    generic("GenericConst(object)", &i32_ty_as_type, Any::from(i32_ty.clone()));
    generic("GenericConst(int)", &i32_ty_as_type, Any::from(1i64));
    generic("GenericConst(bool)", &i32_ty_as_type, Any::from(true));
    generic("GenericConst(float)", &i32_ty_as_type, Any::from(1.0f64));
    generic("GenericConst(string)", &i32_ty_as_type, Any::from(FfiString::from("s")));
    generic("GenericConst(missing type)", &missing, Any::from(i32_ty.clone()));

    let int8 = dtype("int8");
    let native_dtype = native("ir.DataTypeImm", &[(&int8).into(), (&loc()).into()]);
    agree("DataTypeImm", DataTypeImm::from_dtype(int8, loc()), native_dtype);
    let text = FfiString::from("hi");
    let native_string = native("ir.StringImm", &[(&text).into(), (&loc()).into()]);
    agree("StringImm", StringImm::new("hi", loc()), native_string);
}

#[test]
fn shared_nodes_are_the_native_ones() {
    let _serial = serial();
    let unknown: Location = native("ir.UnknownLoc", &[]).unwrap().try_into().unwrap();
    assert!(UnknownLoc::new().unwrap().same_as(&unknown));
    let missing: Type = native("ir.MissingType", &[]).unwrap().try_into().unwrap();
    assert!(MissingType::new().unwrap().same_as(&missing));
    let int32_dtype = dtype("int32");
    let int32_ty: Type =
        native("ir.PrimType", &[(&int32_dtype).into()]).unwrap().try_into().unwrap();
    assert!(PrimType::from_dtype(int32_dtype).unwrap().same_as(&int32_ty));
}

#[test]
fn expressions_agree_with_cpp() {
    let _serial = serial();
    let (x, n) = (int32_var("x"), int32_var("n"));
    let untyped = Var::new("u", None, loc()).unwrap();
    let i32_ty: Type = PrimType::from_dtype(dtype("int32")).unwrap().into();

    let name = FfiString::from("main");
    let native_global = native("ir.GlobalVar", &[(&name).into(), (&loc()).into()]);
    agree_fresh("GlobalVar", GlobalVar::new("main", loc()), native_global);
    let native_var =
        native("ir.Var", &[(&FfiString::from("x")).into(), (&i32_ty).into(), (&loc()).into()]);
    agree_fresh("Var", Var::new("x", Some(&i32_ty), loc()), native_var);
    let no_type = Option::<Type>::None;
    let native_untyped =
        native("ir.Var", &[(&FfiString::from("u")).into(), (&no_type).into(), (&loc()).into()]);
    agree_fresh("Var(untyped)", Var::new("u", None, loc()), native_untyped);

    for (what, fields) in [
        ("Tuple(typed)", vec![Expr::from(&x), int32(1)]),
        ("Tuple(untyped)", vec![Expr::from(&x), Expr::from(&untyped)]),
        ("Tuple(empty)", vec![]),
    ] {
        let array: Array<Expr> = fields.iter().cloned().collect();
        let native_tuple = native("ir.Tuple", &[(&array).into(), (&loc()).into()]);
        agree(what, Tuple::new(fields, loc()), native_tuple);
    }

    let tuple: Expr = Tuple::new([Expr::from(&x), int32(1)], loc()).unwrap().into();
    for (what, value, index) in [
        ("TupleGetItem", tuple.clone(), 1),
        ("TupleGetItem(out of bounds)", tuple.clone(), 2),
        ("TupleGetItem(negative)", tuple.clone(), -1),
        ("TupleGetItem(untyped)", Expr::from(&untyped), 0),
    ] {
        let native_item =
            native("ir.TupleGetItem", &[(&value).into(), (&index).into(), (&loc()).into()]);
        agree(what, TupleGetItem::from_tuple(&value, index, loc()), native_item);
    }

    let a = tensor_var("A");
    let load = |what: &str, source: Expr, indices: Vec<Expr>| {
        let array: Array<Expr> = indices.iter().cloned().collect();
        let native_load =
            native("ir.TensorLoad", &[(&source).into(), (&array).into(), (&loc()).into()]);
        agree(what, TensorLoad::new(&source, indices, loc()), native_load);
    };
    load("TensorLoad", Expr::from(&a), vec![Expr::from(&x)]);
    load("TensorLoad(rank)", Expr::from(&a), vec![Expr::from(&x), Expr::from(&x)]);
    load("TensorLoad(no tensor)", Expr::from(&x), vec![Expr::from(&x)]);
    let text: Expr = StringImm::new("s", loc()).unwrap().into();
    let array: Array<Expr> = [text.clone()].into_iter().collect();
    let native_load =
        native("ir.TensorLoad", &[(&Expr::from(&a)).into(), (&array).into(), (&loc()).into()]);
    agree_kind("TensorLoad(index)", TensorLoad::new(&a, [text], loc()), native_load);

    let vars: Array<Var> = [x.clone()].into_iter().collect();
    let body = op("prim._OpAdd", &x, int32(1));
    let native_lambda = native("ir.LambdaExpr", &[(&vars).into(), (&body).into(), (&loc()).into()]);
    agree("LambdaExpr", LambdaExpr::new([&x], &body, loc()), native_lambda);

    let x_plus_4 = op("prim._OpAdd", &x, int32(4));
    let four_plus_x = op("prim._OpAdd", int32(4), &x);
    for (what, begin, end) in [
        ("Range(x, x + 4)", Expr::from(&x), Some(x_plus_4.clone())),
        ("Range(x, 4 + x)", Expr::from(&x), Some(four_plus_x)),
        ("Range(0, n)", int32(0), Some(Expr::from(&n))),
        ("Range(x, n)", Expr::from(&x), Some(Expr::from(&n))),
        ("Range(n)", Expr::from(&n), None),
    ] {
        let native_range = native("ir.Range", &[(&begin).into(), (&end).into(), (&loc()).into()]);
        agree(what, Range::new(&begin, end.as_ref(), loc()), native_range);
    }
    let native_min_extent = native(
        "ir.RangeFromMinExtent",
        &[(&Expr::from(&x)).into(), (&int32(4)).into(), (&loc()).into()],
    );
    agree("Range::from_min_extent", Range::from_min_extent(&x, int32(4), loc()), native_min_extent);
    let native_extent =
        native("ir.RangeGetExtentFromBeginEnd", &[(&Expr::from(&x)).into(), (&x_plus_4).into()]);
    agree(
        "Range::get_extent_from_begin_end",
        Range::get_extent_from_begin_end(&x, &x_plus_4),
        native_extent,
    );
}

/// A test operator `name` with the attributes `attrs`, registered once per test binary.
fn test_op(name: &str, attrs: &[(&str, Any)]) -> Op {
    if let Ok(op) = Op::get(name) {
        return op;
    }
    Op::register(name, "").unwrap();
    let op = Op::get(name).unwrap();
    let set_attr = Function::from_type_key_method("ir.Op", "_set_attr").unwrap();
    for (key, value) in attrs {
        let key = FfiString::from(*key);
        set_attr
            .call_packed(&[(&op).into(), (&key).into(), value.into(), (&false).into()])
            .unwrap();
    }
    op
}

#[test]
fn calls_agree_with_cpp() {
    let _serial = serial();
    let x = int32_var("x");
    let f32_ty: Type = PrimType::from_dtype(dtype("float32")).unwrap().into();
    let one: Expr = FloatImm::from_dtype(dtype("float32"), 1.0, loc()).unwrap().into();
    let i16_ty: Type = PrimType::from_dtype(dtype("int16")).unwrap().into();
    let returns = |value: Any| {
        Function::from_packed(move |_args: &[AnyView]| -> Result<Any> { Ok(value.clone()) })
    };
    let packed_infer = test_op(
        "rust_test.packed_infer",
        &[("FInferType", returns(Any::from(i16_ty.clone())).into())],
    );
    let packed_mistyped =
        test_op("rust_test.packed_mistyped", &[("FInferType", returns(Any::from(5i64)).into())]);
    let fixed =
        test_op("rust_test.fixed_return", &[("TFixedReturnType", Any::from(f32_ty.clone()))]);
    let plain = test_op("rust_test.no_inference", &[]);

    let callee_ty: Type = FuncType::new([f32_ty.clone()], i16_ty.clone()).unwrap().into();
    let callee = Var::new("f", Some(&callee_ty), loc()).unwrap();
    let cases: Vec<(&str, Option<Type>, Expr, Vec<Expr>)> = vec![
        (
            "Call(typed)",
            Some(f32_ty.clone()),
            Op::get("prim.exp").unwrap().into(),
            vec![one.clone()],
        ),
        ("Call(FInferType)", None, Op::get("prim.exp").unwrap().into(), vec![one.clone()]),
        ("Call(packed FInferType)", None, packed_infer.into(), vec![Expr::from(&x)]),
        ("Call(mistyped packed FInferType)", None, packed_mistyped.into(), vec![Expr::from(&x)]),
        ("Call(TFixedReturnType)", None, fixed.into(), vec![]),
        ("Call(no inference)", None, plain.into(), vec![]),
        ("Call(function callee)", None, Expr::from(&callee), vec![one.clone()]),
        ("Call(untyped callee)", None, Expr::from(&x), vec![]),
    ];
    for (what, ty, op, args) in cases {
        let array: Array<Expr> = args.iter().cloned().collect();
        let (attrs, ty_args) = (Option::<Attrs>::None, Array::<Type>::new(vec![]));
        let native_call = native(
            "ir.Call",
            &[
                (&ty).into(),
                (&op).into(),
                (&array).into(),
                (&attrs).into(),
                (&ty_args).into(),
                (&loc()).into(),
            ],
        );
        let rust = Call::new(ty.as_ref(), &op, args, None, Vec::<Type>::new(), loc());
        agree(what, rust.clone(), native_call);
        if let Ok(call) = rust {
            let native_type = native("ir.CallReinferType", &[(&call).into()]);
            agree(&format!("{what}.reinfer_type"), call.reinfer_type(), native_type);
        }
    }

    // `prim.exp` validates its argument count and type.
    let exp: Expr = Op::get("prim.exp").unwrap().into();
    for (what, args) in [
        ("Call::validate", vec![one.clone()]),
        ("Call::validate(arity)", vec![one.clone(), one.clone()]),
    ] {
        let call = Call::new(Some(&f32_ty), &exp, args, None, Vec::<Type>::new(), loc()).unwrap();
        let native_validate = native("ir.CallValidate", &[(&call).into()]).map(|_| ());
        match (call.validate(), native_validate) {
            (Ok(()), Ok(())) => {}
            (Err(rust), Err(native)) => {
                assert_eq!(
                    (rust.kind(), rust.message()),
                    (native.kind(), native.message()),
                    "{what}"
                )
            }
            (rust, native) => panic!("{what}: {rust:?} vs. {native:?}"),
        }
    }
}
