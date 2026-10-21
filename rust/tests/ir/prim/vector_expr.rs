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

//! Tests of `ir::prim::vector_expr`.

use tvm_tirx::ir::prim;
use tvm_tirx::ir::{Call, Expr, Op, PrimType, Type, Var};
use tvm_tirx::tvm_ffi::{Array, DLDataType, DLDataTypeCode};

use super::{dtype, float, int, ty, var};
use crate::common::{agree, native, serial};
use crate::loc;

#[test]
fn vector_constructors() {
    let _serial = serial();
    let x = var("x", "int32");
    let ramp = prim::Ramp::new(&x, int("int32", 1), int("int32", 4), loc()).unwrap();
    assert_eq!(dtype(&ramp.clone().into()), "int32x4");
    let broadcast = prim::Broadcast::new(&x, int("int32", 4), loc()).unwrap();
    assert_eq!(dtype(&broadcast.clone().into()), "int32x4");
    let indices = [3, 2, 1, 0].map(|i| int("int32", i));
    let shuffle = prim::Shuffle::new([&ramp], indices, loc()).unwrap();
    assert_eq!(dtype(&shuffle.clone().into()), "int32x4");
    assert_eq!(shuffle.indices.len(), 4);
    assert!(prim::Shuffle::new(Vec::<Expr>::new(), [int("int32", 0)], loc()).is_err());
}

/// `vscale * factor`, the lanes of a scalable vector.
fn scalable_lanes(factor: i64) -> Expr {
    let int32: Type = ty("int32").into();
    let vscale = Op::get("prim.vscale").unwrap();
    let vscale =
        Call::new(Some(&int32), vscale, Vec::<Expr>::new(), None, Vec::<Type>::new(), loc());
    prim::Mul::new(vscale.unwrap(), int("int32", factor), loc()).unwrap().into()
}

#[test]
fn vector_nodes_agree_with_cpp() {
    let _serial = serial();
    let x = Expr::from(var("x", "int32"));
    let x64 = Expr::from(var("x64", "int64"));
    let v4 = Expr::from(var("v", "int32x4"));
    let f = Expr::from(var("f", "float32"));
    let lanes_4 = int("int32", 4);
    let factor_first = prim::Mul::new(
        int("int32", 4),
        {
            let int32: Type = ty("int32").into();
            let vscale = Op::get("prim.vscale").unwrap();
            Call::new(Some(&int32), vscale, Vec::<Expr>::new(), None, Vec::<Type>::new(), loc())
                .unwrap()
        },
        loc(),
    )
    .unwrap()
    .into();
    let cases: Vec<(&str, Expr, Expr, Expr)> = vec![
        ("Ramp", x.clone(), int("int32", 1), lanes_4.clone()),
        ("Ramp(int64 lanes)", x.clone(), int("int32", 1), int("int64", 4)),
        ("Ramp(int64 stride constant)", x.clone(), int("int64", 2), lanes_4.clone()),
        ("Ramp(float stride constant)", f.clone(), int("int32", 2), lanes_4.clone()),
        ("Ramp(int32 to int64 stride constant)", x64.clone(), int("int32", -3), lanes_4.clone()),
        ("Ramp(int64 stride)", x.clone(), x64.clone(), lanes_4.clone()),
        ("Ramp(int stride of float base)", f.clone(), x.clone(), lanes_4.clone()),
        (
            "Ramp(float stride constant of int base)",
            x.clone(),
            float("float32", 2.5),
            lanes_4.clone(),
        ),
        ("Ramp(one lane)", x.clone(), int("int32", 1), int("int32", 1)),
        ("Ramp(vector base)", v4.clone(), int("int32", 1), lanes_4.clone()),
        ("Ramp(vector stride)", x.clone(), v4.clone(), lanes_4.clone()),
        ("Ramp(scalable)", x.clone(), int("int32", 1), scalable_lanes(4)),
        ("Ramp(scalable, factor first)", x.clone(), int("int32", 1), factor_first),
        ("Ramp(scalable factor 1)", x.clone(), int("int32", 1), scalable_lanes(1)),
        ("Ramp(variable lanes)", x.clone(), int("int32", 1), x.clone()),
    ];
    for (what, base, stride, lanes) in cases {
        let native_ramp = native(
            "prim.Ramp",
            &[(&base).into(), (&stride).into(), (&lanes).into(), (&loc()).into()],
        );
        agree(what, prim::Ramp::new(&base, &stride, &lanes, loc()), native_ramp);
    }

    for (what, value, lanes) in [
        ("Broadcast", f.clone(), lanes_4.clone()),
        ("Broadcast(scalable)", x.clone(), scalable_lanes(8)),
        ("Broadcast(vector)", v4.clone(), lanes_4.clone()),
        ("Broadcast(one lane)", x.clone(), int("int32", 1)),
    ] {
        let native_broadcast =
            native("prim.Broadcast", &[(&value).into(), (&lanes).into(), (&loc()).into()]);
        agree(what, prim::Broadcast::new(&value, &lanes, loc()), native_broadcast);
    }

    let v2 = Expr::from(var("v2", "int32x2"));
    let f4 = Expr::from(var("f4", "float32x4"));
    let scalable_ty = PrimType::from_dtype(DLDataType {
        code: DLDataTypeCode::kDLInt as u8,
        bits: 32,
        lanes: (-4i16) as u16,
    })
    .unwrap();
    let scalable: Expr = Var::new("s", Some(&scalable_ty.into()), loc()).unwrap().into();
    let indices =
        |values: &[i64]| values.iter().map(|value| int("int32", *value)).collect::<Vec<_>>();
    let cases: Vec<(&str, Vec<Expr>, Vec<Expr>)> = vec![
        ("Shuffle", vec![v4.clone(), v2.clone()], indices(&[0, 5, 2])),
        ("Shuffle(scalars)", vec![x.clone(), x.clone()], indices(&[1, 0])),
        ("Shuffle(no vectors)", vec![], indices(&[0])),
        ("Shuffle(no indices)", vec![v4.clone()], vec![]),
        ("Shuffle(element types)", vec![v4.clone(), f4.clone()], indices(&[0])),
        ("Shuffle(too many indices)", vec![v2.clone()], indices(&[0, 1, 0])),
        ("Shuffle(scalable)", vec![scalable], indices(&[0])),
    ];
    for (what, vectors, indices) in cases {
        let (vector_array, index_array): (Array<Expr>, Array<Expr>) =
            (vectors.iter().cloned().collect(), indices.iter().cloned().collect());
        let native_shuffle = native(
            "prim.Shuffle",
            &[(&vector_array).into(), (&index_array).into(), (&loc()).into()],
        );
        agree(what, prim::Shuffle::new(vectors, indices, loc()), native_shuffle);
    }
}
