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
use tvm_tirx::ir::{UNKNOWN_LOC, Location, Call, Expr, IntImm, Op, PrimType, Type, Var};
use tvm_tirx::tvm_ffi::Array;

use super::{dtype, var};
use crate::common::{agree, native, serial};

#[test]
fn vector_constructors() {
    let _serial = serial();
    let x = var("x", PrimType::int(32));
    let ramp = prim::Ramp::new(&x, 1, 4, UNKNOWN_LOC).unwrap();
    assert_eq!(dtype(&ramp.clone().into()), "int32x4");
    let broadcast = prim::Broadcast::new(&x, 4, UNKNOWN_LOC).unwrap();
    assert_eq!(dtype(&broadcast.clone().into()), "int32x4");
    let indices = [3, 2, 1, 0].map(Expr::from);
    let shuffle = prim::Shuffle::new([&ramp], indices, UNKNOWN_LOC).unwrap();
    assert_eq!(dtype(&shuffle.clone().into()), "int32x4");
    assert_eq!(shuffle.indices.len(), 4);
    assert!(prim::Shuffle::new(Vec::<Expr>::new(), [Expr::from(0)], UNKNOWN_LOC).is_err());
}

/// `vscale * factor`, the lanes of a scalable vector.
fn scalable_lanes(factor: i32) -> Expr {
    let int32: Type = PrimType::int(32).into();
    let vscale = Op::get("prim.vscale").unwrap();
    let vscale =
        Call::new(Some(&int32), vscale, Vec::<Expr>::new(), None, Vec::<Type>::new(), UNKNOWN_LOC);
    prim::Mul::new(vscale.unwrap(), factor, UNKNOWN_LOC).unwrap().into()
}

#[test]
fn vector_nodes_agree_with_cpp() {
    let _serial = serial();
    let x = Expr::from(var("x", PrimType::int(32)));
    let x64 = Expr::from(var("x64", PrimType::int(64)));
    let x16 = Expr::from(var("x16", PrimType::int(16)));
    let u32 = Expr::from(var("u32", PrimType::uint(32)));
    let v4 = Expr::from(var("v", PrimType::int(32).with_lanes(4).unwrap()));
    let bf = Expr::from(var("bf", PrimType::bfloat(32)));
    let f = Expr::from(var("f", PrimType::float(32)));
    let lanes_4 = Expr::from(4);
    let factor_first = prim::Mul::new(
        4,
        {
            let int32: Type = PrimType::int(32).into();
            let vscale = Op::get("prim.vscale").unwrap();
            Call::new(
                Some(&int32),
                vscale,
                Vec::<Expr>::new(),
                None,
                Vec::<Type>::new(),
                UNKNOWN_LOC,
            )
            .unwrap()
        },
        UNKNOWN_LOC,
    )
    .unwrap()
    .into();
    let cases: Vec<(&str, Expr, Expr, Expr)> = vec![
        ("Ramp", x.clone(), Expr::from(1), lanes_4.clone()),
        ("Ramp(int64 lanes)", x.clone(), Expr::from(1), IntImm::int64(4).into()),
        ("Ramp(int64 stride constant)", x.clone(), IntImm::int64(2).into(), lanes_4.clone()),
        ("Ramp(uint32 base)", u32.clone(), Expr::from(1), lanes_4.clone()),
        ("Ramp(int16 base)", x16.clone(), Expr::from(1), lanes_4.clone()),
        ("Ramp(int16 stride)", x.clone(), x16.clone(), lanes_4.clone()),
        ("Ramp(float base)", f.clone(), Expr::from(2), lanes_4.clone()),
        ("Ramp(int32 to int64 stride constant)", x64.clone(), Expr::from(-3), lanes_4.clone()),
        ("Ramp(int64 stride)", x.clone(), x64.clone(), lanes_4.clone()),
        ("Ramp(int stride of float base)", f.clone(), x.clone(), lanes_4.clone()),
        ("Ramp(float stride constant of int base)", x.clone(), Expr::from(2.5f32), lanes_4.clone()),
        ("Ramp(stride constant of bfloat32 base)", bf.clone(), Expr::from(1), lanes_4.clone()),
        ("Ramp(one lane)", x.clone(), Expr::from(1), Expr::from(1)),
        ("Ramp(vector base)", v4.clone(), Expr::from(1), lanes_4.clone()),
        ("Ramp(vector stride)", x.clone(), v4.clone(), lanes_4.clone()),
        ("Ramp(scalable)", x.clone(), Expr::from(1), scalable_lanes(4)),
        ("Ramp(scalable, factor first)", x.clone(), Expr::from(1), factor_first),
        ("Ramp(scalable factor 1)", x.clone(), Expr::from(1), scalable_lanes(1)),
        ("Ramp(variable lanes)", x.clone(), Expr::from(1), x.clone()),
    ];
    for (what, base, stride, lanes) in cases {
        let native_ramp = native(
            "prim.Ramp",
            &[(&base).into(), (&stride).into(), (&lanes).into(), (&Location::unknown()).into()],
        );
        agree(what, prim::Ramp::new(&base, &stride, &lanes, UNKNOWN_LOC), native_ramp);
    }

    for (what, value, lanes) in [
        ("Broadcast", f.clone(), lanes_4.clone()),
        ("Broadcast(scalable)", x.clone(), scalable_lanes(8)),
        ("Broadcast(vector)", v4.clone(), lanes_4.clone()),
        ("Broadcast(one lane)", x.clone(), Expr::from(1)),
    ] {
        let native_broadcast = native(
            "prim.Broadcast",
            &[(&value).into(), (&lanes).into(), (&Location::unknown()).into()],
        );
        agree(what, prim::Broadcast::new(&value, &lanes, UNKNOWN_LOC), native_broadcast);
    }

    let v2 = Expr::from(var("v2", PrimType::int(32).with_lanes(2).unwrap()));
    let f4 = Expr::from(var("f4", PrimType::float(32).with_lanes(4).unwrap()));
    let scalable_ty = PrimType::int(32).with_scalable_lanes(4).unwrap();
    let scalable: Expr = Var::new("s", scalable_ty, UNKNOWN_LOC).into();
    let indices =
        |values: &[i32]| values.iter().map(|value| Expr::from(*value)).collect::<Vec<_>>();
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
            &[(&vector_array).into(), (&index_array).into(), (&Location::unknown()).into()],
        );
        agree(what, prim::Shuffle::new(vectors, indices, UNKNOWN_LOC), native_shuffle);
    }
}
