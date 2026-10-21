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
use tvm_tirx::ir::Expr;

use super::{dtype, float, int, ty, var};
use crate::common::{load_tirx_compiler, serial, structural_equal};
use crate::loc;

#[test]
fn node_constructors() {
    load_tirx_compiler();
    let _serial = serial();
    let (x, y) = (var("x", "int32"), var("y", "int32"));
    let (p, q) = (var("p", "bool"), var("q", "bool"));

    let add = prim::Add::new(&x, &y, loc()).unwrap();
    assert!(structural_equal(&add.a, &Expr::from(&x)));
    assert_eq!(dtype(&add.clone().into()), "int32");
    assert!(prim::Add::new(&x, float("float32", 1.0), loc()).is_err(), "no type promotion");

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

    let cast = prim::Cast::new(ty("float32"), &x, loc()).unwrap();
    assert_eq!(dtype(&cast.into()), "float32");
    let select = prim::Select::new(&p, &x, &y, loc()).unwrap();
    assert!(structural_equal(&select.true_value, &Expr::from(&x)));
    let let_ = prim::Let::new(&x, int("int32", 1), &add, loc()).unwrap();
    assert!(structural_equal(&let_.var, &x));
}
