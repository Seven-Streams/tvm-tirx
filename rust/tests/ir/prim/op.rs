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

//! Tests of `ir::prim::op`: the type-promoting operators.

use tvm_tirx::ir::prim::{self, op};
use tvm_tirx::ir::{Expr, FloatImm};
use tvm_tirx::tvm_ffi::ObjectRefCast;

use super::{dtype, float, int, int_value, ty, var};
use crate::common::{serial, structural_equal};
use crate::loc;

#[test]
fn arithmetic_operators() {
    let _serial = serial();
    let x = var("x", "int32");

    // Promotion: int32 + float32 is float32, and constants fold.
    assert_eq!(dtype(&op::add(&x, float("float32", 1.0), loc()).unwrap()), "float32");
    assert_eq!(dtype(&op::add(&x, int("int64", 1), loc()).unwrap()), "int64");
    macro_rules! folded {
        ($op:path, $a:expr, $b:expr) => {
            int_value($op(int("int32", $a), int("int32", $b), loc()).unwrap())
        };
    }
    assert_eq!(folded!(op::add, 7, 2), 9);
    assert_eq!(folded!(op::sub, 7, 2), 5);
    assert_eq!(folded!(op::mul, 7, 2), 14);
    assert_eq!(folded!(op::div, -7, 2), -3);
    assert_eq!(folded!(op::truncdiv, -7, 2), -3);
    assert_eq!(folded!(op::truncmod, -7, 2), -1);
    assert_eq!(folded!(op::floordiv, -7, 2), -4);
    assert_eq!(folded!(op::floormod, -7, 2), 1);
    assert_eq!(folded!(op::indexdiv, 7, 2), 3);
    assert_eq!(folded!(op::indexmod, 7, 2), 1);
    assert_eq!(folded!(op::ceildiv, 7, 2), 4);
    assert_eq!(folded!(op::min, 7, 2), 2);
    assert_eq!(folded!(op::max, 7, 2), 7);
    assert_eq!(folded!(op::bitwise_and, 6, 3), 2);
    assert_eq!(folded!(op::bitwise_or, 6, 3), 7);
    assert_eq!(folded!(op::bitwise_xor, 6, 3), 5);
    assert_eq!(folded!(op::shift_left, 1, 3), 8);
    assert_eq!(folded!(op::shift_right, 8, 3), 1);
    let not = op::bitwise_not(int("int32", 0), loc()).unwrap();
    assert!(not.try_cast::<prim::BitwiseNot>().is_ok(), "the node, not folded");

    // On a variable they build the nodes.
    let floordiv = op::floordiv(&x, int("int32", 2), loc()).unwrap();
    assert!(floordiv.try_cast::<prim::FloorDiv>().is_ok());
    assert_eq!(
        dtype(&op::pow(float("float32", 2.0), float("float32", 3.0), loc()).unwrap()),
        "float32"
    );
}

#[test]
fn comparison_and_logic_operators() {
    let _serial = serial();
    let x = var("x", "int32");
    let one = int("int32", 1);
    for cmp in [op::eq, op::ne, op::lt, op::le, op::gt, op::ge] {
        assert_eq!(dtype(&cmp(Expr::from(&x), one.clone(), loc()).unwrap()), "bool");
    }
    assert!(op::lt(&x, &one, loc()).unwrap().try_cast::<prim::LT>().is_ok());

    let conds = [op::lt(&x, &one, loc()).unwrap(), op::gt(&x, int("int32", -5), loc()).unwrap()];
    assert!(op::all(conds.clone(), loc()).unwrap().try_cast::<prim::And>().is_ok());
    assert!(op::any(conds.clone(), loc()).unwrap().try_cast::<prim::Or>().is_ok());
    assert!(structural_equal(&op::all([&conds[0]], loc()).unwrap(), &conds[0]));
    assert!(op::any(Vec::<Expr>::new(), loc()).is_err());

    let choice = op::if_then_else(&conds[0], &x, &one, loc()).unwrap();
    assert_eq!(dtype(&choice), "int32");
    assert_eq!(dtype(&op::likely(&conds[0], loc()).unwrap()), "bool");
    assert_eq!(dtype(&op::assume(&conds[0], loc()).unwrap()), "bool");
}

#[test]
fn casts_and_conversions() {
    let _serial = serial();
    let x = var("x", "int32");
    assert_eq!(dtype(&op::cast(&x, ty("float16"), loc()).unwrap()), "float16");
    assert_eq!(dtype(&op::convert(5i64).unwrap()), "int32");
    assert_eq!(dtype(&op::convert(2.5f64).unwrap()), "float32");
    assert!(structural_equal(&op::convert(Expr::from(&x)).unwrap(), &x));
}

#[test]
fn math_operators() {
    let _serial = serial();
    let f = var("f", "float32");
    for unary in [op::ceil, op::floor, op::round, op::nearbyint, op::trunc] {
        assert_eq!(dtype(&unary(Expr::from(&f), loc()).unwrap()), "float32");
    }
    for predicate in [op::isnan, op::isfinite, op::isinf] {
        assert_eq!(dtype(&predicate(Expr::from(&f), loc()).unwrap()), "bool");
    }
    assert_eq!(int_value(op::abs(int("int32", -3), loc()).unwrap()), 3);
    assert_eq!(int_value(op::max_value(ty("int8"), loc()).unwrap()), 127);
    assert_eq!(int_value(op::min_value(ty("int8"), loc()).unwrap()), -128);
    let inf: FloatImm = op::infinity(ty("float32"), loc()).unwrap().try_cast().unwrap();
    assert!(inf.value.is_infinite());
}
