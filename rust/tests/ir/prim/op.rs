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
use tvm_tirx::ir::{UNKNOWN_LOC, Expr, FloatImm, IntImm, PrimType};
use tvm_tirx::tvm_ffi::{ObjectRefCast, ObjectRefCore};

use super::{dtype, int_value, var};
use crate::common::{serial, structural_equal};

#[test]
fn arithmetic_operators() {
    let _serial = serial();
    let x = var("x", PrimType::int(32));

    // Promotion: int32 + float32 is float32, and constants fold.
    assert_eq!(dtype(&op::add(&x, 1.0f32, UNKNOWN_LOC).unwrap()), "float32");
    assert_eq!(dtype(&op::add(&x, IntImm::int64(1), UNKNOWN_LOC).unwrap()), "int64");
    macro_rules! folded {
        ($op:path, $a:expr, $b:expr) => {
            int_value($op($a, $b, UNKNOWN_LOC).unwrap())
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
    let not = op::bitwise_not(0, UNKNOWN_LOC).unwrap();
    assert!(not.try_cast::<prim::BitwiseNot>().is_ok(), "the node, not folded");

    // On a variable they build the nodes.
    let floordiv = op::floordiv(&x, 2, UNKNOWN_LOC).unwrap();
    assert!(floordiv.try_cast::<prim::FloorDiv>().is_ok());
    assert_eq!(dtype(&op::pow(2.0f32, 3.0f32, UNKNOWN_LOC).unwrap()), "float32");
}

#[test]
fn comparison_and_logic_operators() {
    let _serial = serial();
    let x = var("x", PrimType::int(32));
    let one = Expr::from(1);
    for cmp in [op::eq, op::ne, op::lt, op::le, op::gt, op::ge] {
        assert_eq!(dtype(&cmp(Expr::from(&x), one.clone(), UNKNOWN_LOC).unwrap()), "bool");
    }
    assert!(op::lt(&x, &one, UNKNOWN_LOC).unwrap().try_cast::<prim::LT>().is_ok());

    let conds = [op::lt(&x, &one, UNKNOWN_LOC).unwrap(), op::gt(&x, -5, UNKNOWN_LOC).unwrap()];
    assert!(op::all(conds.clone(), UNKNOWN_LOC).unwrap().try_cast::<prim::And>().is_ok());
    assert!(op::any(conds.clone(), UNKNOWN_LOC).unwrap().try_cast::<prim::Or>().is_ok());
    assert!(structural_equal(&op::all([&conds[0]], UNKNOWN_LOC).unwrap(), &conds[0]));
    assert!(op::any(Vec::<Expr>::new(), UNKNOWN_LOC).is_err());

    let choice = op::if_then_else(&conds[0], &x, &one, UNKNOWN_LOC).unwrap();
    assert_eq!(dtype(&choice), "int32");
    assert_eq!(dtype(&op::likely(&conds[0], UNKNOWN_LOC).unwrap()), "bool");
    assert_eq!(dtype(&op::assume(&conds[0], UNKNOWN_LOC).unwrap()), "bool");
}

#[test]
fn casts_and_conversions() {
    let _serial = serial();
    let x = var("x", PrimType::int(32));
    assert_eq!(dtype(&op::cast(&x, PrimType::float(16), UNKNOWN_LOC).unwrap()), "float16");
    assert_eq!(dtype(&op::convert(5i64).unwrap()), "int32");
    assert_eq!(dtype(&op::convert(2.5f64).unwrap()), "float32");
    assert!(structural_equal(&op::convert(Expr::from(&x)).unwrap(), &x));
}

#[test]
fn math_operators() {
    let _serial = serial();
    let f = var("f", PrimType::float(32));
    for unary in [op::ceil, op::floor, op::round, op::nearbyint, op::trunc] {
        assert_eq!(dtype(&unary(Expr::from(&f), UNKNOWN_LOC).unwrap()), "float32");
    }
    for predicate in [op::isnan, op::isfinite, op::isinf] {
        assert_eq!(dtype(&predicate(Expr::from(&f), UNKNOWN_LOC).unwrap()), "bool");
    }
    assert_eq!(int_value(op::abs(-3, UNKNOWN_LOC).unwrap()), 3);
    assert_eq!(int_value(op::max_value(PrimType::int(8), UNKNOWN_LOC).unwrap()), 127);
    assert_eq!(int_value(op::min_value(PrimType::int(8), UNKNOWN_LOC).unwrap()), -128);
    let inf: FloatImm = op::infinity(PrimType::float(32), UNKNOWN_LOC).unwrap().try_cast().unwrap();
    assert!(inf.value.is_infinite());
}

/// `make_const` builds the constant of the kind of its type, broadcast over the lanes of a vector.
#[test]
fn make_const() {
    let _serial = serial();
    let seven = op::make_const(PrimType::int(32), 7, UNKNOWN_LOC).unwrap();
    assert_eq!(dtype(&seven), "int32");
    assert_eq!(int_value(seven), 7);
    assert_eq!(int_value(op::make_const(PrimType::bool(), 1, UNKNOWN_LOC).unwrap()), 1);
    assert_eq!(int_value(op::make_const(PrimType::uint(8), 255, UNKNOWN_LOC).unwrap()), 255);
    assert!(op::make_const(PrimType::uint(8), -1, UNKNOWN_LOC).is_err());
    assert!(op::make_const(PrimType::int(8), 128, UNKNOWN_LOC).is_err());
    for ty in [PrimType::float(16), PrimType::bfloat(16), PrimType::float(64)] {
        let two: FloatImm = op::make_const(ty.clone(), 2, UNKNOWN_LOC).unwrap().try_cast().unwrap();
        assert_eq!(two.value, 2.0);
        assert!(structural_equal(&two.ty, &ty));
    }
    assert!(op::make_const(PrimType::void(), 0, UNKNOWN_LOC).is_err());

    let vector = op::make_const(PrimType::int(32).with_lanes(4).unwrap(), 1, UNKNOWN_LOC).unwrap();
    assert_eq!(dtype(&vector), "int32x4");
    let broadcast: prim::Broadcast = vector.try_cast().unwrap();
    assert_eq!(int_value(broadcast.value.clone()), 1);
    assert_eq!(int_value(broadcast.lanes.clone()), 4);

    let scalable = PrimType::int(32).with_scalable_lanes(4).unwrap();
    let vector = op::make_const(scalable.clone(), 1, UNKNOWN_LOC).unwrap();
    assert!(structural_equal(&vector.ty, &scalable));
    let broadcast: prim::Broadcast = vector.try_cast().unwrap();
    let lanes = broadcast.lanes.as_node::<prim::MulObj>().expect("vscale * 4");
    assert_eq!(int_value(lanes.b.clone()), 4);
    let _: IntImm = IntImm::int32(0);
}
