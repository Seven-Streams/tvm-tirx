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
use tvm_tirx::tvm_ffi::Any;

use super::{dtype, int32, int32_var, loc, op, tensor_var};
use crate::common::{load_tirx_compiler, serial, structural_equal};

#[test]
fn expressions() {
    load_tirx_compiler();
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
    load_tirx_compiler();
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
