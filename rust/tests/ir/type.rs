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

//! Tests of `ir::type`.

use tvm_tirx::ir::{
    FuncType, MissingType, NoneType, PrimType, PtrType, TensorRegionType, TupleType, Type,
};
use tvm_tirx::tvm_ffi::{Array, ObjectRefCore};

use super::{dtype, loc, scope};
use crate::common::{agree, native, serial, structural_equal};

#[test]
fn types() {
    let _serial = serial();
    MissingType::new().unwrap();
    NoneType::instance().unwrap();
    TensorRegionType::instance().unwrap();
    let i32_ty = PrimType::from_dtype(dtype("int32")).unwrap();
    assert_eq!(i32_ty.dtype, dtype("int32"));
    let ptr = PtrType::new(&i32_ty, &scope()).unwrap();
    assert!(structural_equal(&ptr.element_type, &i32_ty));
    let tuple = TupleType::new([&i32_ty, &i32_ty], loc()).unwrap();
    assert_eq!(tuple.fields.len(), 2);
    let func = FuncType::new([&i32_ty], &i32_ty).unwrap();
    assert_eq!(func.arg_types.len(), 1);
}

#[test]
fn types_agree_with_cpp() {
    let _serial = serial();
    let int32: Type = PrimType::from_dtype(dtype("int32")).unwrap().into();
    let float32: Type = PrimType::from_dtype(dtype("float32")).unwrap().into();
    let missing: Type = MissingType::new().unwrap().into();
    for (what, element) in [("PtrType", &int32), ("PtrType(missing)", &missing)] {
        let native_ptr = native("ir.PtrType", &[element.into(), (&scope()).into()]);
        agree(what, PtrType::new(element, &scope()), native_ptr);
    }
    for (what, fields) in
        [("TupleType", vec![int32.clone(), float32.clone()]), ("TupleType(empty)", vec![])]
    {
        let array: Array<Type> = fields.iter().cloned().collect();
        let native_tuple = native("ir.TupleType", &[(&array).into(), (&loc()).into()]);
        agree(what, TupleType::new(fields, loc()), native_tuple);
    }
    let args: Array<Type> = [int32.clone()].into_iter().collect();
    let native_func = native("ir.FuncType", &[(&args).into(), (&float32).into()]);
    agree("FuncType", FuncType::new([&int32], &float32), native_func);

    let none: Type = native("ir.NoneType", &[]).unwrap().try_into().unwrap();
    assert!(NoneType::instance().unwrap().same_as(&none));
    let region: Type = native("ir.TensorRegionType", &[]).unwrap().try_into().unwrap();
    assert!(TensorRegionType::instance().unwrap().same_as(&region));
}
