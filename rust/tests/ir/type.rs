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
    dtype, AnyTypeObj, FuncType, FuncTypeObj, MissingType, MissingTypeObj, NoneType, NoneTypeObj,
    OpaqueTypeObj, PrimType, PrimTypeObj, PtrType, PtrTypeObj, StringTypeObj, TensorRegionType,
    TensorRegionTypeObj, TupleType, TupleTypeObj, Type, TypeObj, UNKNOWN_LOC, Location,
};
use tvm_tirx::tvm_ffi::{Array, DLDataType, DLDataTypeExt, Object, ObjectRefCore};

use super::{scope};
use crate::common::{agree, assert_type_contracts, native, serial, structural_equal};

#[test]
fn generated_types_match_the_registry() {
    let _serial = serial();
    assert_type_contracts!(true;
        TypeObj: Object, AnyTypeObj: TypeObj, FuncTypeObj: TypeObj, MissingTypeObj: TypeObj,
        OpaqueTypeObj: TypeObj, PrimTypeObj: TypeObj, PtrTypeObj: TypeObj, StringTypeObj: TypeObj,
        TensorRegionTypeObj: TypeObj, TupleTypeObj: TypeObj, NoneTypeObj: TypeObj,
    );
}

#[test]
fn types() {
    let _serial = serial();
    MissingType::new();
    NoneType::new();
    TensorRegionType::new();
    let i32_ty = PrimType::int(32);
    assert_eq!(i32_ty.dtype, dtype::int(32));
    let ptr = PtrType::new(&i32_ty, &scope()).unwrap();
    assert!(structural_equal(&ptr.element_type, &i32_ty));
    let tuple = TupleType::new([&i32_ty, &i32_ty], UNKNOWN_LOC);
    assert_eq!(tuple.fields.len(), 2);
    let func = FuncType::new([&i32_ty], &i32_ty);
    assert_eq!(func.arg_types.len(), 1);
}

/// The builtin types are the dtypes their names parse to, and the nodes C++ shares.
#[test]
fn builtin_types() {
    let _serial = serial();
    let cases: [(&str, PrimType, DLDataType); 9] = [
        ("int32", PrimType::int(32), dtype::int(32)),
        ("int64", PrimType::int(64), dtype::int(64)),
        ("uint8", PrimType::uint(8), dtype::uint(8)),
        ("float32", PrimType::float(32), dtype::float(32)),
        ("float16", PrimType::float(16), dtype::float(16)),
        ("bfloat16", PrimType::bfloat(16), dtype::bfloat(16)),
        ("bool", PrimType::bool(), dtype::bool()),
        (
            "float16x4",
            PrimType::float(16).with_lanes(4).unwrap(),
            dtype::with_lanes(dtype::float(16), 4),
        ),
        ("int8x32", PrimType::int(8).with_lanes(32).unwrap(), dtype::with_lanes(dtype::int(8), 32)),
    ];
    for (name, ty, dtype) in cases {
        assert_eq!(dtype, DLDataType::try_from_str(name).unwrap(), "{name}");
        assert_eq!(ty.dtype, dtype, "{name}");
        let shared: Type = native("ir.PrimType", &[(&dtype).into()]).unwrap().try_into().unwrap();
        assert!(ty.same_as(&shared), "{name}");
        assert!(ty.same_as(&PrimType::from_dtype(dtype).unwrap()), "{name}");
    }
    let void = PrimType::void();
    assert_eq!(void.dtype, dtype::void());
    assert_eq!((void.dtype.code, void.dtype.bits, void.dtype.lanes), (3, 0, 0));
    assert!(void.with_lanes(4).is_err());
    assert!(PrimType::int(32).with_lanes(40000).is_err());
}

#[test]
fn types_agree_with_cpp() {
    let _serial = serial();
    let int32: Type = PrimType::int(32).into();
    let float32: Type = PrimType::float(32).into();
    let missing: Type = MissingType::new().into();
    for (what, element) in [("PtrType", &int32), ("PtrType(missing)", &missing)] {
        let native_ptr = native("ir.PtrType", &[element.into(), (&scope()).into()]);
        agree(what, PtrType::new(element, &scope()), native_ptr);
    }
    for (what, fields) in
        [("TupleType", vec![int32.clone(), float32.clone()]), ("TupleType(empty)", vec![])]
    {
        let array: Array<Type> = fields.iter().cloned().collect();
        let native_tuple =
            native("ir.TupleType", &[(&array).into(), (&Location::unknown()).into()]);
        agree(what, Ok(TupleType::new(fields, UNKNOWN_LOC)), native_tuple);
    }
    let args: Array<Type> = [int32.clone()].into_iter().collect();
    let native_func = native("ir.FuncType", &[(&args).into(), (&float32).into()]);
    agree("FuncType", Ok(FuncType::new([&int32], &float32)), native_func);

    let none: Type = native("ir.NoneType", &[]).unwrap().try_into().unwrap();
    assert!(NoneType::new().same_as(&none));
    let region: Type = native("ir.TensorRegionType", &[]).unwrap().try_into().unwrap();
    assert!(TensorRegionType::new().same_as(&region));
}
