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

use tvm_tirx::ir::{FuncType, MissingType, NoneType, PrimType, PtrType, TensorRegionType, TupleType};

use super::{dtype, loc, scope};
use crate::common::{serial, structural_equal};

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
