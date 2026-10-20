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
#![allow(dead_code, unused_imports)]

//! A supply of unique names, for building and rewriting IR.
//!
//! The objects `python/tvm_tirx/ir/unique_name_supply.py` defines. A [`UniqueNameSupply`] is
//! opaque: it is used through the native `ir.UniqueNameSupply*` functions.

// tvm-ffi-stubgen(begin): import-section
use std::ops::Deref;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.UniqueNameSupply
/// Opaque: bytes [24, 64) of [24, 64) are not accounted for by reflected fields. Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.UniqueNameSupply"]
#[type_final]
pub struct UniqueNameSupplyObj {
    base: Object,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct UniqueNameSupply {
    base: ObjectArc<UniqueNameSupplyObj>,
}

impl Deref for UniqueNameSupply {
    type Target = UniqueNameSupplyObj;
    fn deref(&self) -> &UniqueNameSupplyObj {
        &self.base
    }
}
// tvm-ffi-stubgen(end)
