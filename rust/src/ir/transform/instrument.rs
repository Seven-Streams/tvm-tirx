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

//! Pass instrumentation.
//!
//! Laid out like `python/tvm_tirx/ir/transform/instrument.py`. A [`BasePassInstrument`] is a
//! [`PassInstrument`] whose hooks are callbacks: the object that `pass_instrument` (in
//! `ir/transform/core.py`) creates for an instrument written in Python.

// tvm-ffi-stubgen(begin): import-section
use crate::ir::transform::core::PassInstrument;
use crate::ir::transform::core::PassInstrumentObj;
use std::ops::Deref;
use tvm_ffi::ObjectArc;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/transform.BasePassInstrument
/// Opaque: no metadata of its own: total_size is unknown. Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "transform.BasePassInstrument"]
pub struct BasePassInstrumentObj {
    base: PassInstrumentObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct BasePassInstrument {
    base: ObjectArc<BasePassInstrumentObj>,
}

impl Deref for BasePassInstrument {
    type Target = BasePassInstrumentObj;
    fn deref(&self) -> &BasePassInstrumentObj {
        &self.base
    }
}

impl Deref for BasePassInstrumentObj {
    type Target = PassInstrumentObj;
    fn deref(&self) -> &PassInstrumentObj {
        &self.base
    }
}

tvm_ffi::impl_object_upcast!(BasePassInstrument => PassInstrument);
// tvm-ffi-stubgen(end)
