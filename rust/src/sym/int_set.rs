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

//! Sets of integers.
//!
//! The objects `python/tvm_tirx/sym/int_set.py` defines. [`IntSet`] is the base, and an
//! [`IntervalSet`] is the interval between its `min_value` and `max_value`.

// tvm-ffi-stubgen(begin): import-section
use crate::ir::expr::Expr;
use std::ops::Deref;
use tvm_ffi::FieldGetter;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::ObjectCore;
use tvm_ffi::Result;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/sym.IntervalSet
/// Opaque: parent 'sym.IntSet' is opaque (layout-unknown). Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.IntervalSet"]
#[type_final]
pub struct IntervalSetObj {
    base: IntSetObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IntervalSet {
    base: ObjectArc<IntervalSetObj>,
}

impl Deref for IntervalSet {
    type Target = IntervalSetObj;
    fn deref(&self) -> &IntervalSetObj {
        &self.base
    }
}

impl Deref for IntervalSetObj {
    type Target = IntSetObj;
    fn deref(&self) -> &IntSetObj {
        &self.base
    }
}

impl IntervalSetObj {
    pub fn min_value(&self) -> Result<Expr> {
        FieldGetter::new(Self::type_index(), "min_value")?.get(self)
    }

    pub fn max_value(&self) -> Result<Expr> {
        FieldGetter::new(Self::type_index(), "max_value")?.get(self)
    }
}

tvm_ffi::impl_object_upcast!(IntervalSet => IntSet);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/sym.IntSet
/// Opaque: no metadata of its own: total_size is unknown. Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.IntSet"]
pub struct IntSetObj {
    base: Object,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IntSet {
    base: ObjectArc<IntSetObj>,
}

impl Deref for IntSet {
    type Target = IntSetObj;
    fn deref(&self) -> &IntSetObj {
        &self.base
    }
}
// tvm-ffi-stubgen(end)
