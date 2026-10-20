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

//! Tensor functions.
//!
//! The objects `python/tvm_tirx/tirx/function.py` defines. A [`Function`] is a
//! [`BaseFunc`] with parameters, a return type and an optional body.

// `custom-new` reserves `new` for a constructor with the native one's semantics (`ty` derived from
// the signature); the generated `from_complete_fields` allocates the fields as given.
// tvm-ffi-stubgen(custom-new): tirx.Function

// tvm-ffi-stubgen(begin): import-section
use crate::ir::attrs::DictAttrs;
use crate::ir::expr::Expr;
use crate::ir::expr::Var;
use crate::ir::function::BaseFunc;
use crate::ir::function::BaseFuncObj;
use crate::ir::location::Location;
use crate::ir::r#type::Type;
use crate::ir::stmt::SeqStmt;
use std::ops::Deref;
use tvm_ffi::Array;
use tvm_ffi::ObjectArc;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.Function
/// Complete: reflected fields fill [48, 72) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.Function"]
#[type_final]
pub struct FunctionObj {
    base: BaseFuncObj,
    pub params: Array<Var>,
    pub ret_type: Type,
    pub body: Option<SeqStmt>,
}

const _: () = {
    assert!(::core::mem::size_of::<FunctionObj>() == 72);
    assert!(::core::mem::align_of::<FunctionObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Function {
    base: ObjectArc<FunctionObj>,
}

impl Deref for Function {
    type Target = FunctionObj;
    fn deref(&self) -> &FunctionObj {
        &self.base
    }
}

impl Deref for FunctionObj {
    type Target = BaseFuncObj;
    fn deref(&self) -> &BaseFuncObj {
        &self.base
    }
}

impl FunctionObj {
    pub(crate) fn new(
        loc: Location,
        ty: Type,
        attrs: DictAttrs,
        params: Array<Var>,
        ret_type: Type,
        body: Option<SeqStmt>,
    ) -> Self {
        let base = BaseFuncObj::new(loc, ty, attrs);
        Self { base, params, ret_type, body }
    }
}

impl Function {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        loc: Location,
        ty: Type,
        attrs: DictAttrs,
        params: Array<Var>,
        ret_type: Type,
        body: Option<SeqStmt>,
    ) -> Self {
        let obj = FunctionObj::new(loc, ty, attrs, params, ret_type, body);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Function => Expr, Function => BaseFunc);
// tvm-ffi-stubgen(end)
