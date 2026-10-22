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

//! Helpers over the IR constructors for code that builds IR the way C++ passes do (see the module
//! doc): the constants C++ spells with a bare literal and the C++ members the constructors of
//! [`crate::ir`] do not cover.

use std::cell::OnceCell;

use tvm_ffi::{
    BigInt, DLDataType, Function as FfiFunction, ObjectCore, ObjectRefCast, ObjectRefCore, Result,
    String, INTERNAL_ERROR,
};

use crate::ir::prim::Shuffle;
use crate::ir::support;
use crate::ir::{
    dtype, Call, Expr, ExprObj, FloatImm, For, IntImm, IntoExpr, PrimType, PrimTypeObj, PtrType,
    Range, SeqStmt, SeqStmtObj, Stmt, Type, Var, UNKNOWN_LOC,
};
use crate::runtime::{MemoryScope, MemoryScopeObj};
use crate::tirx::{TensorType, TensorTypeObj, ThreadScope, ThreadScopeObj};

// ---------------------------------------------------------------------------------------------
// Constants and types

/// `int32`, the type of a bare integer argument such as C++ `Evaluate(0)` ([`dtype::int`]).
pub fn int32_dtype() -> DLDataType {
    dtype::int(32)
}

/// `PrimType(dtype)` ([`PrimType::from_dtype`]), the node C++ shares for each dtype.
pub fn prim_type(dtype: DLDataType) -> Result<PrimType> {
    PrimType::from_dtype(dtype)
}

impl ExprObj {
    /// The type of a PrimExpr (C++ `expr.ty()` of a `PrimExpr`); an error for any other
    /// expression. Every expression node has it: `var.prim_ty()`, `call.args.get(0)?.prim_ty()`.
    pub fn prim_ty(&self) -> Result<PrimType> {
        self.prim_dtype()?;
        self.ty.clone().try_cast()
    }

    /// The dtype of a PrimExpr (C++ `expr.ty()->dtype`); an error for any other expression.
    pub fn prim_dtype(&self) -> Result<DLDataType> {
        match self.ty.as_node::<PrimTypeObj>() {
            Some(ty) => Ok(ty.dtype),
            None => {
                Err(tvm_ffi::Error::new(tvm_ffi::TYPE_ERROR, "expected a PrimType expression", ""))
            }
        }
    }
}

/// `IntImm::Int32(value)`, the C++ meaning of a bare integer argument such as `Evaluate(0)`; the
/// value must fit `int32` ([`IntImm::from_dtype`]).
pub fn int32(value: i64) -> Result<IntImm> {
    int_imm(dtype::int(32), value as i128)
}

/// C++ `IntImm(dtype, value)`: a scalar integer or boolean constant at the unknown location; the
/// value must fit the type ([`IntImm::from_dtype`]).
pub fn int_imm(dtype: DLDataType, value: i128) -> Result<IntImm> {
    big_int_imm(dtype, BigInt::from(value))
}

pub(crate) fn big_int_imm(dtype: DLDataType, value: BigInt) -> Result<IntImm> {
    IntImm::from_dtype(dtype, value, UNKNOWN_LOC)
}

/// C++ `FloatImm(dtype, value)`: a scalar floating-point constant at the unknown location; a
/// finite value must be in the range of the type ([`FloatImm::from_dtype`]).
pub fn float_imm(dtype: DLDataType, value: f64) -> Result<FloatImm> {
    FloatImm::from_dtype(dtype, value, UNKNOWN_LOC)
}

// ---------------------------------------------------------------------------------------------
// Memory scopes, pointers and variables

thread_local! {
    static GLOBAL_SCOPE: OnceCell<MemoryScope> = const { OnceCell::new() };
}

impl MemoryScope {
    /// C++ `MemoryScope::Get(name)`: the registered scope of that name; an error for an
    /// unregistered one. The enum's registry lives in C++; its JSON decoder is this lookup.
    pub fn get(name: &str) -> Result<Self> {
        let decode = tvm_ffi::get_type_attr(MemoryScopeObj::type_index(), "__data_from_json__")
            .ok_or_else(|| error!(INTERNAL_ERROR, "runtime.MemoryScope has no JSON decoder"))?;
        FfiFunction::try_from(decode)?.call_tuple((String::from(name),))?.try_into()
    }

    /// C++ `scope.name()`: the registered name. The enum's JSON encoder is this lookup.
    pub fn name(&self) -> Result<String> {
        let encode = tvm_ffi::get_type_attr(MemoryScopeObj::type_index(), "__data_to_json__")
            .ok_or_else(|| error!(INTERNAL_ERROR, "runtime.MemoryScope has no JSON encoder"))?;
        FfiFunction::try_from(encode)?.call_tuple((self.clone(),))?.try_into()
    }

    /// C++ `MemoryScope::Global()`, the default scope, which the library always registers; fetched
    /// once per thread.
    pub fn global() -> Self {
        support::shared(&GLOBAL_SCOPE, || Self::get("global"))
    }
}

impl PtrType {
    /// C++ `PtrType::VoidPtrType()`: an untyped pointer to global memory.
    pub fn void() -> Result<Self> {
        Self::new(PrimType::void(), &MemoryScope::global())
    }
}

impl TensorTypeObj {
    /// C++ `TensorTypeNode::DataPtrType()`: the type of the tensor's data pointer.
    pub fn data_ptr_type(&self) -> Result<PtrType> {
        PtrType::new(self.dtype.clone(), &self.scope)
    }
}

impl Var {
    /// C++ `PrimVar(name, dtype)`: a variable of the primitive type of `dtype`, at the unknown
    /// location.
    pub fn prim(name: &str, dtype: DLDataType) -> Result<Self> {
        Ok(Self::new(name, PrimType::from_dtype(dtype)?, UNKNOWN_LOC))
    }

    /// C++ `TensorVar::data()`: `tensor_data_ptr(var)`, the data pointer of a tensor variable.
    pub fn tensor_data(&self) -> Result<Expr> {
        let ty: TensorType = self.ty.clone().try_cast()?;
        let ptr: Type = ty.data_ptr_type()?.into();
        let op = support::cached_op("tirx.tensor_data_ptr")?;
        let this: Expr = self.clone().into();
        Ok(Call::new(Some(&ptr), op, [this], None, Vec::<Type>::new(), UNKNOWN_LOC)?.into())
    }
}

// ---------------------------------------------------------------------------------------------
// Expressions and statements

impl Shuffle {
    /// C++ `Shuffle::Concat(vectors)`: the lanes of the vectors in order; one vector is itself.
    pub fn concat(vectors: impl IntoIterator<Item = Expr>) -> Result<Expr> {
        let mut vectors: Vec<Expr> = vectors.into_iter().collect();
        if vectors.len() <= 1 {
            return vectors.pop().ok_or_else(|| {
                support::compare_failed(INTERNAL_ERROR, "vectors.size() != 0", 0, 0, "")
            });
        }
        let mut indices = Vec::new();
        for vector in &vectors {
            for _ in 0..support::lanes(vector.prim_dtype()?)? {
                indices.push(Expr::from(IntImm::int32(indices.len() as i32)));
            }
        }
        Ok(Self::new(vectors, indices, UNKNOWN_LOC)?.into())
    }

    /// C++ `Shuffle::ExtractElement(vector, index)`: lane `index` of `vector`.
    pub fn extract_element(vector: impl IntoExpr, index: i32) -> Result<Self> {
        Self::new([vector.into_expr()?], [Expr::from(IntImm::int32(index))], UNKNOWN_LOC)
    }
}

impl SeqStmt {
    /// The C++ implicit `Stmt -> SeqStmt` conversion (`SeqStmt(stmt)`): a `SeqStmt` passes
    /// through, any other statement is wrapped with its own location.
    pub fn from_stmt(stmt: impl Into<Stmt>) -> Result<Self> {
        let stmt = stmt.into();
        if stmt.as_node::<SeqStmtObj>().is_some() {
            return stmt.try_cast();
        }
        let loc = stmt.loc.clone();
        Ok(Self::new([stmt], loc))
    }
}

impl For {
    /// C++ `ForNode::extent()`: the iteration count `end - begin` of a unit-step loop, as
    /// `Range::GetExtentFromBeginEnd` spells it.
    pub fn extent(&self) -> Result<Expr> {
        Range::get_extent_from_begin_end(self.begin.clone(), self.end.clone())
    }
}

// ---------------------------------------------------------------------------------------------
// Objects whose state Rust cannot see

impl ThreadScope {
    /// C++ `scope.name()`: the registered name `<backend>.<scope>`. The enum's JSON encoder is
    /// this lookup.
    pub fn name_str(&self) -> Result<String> {
        let encode = tvm_ffi::get_type_attr(ThreadScopeObj::type_index(), "__data_to_json__")
            .ok_or_else(|| error!(INTERNAL_ERROR, "tirx.ThreadScope has no JSON encoder"))?;
        FfiFunction::try_from(encode)?.call_tuple((self.clone(),))?.try_into()
    }
}
