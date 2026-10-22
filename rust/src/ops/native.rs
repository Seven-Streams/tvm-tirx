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

//! Typed wrappers of the native functions the operators and the passes call rather than write
//! again: printing, structural comparison, and the queries that have no wrapper of their own.
//! Each calls the global C++ (or, where noted, Python) registers for it.

use tvm_ffi::object::ObjectRef;
use tvm_ffi::{Any, AnyView, FieldGetter, ObjectCore, ObjectRefCore, Result, String};

use crate::ir::support;
use crate::ir::Expr;
use crate::runtime::{MemoryScope, MemoryScopeObj};
use crate::target::Target;

// ---------------------------------------------------------------------------------------------
// Printing and structural comparison

/// How C++ prints a node in an error message (`ffi.ReprPrint`), such as `T.float32`.
pub fn repr<T>(value: &T) -> std::string::String
where
    for<'a> AnyView<'a>: From<&'a T>,
{
    support::repr(value)
}

/// C++ `ffi::StructuralEqual()(lhs, rhs)`, which does not map free variables.
pub fn structural_equal<L, R>(lhs: &L, rhs: &R) -> Result<bool>
where
    for<'a> AnyView<'a>: From<&'a L> + From<&'a R>,
{
    support::structural_equal(lhs, rhs)
}

/// C++ `ffi::StructuralHash()(value)`.
pub fn structural_hash<T>(value: &T) -> Result<i64>
where
    for<'a> AnyView<'a>: From<&'a T>,
{
    let flag = <AnyView as From<&bool>>::from;
    let (map_free_vars, skip_tensor_content) = (false, false);
    tvm_ffi::cached_global_func!("ffi.StructuralHash")
        .call_packed(&[AnyView::from(value), flag(&map_free_vars), flag(&skip_tensor_content)])?
        .try_into()
}

/// C++ `prim::ExprDeepEqual()(lhs, rhs)`: equal expressions, variables compared by identity
/// ([`crate::ir::prim::expr_deep_equal`]).
pub fn deep_equal(lhs: &Expr, rhs: &Expr) -> Result<bool> {
    crate::ir::prim::expr_deep_equal(lhs.clone(), rhs.clone())
}

// ---------------------------------------------------------------------------------------------
// Targets

impl Target {
    /// `codegen.target_has_feature(feature, target)`, an LLVM target query; false in a build
    /// without LLVM.
    pub fn has_llvm_feature(&self, feature: &str) -> Result<bool> {
        let Ok(has_feature) = tvm_ffi::Function::get_global("codegen.target_has_feature") else {
            return Ok(false);
        };
        has_feature.call_tuple((String::from(feature), self.clone()))?.try_into()
    }

    /// `tvm.support.nvcc.get_compute_version(target)`, which Python registers: the CUDA compute
    /// version such as `"9.0"`, or `None` when Python has not registered it.
    pub fn cuda_compute_version(&self) -> Result<Option<String>> {
        let Ok(version) = tvm_ffi::Function::get_global("tvm.support.nvcc.get_compute_version")
        else {
            return Ok(None);
        };
        Ok(Some(version.call_tuple((self.clone(),))?.try_into()?))
    }
}

// ---------------------------------------------------------------------------------------------
// Memory scopes

impl MemoryScope {
    /// The value of the enum attribute column `column` for this scope (C++
    /// `MemoryScope::GetAttr`), `None` when the column has none for it.
    pub fn attr(&self, column: &str) -> Result<Option<Any>> {
        let Some(state) = tvm_ffi::get_type_attr(MemoryScopeObj::type_index(), "__ffi_enum__")
        else {
            return Ok(None);
        };
        let getter = FieldGetter::new(state.type_index(), "attrs")?;
        let state: ObjectRef = state.try_into()?;
        let columns = getter.get_any(&**ObjectRefCore::data(&state))?;
        let (count, get) = (
            tvm_ffi::cached_global_func!("ffi.DictCount"),
            tvm_ffi::cached_global_func!("ffi.DictGetItem"),
        );
        let column = String::from(column);
        if i64::try_from(count.call_tuple((columns.clone(), column.clone()))?)? == 0 {
            return Ok(None);
        }
        let values = get.call_tuple((columns, column))?;
        if i64::try_from(count.call_tuple((values.clone(), self.clone()))?)? == 0 {
            return Ok(None);
        }
        Ok(Some(get.call_tuple((values, self.clone()))?))
    }
}
