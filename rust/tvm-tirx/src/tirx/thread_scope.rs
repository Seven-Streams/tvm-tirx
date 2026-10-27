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

//! Thread scopes: the levels of a backend's thread hierarchy.
//!
//! The objects `python/tvm_tirx/tirx/thread_scope.py` defines. A [`ThreadScope`] is an enum
//! object, such as `cuda.warp`, looked up by name with [`ThreadScope::get`].

// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): tvm_ffi::String

// tvm-ffi-stubgen(begin): import-section
use std::ops::Deref;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::Result;
use tvm_ffi::String;

/// Header-only stand-in for the builtin `ffi.Enum`; it only carries the ancestor depth.
#[allow(dead_code)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ffi.Enum"]
struct FfiEnumObj {
    base: Object,
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.ThreadScope
/// Opaque: parent 'ffi.Enum' is opaque (no-mirror). Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.ThreadScope"]
#[type_final]
pub struct ThreadScopeObj {
    base: FfiEnumObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct ThreadScope {
    base: ObjectArc<ThreadScopeObj>,
}

impl Deref for ThreadScope {
    type Target = ThreadScopeObj;
    fn deref(&self) -> &ThreadScopeObj {
        &self.base
    }
}
// tvm-ffi-stubgen(end)

impl ThreadScope {
    /// The thread scope named `name`, such as `cuda.warp` (`tirx.ThreadScopeGet`).
    pub fn get(name: &str) -> Result<Self> {
        tvm_ffi::cached_global_func!("tirx.ThreadScopeGet")
            .call_tuple((String::from(name),))?
            .try_into()
    }

    /// The backend that registered the scope, such as `cuda` (`tirx.ThreadScopeGetBackend`).
    pub fn backend(&self) -> Result<String> {
        tvm_ffi::cached_global_func!("tirx.ThreadScopeGetBackend").call_tuple((self,))?.try_into()
    }

    /// The name of the scope within its backend, such as `warp` (`tirx.ThreadScopeGetLocalName`).
    pub fn local_name(&self) -> Result<String> {
        tvm_ffi::cached_global_func!("tirx.ThreadScopeGetLocalName").call_tuple((self,))?.try_into()
    }

    /// The level of the scope in its backend's hierarchy; outer scopes rank lower
    /// (`tirx.ThreadScopeGetRank`).
    pub fn rank(&self) -> Result<i64> {
        tvm_ffi::cached_global_func!("tirx.ThreadScopeGetRank").call_tuple((self,))?.try_into()
    }

    /// Whether this scope is above `other` in the hierarchy of their backend (`tirx.ScopeHigher`).
    pub fn is_higher_than(&self, other: impl Into<ThreadScope>) -> Result<bool> {
        tvm_ffi::cached_global_func!("tirx.ScopeHigher")
            .call_tuple((self, &other.into()))?
            .try_into()
    }
}
