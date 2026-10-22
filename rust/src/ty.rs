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

//! IR types defined in Rust: new `ir.Type`s, types of expressions as `PrimType` and `TupleType`
//! are, whose fields the C++ structural equality and hashing, deep copy, `repr`, the TVMScript
//! printer and Python see, as they see a C++ type's.
//!
//! [`tirx_type`](crate::tirx_type) defines one. It builds on tvm-ffi's `#[derive(Object)]`, which
//! registers the reflection of the fields (C++ `refl::ObjectDef<T>`) and copies the parent
//! through [`ShallowCopy`], which this module implements for `TypeObj`, and on [`type_base`],
//! since `TypeObj::new` is crate-private generated code.
//!
//! ```no_run
//! use tvm_tirx::ir::{PrimType, Type, Var, UNKNOWN_LOC};
//! use tvm_tirx::tirx_type;
//! use tvm_tirx::tvm_ffi::ObjectRefCore;
//!
//! /// The type of a pair of values.
//! #[tirx_type("my_ext.PairType")]
//! pub struct PairType {
//!     pub first: Type,
//!     pub second: Type,
//!     /// Not part of structural equality or hashing.
//!     #[seq_hash_ignore]
//!     pub note: tvm_tirx::tvm_ffi::String,
//! }
//!
//! impl PairType {
//!     pub fn new(first: impl Into<Type>, second: impl Into<Type>) -> Self {
//!         Self::from_complete_fields(UNKNOWN_LOC, first, second, "")
//!     }
//! }
//!
//! fn main() {
//!     let pair = PairType::new(PrimType::int(32), PrimType::float(32));
//!     let p = Var::new("p", pair, UNKNOWN_LOC);
//!     assert!(p.ty.as_node::<PairTypeObj>().is_some());
//! }
//! ```

use tvm_ffi::reflection::ShallowCopy;

use crate::ir::{Location, TypeObj};

impl ShallowCopy for TypeObj {
    fn shallow_copy(&self) -> Self {
        TypeObj::new(self.loc.clone())
    }
}

/// The `ir.Type` base of a Rust-defined type, at `loc` (the C++ `TypeNode` constructor).
pub fn type_base(loc: Location) -> TypeObj {
    TypeObj::new(loc)
}
