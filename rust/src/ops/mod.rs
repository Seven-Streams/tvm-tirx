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

//! Helpers for code that builds IR the way C++ passes do, over the constructors of [`crate::ir`]
//! and the operators of [`crate::ir::prim::op`].
//!
//! Helpers (`build`). What C++ code spells without an argument: the constants of a bare literal
//! ([`int32`], [`int_imm`], [`float_imm`]), the `PrimExpr` queries of an expression
//! (`expr.prim_dtype()`), and the C++ members the constructors do not cover
//! (`MemoryScope::Get`, `PtrType::VoidPtrType`, `TensorVar::data`, `Shuffle::Concat`, the
//! `Stmt -> SeqStmt` conversion).
//!
//! Patterns (`pattern`). The expression patterns of `src/sym/pattern_match.h`.
//!
//! Native functions (`native`). Printing, structural comparison and the target queries that Rust
//! calls rather than writes again, as typed functions and methods.

mod build;
mod native;
pub mod pattern;

pub use build::*;
pub use native::{deep_equal, repr, structural_equal, structural_hash};
