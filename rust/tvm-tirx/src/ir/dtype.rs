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

//! The DLPack data types by their fields, for a caller that knows the type it wants:
//! `dtype::int(32)` where C++ writes `DataType::Int(32)`, instead of parsing `"int32"` with
//! `DLDataType::try_from_str`. These are plain values with no native call, so they can be
//! constants and can be compared; [`PrimType::int`](crate::ir::PrimType::int) and its siblings
//! give the IR type of each.

use tvm_ffi::DLDataType;
use tvm_ffi::DLDataTypeCode;

/// The scalar type of `code` and `bits`.
const fn scalar(code: DLDataTypeCode, bits: u8) -> DLDataType {
    DLDataType { code: code as u8, bits, lanes: 1 }
}

/// The signed integer of `bits` bits (`int32` for 32).
pub const fn int(bits: u8) -> DLDataType {
    scalar(DLDataTypeCode::kDLInt, bits)
}

/// The unsigned integer of `bits` bits (`uint8` for 8).
pub const fn uint(bits: u8) -> DLDataType {
    scalar(DLDataTypeCode::kDLUInt, bits)
}

/// The IEEE floating-point type of `bits` bits (`float32` for 32).
pub const fn float(bits: u8) -> DLDataType {
    scalar(DLDataTypeCode::kDLFloat, bits)
}

/// The brain floating-point type of `bits` bits (`bfloat16` for 16).
pub const fn bfloat(bits: u8) -> DLDataType {
    scalar(DLDataTypeCode::kDLBfloat, bits)
}

/// The boolean type, stored in 8 bits (`bool`).
pub const fn bool() -> DLDataType {
    scalar(DLDataTypeCode::kDLBool, 8)
}

/// The void type, which C++ encodes as `handle(0, 0)` (`DataType::Void()`).
pub const fn void() -> DLDataType {
    DLDataType { code: DLDataTypeCode::kDLOpaqueHandle as u8, bits: 0, lanes: 0 }
}

/// The vector of `lanes` elements of the scalar `dtype` (C++ `DataType::with_lanes`):
/// `with_lanes(float(16), 4)` is `float16x4`. `lanes` is stored as the DLPack lane field: a
/// fixed-length vector has fewer than 32768 lanes, and the values from 32768 up encode the
/// scalable vectors of [`PrimType::with_scalable_lanes`], which [`PrimType::with_lanes`]
/// rejects.
///
/// [`PrimType::with_scalable_lanes`]: crate::ir::PrimType::with_scalable_lanes
/// [`PrimType::with_lanes`]: crate::ir::PrimType::with_lanes
pub const fn with_lanes(dtype: DLDataType, lanes: u16) -> DLDataType {
    DLDataType { code: dtype.code, bits: dtype.bits, lanes }
}
