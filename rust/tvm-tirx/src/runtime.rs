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

//! The runtime objects the IR refers to.
//!
//! A [`MemoryScope`], such as the `scope` of a [`PtrType`](crate::ir::PtrType), is an enum object
//! that `python/tvm_tirx/runtime/memory_scope.py` defines. The runtime functions the Python package
//! wraps are here too: tensor allocation and views, device attributes and runtime queries.

// No `prefix`: only the objects with a block below are bound, not all of `runtime`.
// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::Any
// tvm-ffi-stubgen(import-object): tvm_ffi::DLDataType
// tvm-ffi-stubgen(import-object): tvm_ffi::DLDevice
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): tvm_ffi::Shape
// tvm-ffi-stubgen(import-object): tvm_ffi::String
// tvm-ffi-stubgen(import-object): tvm_ffi::Tensor

// tvm-ffi-stubgen(begin): import-section
use std::ops::Deref;
use tvm_ffi::Any;
use tvm_ffi::DLDataType;
use tvm_ffi::DLDevice;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::Result;
use tvm_ffi::Shape;
use tvm_ffi::String;
use tvm_ffi::Tensor;

/// Header-only stand-in for the builtin `ffi.Enum`; it only carries the ancestor depth.
#[allow(dead_code)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ffi.Enum"]
struct FfiEnumObj {
    base: Object,
}
// tvm-ffi-stubgen(end)

// `ir.PtrType.storage_scope` refers to this type.
// tvm-ffi-stubgen(begin): object/runtime.MemoryScope
/// Opaque: parent 'ffi.Enum' is opaque (no-mirror). Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "runtime.MemoryScope"]
#[type_final]
pub struct MemoryScopeObj {
    base: FfiEnumObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct MemoryScope {
    base: ObjectArc<MemoryScopeObj>,
}

impl Deref for MemoryScope {
    type Target = MemoryScopeObj;
    fn deref(&self) -> &MemoryScopeObj {
        &self.base
    }
}
// tvm-ffi-stubgen(end)

impl MemoryScope {
    /// Register the memory scope `name`; an error if it is already registered
    /// (`runtime.MemoryScopeRegister`).
    pub fn register(name: &str) -> Result<Self> {
        tvm_ffi::cached_global_func!("runtime.MemoryScopeRegister")
            .call_tuple((String::from(name),))?
            .try_into()
    }
}

/// An uninitialized tensor on `device`, in the memory scope `mem_scope` or the global one
/// (`runtime.TensorAllocWithScope`).
pub fn empty(
    shape: &[i64],
    dtype: DLDataType,
    device: DLDevice,
    mem_scope: Option<&MemoryScope>,
) -> Result<Tensor> {
    tvm_ffi::cached_global_func!("runtime.TensorAllocWithScope")
        .call_tuple((Shape::from(shape), dtype, device, mem_scope.cloned()))?
        .try_into()
}

/// A view of the storage of `source`, `relative_byte_offset` bytes past its data, with another
/// shape and data type (`runtime.TensorCreateView`).
pub fn create_view(
    source: &Tensor,
    shape: &[i64],
    dtype: DLDataType,
    relative_byte_offset: u64,
) -> Result<Tensor> {
    tvm_ffi::cached_global_func!("runtime.TensorCreateView")
        .call_tuple((source, Shape::from(shape), dtype, relative_byte_offset))?
        .try_into()
}

/// The attribute `name` of `device`, such as `exist` or `max_threads_per_block`; `None` when the
/// device does not report it (`runtime.GetDeviceAttr`).
pub fn get_device_attr(device: DLDevice, name: &str) -> Result<Any> {
    tvm_ffi::cached_global_func!("runtime.GetDeviceAttr").call_tuple((
        device.device_type as i32,
        device.device_id,
        String::from(name),
    ))
}

/// Whether the runtime of the device or target kind `target` (`"cpu"`, `"cuda"`, `"llvm"`...) is
/// enabled (`runtime.RuntimeEnabled`).
pub fn enabled(target: &str) -> Result<bool> {
    tvm_ffi::cached_global_func!("runtime.RuntimeEnabled")
        .call_tuple((String::from(target),))?
        .try_into()
}

/// The number of threads the runtime thread pool uses (`runtime.NumThreads`).
pub fn num_threads() -> Result<i32> {
    tvm_ffi::cached_global_func!("runtime.NumThreads").call_tuple(())?.try_into()
}

// The CUDA queries are in the CUDA runtime library, which may not be built or loaded: they are
// looked up when called, and their absence is an error.

/// The number of CUDA devices (`runtime.GetCudaDeviceCount`).
pub fn get_cuda_device_count() -> Result<i32> {
    tvm_ffi::Function::get_global("runtime.GetCudaDeviceCount")?.call_tuple(())?.try_into()
}

/// The free and total memory of the current CUDA device, as a message
/// (`runtime.GetCudaFreeMemory`).
pub fn get_cuda_free_memory() -> Result<String> {
    tvm_ffi::Function::get_global("runtime.GetCudaFreeMemory")?.call_tuple(())?.try_into()
}
