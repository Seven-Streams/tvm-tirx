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

//! Attributes of TIRx operator calls.
//!
//! The objects `python/tvm_tirx/tirx/op/__init__.py` defines. A [`CallFFIKernelAttr`] holds the
//! launch parameters, the launch fields, the number of kernel arguments and the kernel attributes
//! of a `call_ffi_kernel`.

// tvm-ffi-stubgen(begin): import-section
use crate::ir::attrs::Attrs;
use crate::ir::attrs::AttrsObj;
use std::ops::Deref;
use tvm_ffi::Array;
use tvm_ffi::Map;
use tvm_ffi::ObjectArc;
use tvm_ffi::String;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.CallFFIKernelAttr
/// Complete: reflected fields fill [24, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.CallFFIKernelAttr"]
#[type_final]
pub struct CallFFIKernelAttrObj {
    base: AttrsObj,
    pub launch_params: Array<String>,
    pub launch_fields: Array<String>,
    pub num_kernel_args: i64,
    pub kernel_attrs: Map<String, i64>,
}

const _: () = {
    assert!(::core::mem::size_of::<CallFFIKernelAttrObj>() == 56);
    assert!(::core::mem::align_of::<CallFFIKernelAttrObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct CallFFIKernelAttr {
    base: ObjectArc<CallFFIKernelAttrObj>,
}

impl Deref for CallFFIKernelAttr {
    type Target = CallFFIKernelAttrObj;
    fn deref(&self) -> &CallFFIKernelAttrObj {
        &self.base
    }
}

impl Deref for CallFFIKernelAttrObj {
    type Target = AttrsObj;
    fn deref(&self) -> &AttrsObj {
        &self.base
    }
}

impl CallFFIKernelAttrObj {
    pub(crate) fn new(
        launch_params: Array<String>,
        launch_fields: Array<String>,
        num_kernel_args: i64,
        kernel_attrs: Map<String, i64>,
    ) -> Self {
        let base = AttrsObj::new();
        Self { base, launch_params, launch_fields, num_kernel_args, kernel_attrs }
    }
}

impl CallFFIKernelAttr {
    /// Lossless complete-field allocation.
    pub fn new(
        launch_params: Array<String>,
        launch_fields: Array<String>,
        num_kernel_args: i64,
        kernel_attrs: Map<String, i64>,
    ) -> Self {
        let obj =
            CallFFIKernelAttrObj::new(launch_params, launch_fields, num_kernel_args, kernel_attrs);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(CallFFIKernelAttr => Attrs);
// tvm-ffi-stubgen(end)
