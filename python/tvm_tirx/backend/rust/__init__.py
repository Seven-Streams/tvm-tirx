# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Rust-owned backend hooks."""

from pathlib import Path

import tvm_ffi
from tvm_ffi.libinfo import load_lib_ctypes

from tvm_tirx.base import _LOADED_LIBS


def register_backend():
    """Register Rust-owned Python semantics."""
    runtime_dir = Path(_LOADED_LIBS["tvm_runtime"]._name).resolve().parent
    try:
        _LOADED_LIBS["tvm_tirx_rust"] = load_lib_ctypes(
            package="tvm_tirx",
            target_name="tvm_tirx_rust",
            extra_lib_paths=[runtime_dir],
            mode="RTLD_LOCAL",
        )
    except (OSError, FileNotFoundError, RuntimeError):
        pass
    else:
        # Loading the library registers its operators and types, and keeps it, and their hooks,
        # loaded for the process; ``tirx_register_ir`` raises the error of a block that failed.
        tvm_ffi.load_module(_LOADED_LIBS["tvm_tirx_rust"]._name).tirx_register_ir()
    return None


__all__ = ["register_backend"]
