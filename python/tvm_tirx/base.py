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
# coding: utf-8
# pylint: disable=invalid-name, import-outside-toplevel
"""Base library for TVM."""

import ctypes
import os
import sys
from pathlib import Path

import tvm_ffi
from tvm_ffi import libinfo as tvm_ffi_libinfo
from tvm_ffi.libinfo import load_lib_ctypes

from . import libinfo

# ----------------------------
# library loading
# ----------------------------

# Whether only the runtime library is loaded (runtime-only wheel, or
# ``TVM_TIRX_USE_RUNTIME_LIB=1``). Set during library loading below.
_RUNTIME_ONLY = os.environ.get("TVM_TIRX_USE_RUNTIME_LIB") == "1"

# Handles of the core libraries actually loaded, keyed by basename
# (e.g. ``{"tvm_runtime": <CDLL>, "tvm_tirx_compiler": <CDLL>}``). Downstream /
# autoloaded extensions can inspect this to skip duplicate libraries
# (``"tvm_runtime" in _LOADED_LIBS``) and obtain the loaded handle.
_LOADED_LIBS = {}


# runtime is loaded RTLD_GLOBAL to expose its symbols to subsequent loads;
# compiler is loaded RTLD_LOCAL.
_LOADED_LIBS["tvm_runtime"] = load_lib_ctypes(
    "tvm_tirx", "tvm_runtime", "RTLD_GLOBAL", extra_lib_paths=libinfo.package_lib_paths()
)

if not _RUNTIME_ONLY:
    _lib_paths = [Path(_LOADED_LIBS["tvm_runtime"]._name).resolve().parent]
    try:
        tvm_ffi_libinfo._find_library_by_basename(
            "tvm_tirx", "tvm_tirx_compiler", extra_lib_paths=_lib_paths
        )
    except RuntimeError:
        # A runtime-only installation need not contain the compiler library.
        _RUNTIME_ONLY = True
    else:
        _LOADED_LIBS["tvm_tirx_compiler"] = load_lib_ctypes(
            "tvm_tirx", "tvm_tirx_compiler", "RTLD_LOCAL", extra_lib_paths=_lib_paths
        )

# The operators written in Rust are in libtvm_tirx_rust, when the build has it
# (``TVM_TIRX_USE_RUST``). They are registered now, as loading the compiler library registered the
# C++ ones, before any module looks for them; ``TVM_TIRX_RUST_LIBRARY`` names another build of the
# library to load instead, such as ``rust/target/debug/libtvm_tirx.so``. Otherwise only the one next
# to the compiler library is loaded, which the same build made: one found elsewhere, such as on
# ``LD_LIBRARY_PATH``, may be another build's.
if not _RUNTIME_ONLY:
    _rust_lib_path = os.environ.get("TVM_TIRX_RUST_LIBRARY")
    if not _rust_lib_path:
        _rust_lib_path = Path(_LOADED_LIBS["tvm_tirx_compiler"]._name).resolve().parent / (
            "libtvm_tirx_rust.dylib" if sys.platform == "darwin" else "libtvm_tirx_rust.so"
        )
        if not _rust_lib_path.is_file():
            _rust_lib_path = None
    if _rust_lib_path:
        # Loading the library registers its operators, and the handle keeps it, and the hooks it
        # registers, loaded for the process; ``tirx_register_ops`` raises the error of a block
        # that failed.
        _LOADED_LIBS["tvm_tirx_rust"] = ctypes.CDLL(str(_rust_lib_path), ctypes.RTLD_LOCAL)
        tvm_ffi.load_module(str(_rust_lib_path)).tirx_register_ops()

if _RUNTIME_ONLY:
    from tvm_ffi import registry as _tvm_ffi_registry

    _tvm_ffi_registry._SKIP_UNKNOWN_OBJECTS = True
