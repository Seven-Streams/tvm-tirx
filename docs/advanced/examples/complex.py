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
"""This example uses complex numbers that a Rust library adds to the IR.

``rust/examples/complex.rs`` defines the type ``my_ext.Complex``, the operators
``my_ext.complex``, ``my_ext.cmul``, ``my_ext.real`` and ``my_ext.imag``, and the pass
``LowerComplex``, which lowers them to scalar arithmetic. TIRx needs LLVM support. The example
builds the library and runs from the repository root with:
    uv run --no-sync cargo build --manifest-path rust/Cargo.toml --example complex
    uv run --no-sync python docs/advanced/examples/complex.py \\
        rust/target/debug/examples/libcomplex.so
"""

import sys

import numpy as np

# begin load
import tvm_ffi

from tvm_tirx import ir, runtime, tirx
from tvm_tirx.script import tirx as T

# Loading the library registers its type and operators, and their TVMScript names.
library = tvm_ffi.load_module(sys.argv[1])
from tvm_tirx.script.parser import my_ext  # noqa: E402

# end load

# begin type
complex_type = my_ext.Complex(T.float32)
print(complex_type)  # my_ext.Complex(T.float32)
# end type

# begin type check
try:
    my_ext.real(T.float32(1))
except TypeError as error:
    print(error)  # my_ext.real expects a complex value
# end type check

# begin function
n = T.dynamic("n")


@T.function
def cube(
    A: T.Tensor((n,), "float32"),
    B: T.Tensor((n,), "float32"),
    Re: T.Tensor((n,), "float32"),
    Im: T.Tensor((n,), "float32"),
):
    for i in range(n):
        z = my_ext.complex(A[i], B[i])
        w = my_ext.cmul(my_ext.cmul(z, z), z)
        Re[i] = my_ext.real(w)
        Im[i] = my_ext.imag(w)


# end function

print(cube.script())

# begin compile without the pass
try:
    tirx.compile(cube, target="llvm")
except Exception as error:  # pylint: disable=broad-exception-caught
    print(error)  # Error in codegen: llvm, at `z: T.let[my_ext.Complex(T.float32)] = ...`
# end compile without the pass

# begin lower
lower_complex = library.lower_complex()
lowered = lower_complex(ir.IRModule({"main": cube}))
print(lowered["main"].script())
# end lower


# begin pipeline
def complex_pipeline(*, target=None):
    lower, finalize_host, finalize_device = tirx.get_tir_pipeline("default", target=target)
    lower = ir.transform.Sequential([library.lower_complex(), lower])
    return lower, finalize_host, finalize_device


tirx.register_tir_pipeline("complex", complex_pipeline)
compiled = tirx.compile(cube, target="llvm", pipeline="complex")
# end pipeline

# begin run
a, b = np.random.default_rng(0).standard_normal((2, 16)).astype("float32")
re, im = runtime.empty((16,), "float32"), runtime.empty((16,), "float32")
compiled(runtime.tensor(a), runtime.tensor(b), re, im)
expected = (a + 1j * b) ** 3
np.testing.assert_allclose(re.numpy(), expected.real, rtol=1e-5, atol=1e-5)
np.testing.assert_allclose(im.numpy(), expected.imag, rtol=1e-5, atol=1e-5)
# end run
