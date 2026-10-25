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
"""This example defines, compiles, and runs a bias + sigmoid program in TVMScript.

TIRx needs LLVM support. The example runs from the repository root with:
    uv run --no-sync python docs/get_started/examples/program.py

X and Y have shape (2*n, channels); bias has one value per channel.
The code in bias_sigmoid processes four channels at a time and handles any
remaining channels with a scalar loop. run() compiles a 130-channel
specialization and compares the output against NumPy.
"""

# begin program
from __future__ import annotations

from tvm_tirx import ir, tirx
from tvm_tirx.script import tirx as T


def make_bias_sigmoid(channels=130):
    n = T.dynamic("n", "int64")

    @T.function
    def bias_sigmoid(
        X: T.Tensor((2 * n, channels), T.float32),
        bias: T.Tensor((channels,), T.float32),
        Y: T.Tensor((2 * n, channels), T.float32),
    ):
        for i in range(2 * n):
            for j in range(T.int64(0), T.int64(channels), T.int64(4)):
                if j + T.int64(4) <= T.int64(channels):
                    biased: T.let[T.float32x4] = (
                        X[i, T.Ramp(j, T.int64(1), 4)] + bias[T.Ramp(j, T.int64(1), 4)]
                    )
                    Y[i, j : j + T.int64(4)] = T.float32(1) / (T.float32(1) + T.exp(-biased))
                else:
                    for k in range(j, T.int64(channels)):
                        activated_tail: T.let[T.float32] = T.float32(1) / (
                            T.float32(1) + T.exp(-(X[i, k] + bias[k]))
                        )
                        Y[i, k] = activated_tail

    return bias_sigmoid


program = ir.IRModule({"bias_sigmoid": make_bias_sigmoid(130)})


def run():
    import numpy as np

    from tvm_tirx import runtime

    function = program["bias_sigmoid"]
    input_var = function.params[0]
    tensor_type = input_var.ty
    row_extent = tensor_type.shape[0]
    location = row_extent.loc
    print("Input type:", tensor_type.script())
    print("Rows:", row_extent.script())
    print("Rows source:", location.source_name.name)
    print("Rows start:", location.start_line, location.start_column)

    print(program.script())
    compiled = tirx.compile(program, target="llvm")
    batch, channels = 16, 130
    rng = np.random.default_rng(0)
    inputs = rng.normal(size=(batch, channels)).astype("float32")
    offsets = rng.normal(size=channels).astype("float32")
    x = runtime.tensor(inputs)
    bias = runtime.tensor(offsets)
    output = runtime.tensor(np.full_like(inputs, np.nan))
    compiled["bias_sigmoid"](x, bias, output)

    expected = 1.0 / (1.0 + np.exp(-(inputs + offsets)))
    np.testing.assert_allclose(output.numpy(), expected, rtol=1e-6, atol=1e-7)
    print("bias + sigmoid: vector and tail paths match NumPy.")


if __name__ == "__main__":
    run()
# end program
