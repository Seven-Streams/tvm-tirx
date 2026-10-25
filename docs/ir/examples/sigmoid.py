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
"""This example builds and executes model-specific bias + sigmoid programs.

TIRx needs LLVM support. The example runs from the repository root with:
    uv run --no-sync python docs/ir/examples/sigmoid.py

X and Y have shape (2*n, channels); bias has one value per channel.
The code built by make_bias_sigmoid processes one element at a time.
It writes Y[i, j] = sigmoid(X[i, j] + bias[j]) without an intermediate tensor.
The symbol n is derived from tensor shapes at runtime.
Channels selects a model specialization.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

# begin sigmoid expression
from tvm_tirx import ir, tirx
from tvm_tirx.script import from_source
from tvm_tirx.script import tirx as T

exp_op = ir.Op.get("prim.exp")


def sigmoid(x):
    one = ir.const(1, x.ty)
    exp_neg_x = ir.Call(exp_op, [-x])
    return one / (one + exp_neg_x)


# end sigmoid expression

# begin primitive types
index_type = ir.PrimType("int64")
element_type = ir.PrimType("float32")
# end primitive types


def make_bias_sigmoid(channels=128, activation=sigmoid):
    """The factory builds scalar computations inside batch and channel loops."""
    # begin tensor types
    n = ir.Var("n", "int64")
    matrix_type = tirx.TensorType((2 * n, channels), "float32")
    bias_type = tirx.TensorType((channels,), "float32")
    # end tensor types

    # begin variables
    X = ir.Var("X", matrix_type)
    bias = ir.Var("bias", bias_type)
    Y = ir.Var("Y", matrix_type)
    i, j = [ir.Var(name, "int64") for name in ("i", "j")]
    biased = ir.Var("biased", "float32")
    # end variables

    zero, step = [ir.const(value, index_type) for value in (0, 1)]
    end = ir.const(channels, index_type)

    # begin biased activation
    biased_value = X[i, j] + bias[j]
    result = activation(biased)
    # end biased activation

    # begin statement sequence
    body = ir.SeqStmt(
        [
            ir.Bind(biased, biased_value),
            ir.TensorStore(Y, [i, j], result),
        ]
    )
    # end statement sequence

    # begin loop
    channel_loop = ir.For(j, zero, end, step, ir.ForKind.DEFAULT, body)
    batch_loop = ir.For(i, zero, 2 * n, step, ir.ForKind.DEFAULT, channel_loop)
    # end loop

    # begin function construction
    function = tirx.Function([X, bias, Y], batch_loop, ret_type=ir.TupleType([]))
    function = function.with_attr("global_symbol", "bias_sigmoid")
    # end function construction
    return function


# begin module construction
channels_128 = make_bias_sigmoid(128).with_attr("global_symbol", "bias_sigmoid_128")
channels_130 = make_bias_sigmoid(130).with_attr("global_symbol", "bias_sigmoid_130")
module = ir.IRModule({"bias_sigmoid_128": channels_128, "bias_sigmoid_130": channels_130})
# end module construction

# begin module lookup
entry = module.get_global_var("bias_sigmoid_130")
function = module[entry]
# end module lookup

batch_loop = function.body[0]
channel_loop = batch_loop.body[0]
binding, store = channel_loop.body.seq
activation = store.value
exp_call = activation.b.b

# begin operator lookup
op = ir.Op.get("prim.exp")
argument_names = [argument.name for argument in op.args_info]
effect = op.get_attr("TCallEffectKind")
# end operator lookup

# begin validation failure
invalid = ir.Call("prim.exp", [], ty="float32")
# end validation failure
try:
    invalid.validate()
except TypeError:
    pass
else:
    raise AssertionError("The exponential operator requires its input argument")

# begin provisional call
provisional = ir.Call(op, exp_call.args, ty=ir.Type.missing())
inferred = ir.reinfer_type(provisional)
rebuilt_call = ir.Call(op, provisional.args, ty=inferred)
# end provisional call
exp_call.validate()
rebuilt_call.validate()

# begin module update
variant = module.clone()
replacement = make_bias_sigmoid(130, activation=ir.prim.sigmoid)
variant[entry] = replacement.with_attr("global_symbol", "bias_sigmoid_130")
# end module update

# begin module serialization
restored_module = ir.load_json(ir.save_json(module))
ir.assert_structural_equal(module, restored_module)
# end module serialization

# begin print program
source = function.script()
print(source)
# end print program

# begin printing and parsing
restored_function = from_source(source, extra_vars={"T": T})
ir.assert_structural_equal(function, restored_function)
# end printing and parsing


# begin save program
def save_program(program, output_path):
    source = program.script(extra_config={"ir.comment_imports": False})
    output_path.write_text(source)
    restored = from_source(output_path.read_text())
    ir.assert_structural_equal(program, restored)
    return restored


# end save program


def run():
    """Both specializations and the replacement run at two batch sizes."""
    import numpy as np

    from tvm_tirx import runtime

    assert tirx.analysis.check_well_formed(module)
    assert tirx.analysis.check_well_formed(variant)
    assert inferred == element_type
    ir.assert_structural_equal(exp_call, rebuilt_call)
    assert module[entry].same_as(channels_130)

    with TemporaryDirectory() as directory:
        cached = save_program(variant, Path(directory) / "activation.py")
        # begin compilation
        compiled = tirx.compile(cached, target="llvm")
        # end compilation
        baseline = tirx.compile(module, target="llvm")

        rng = np.random.default_rng(0)
        for channels in (128, 130):
            for batch in (2, 16):
                # begin execution
                inputs = rng.normal(size=(batch, channels)).astype("float32")
                offsets = rng.normal(size=channels).astype("float32")
                x = runtime.tensor(inputs)
                bias = runtime.tensor(offsets)
                output = runtime.tensor(np.full_like(inputs, np.nan))
                name = f"bias_sigmoid_{channels}"
                compiled[name](x, bias, output)

                expected = 1.0 / (1.0 + np.exp(-(inputs + offsets)))
                np.testing.assert_allclose(output.numpy(), expected, rtol=1e-6, atol=1e-7)
                # end execution
                baseline_output = runtime.tensor(np.full_like(inputs, np.nan))
                baseline[name](x, bias, baseline_output)
                np.testing.assert_allclose(baseline_output.numpy(), expected, rtol=1e-6, atol=1e-7)
                print(
                    f"bias + sigmoid: shape {(batch, channels)}, both implementations match NumPy"
                )


if __name__ == "__main__":
    run()
