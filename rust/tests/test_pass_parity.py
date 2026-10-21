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
"""Parity of the Rust tirx passes with their C++ counterparts on the recorded corpus.

Build the crate first (``cargo build`` in ``rust/``), record the corpus
(``rust/tests/corpus/generate.sh``), then run this file with pytest from a Python where
``tvm_tirx`` is importable. Each case runs ``tirx.transform.<Name>`` and the Rust
``rust.tirx.transform.<Name>`` on every record, inside a PassContext carrying the recorded
config, and requires structurally equal results or a failure on both sides.

Environment: ``TVM_RUST_PASSES`` (the built library), ``TVM_PASS_CORPUS`` (the corpus).
"""

import contextlib
import ctypes
import os
import sys
from pathlib import Path

import pytest
import tvm_ffi

import tvm_tirx

HERE = Path(__file__).resolve().parent
CORPUS = Path(os.environ.get("TVM_PASS_CORPUS", HERE / "corpus" / "data"))
LIBRARY = Path(
    os.environ.get("TVM_RUST_PASSES", HERE.parent / "target" / "debug" / "libtvm_tirx.so")
)

_LIB = ctypes.CDLL(str(LIBRARY))
assert _LIB.tvm_rust_register_passes() == 0, "registering the Rust passes failed"


def _is_host_function(func):
    """The host/device split `tvm_tirx.tirx.compile` makes with `tirx.transform.Filter`."""
    target = func.attrs.get("target", tvm_tirx.target.Target("llvm"))
    return target.kind.name in ["llvm", "c", "cuda_host"]


# Case -> the factory it compares, the corpus directories (pass names) whose records it runs
# on, and the factory arguments.
CASES = {
    # No test runs SkipAssert itself; host modules around MakePackedAPI carry the asserts.
    "SkipAssert": dict(sources=["tirx.MakePackedAPI", "tirx.LowerTVMBuiltin"]),
    "StmtSimplify": dict(sources=["tirx.StmtSimplify", "tirx.RemoveNoOp", "tirx.LowerIntrin"]),
    "RemoveNoOp": dict(sources=["tirx.RemoveNoOp", "tirx.StmtSimplify", "tirx.LowerIntrin"]),
    "ConvertSSA": dict(sources=["tirx.ConvertSSA", "tirx.StmtSimplify"]),
    "DecorateDeviceScope": dict(sources=["tirx.DecorateDeviceScope", "tirx.StmtSimplify"]),
    "AnnotateEntryFunc": dict(
        sources=["tirx.AnnotateEntryFunc", "tirx.SplitHostDevice", "tirx.MakePackedAPI"]
    ),
    # The recorded inputs do not carry the target BindTarget ran with; try three shapes.
    "BindTarget[cuda+llvm]": dict(
        factory="BindTarget",
        sources=["tirx.BindTarget"],
        args=lambda: (tvm_tirx.target.Target("cuda", host="llvm"),),
    ),
    "BindTarget[llvm]": dict(
        factory="BindTarget",
        sources=["tirx.BindTarget"],
        args=lambda: (tvm_tirx.target.Target("llvm"),),
    ),
    "BindTarget[cuda]": dict(
        factory="BindTarget",
        sources=["tirx.BindTarget"],
        args=lambda: (tvm_tirx.target.Target("cuda"),),
    ),
    "VerifyMemory": dict(sources=["tirx.VerifyMemory", "tirx.SplitHostDevice", "tirx.BindTarget"]),
    # The inputs of other passes give CSE many more expressions to share.
    "CommonSubexprElim": dict(
        sources=[
            "tirx.CommonSubexprElim",
            "tirx.NarrowDataType",
            "tirx.StmtSimplify",
            "tirx.FlattenTensor",
            "tirx.UnrollLoop",
            "tirx.VectorizeLoop",
        ]
    ),
    "HoistIf": dict(sources=["tirx.HoistIf"]),
    "UnrollLoop": dict(sources=["tirx.UnrollLoop", "tirx.VectorizeLoop"]),
    # Inputs of the passes around LowerIntrin in the pipeline carry the `target` attribute too.
    "LowerIntrin": dict(
        sources=[
            "tirx.LowerIntrin",
            "tirx.MakePackedAPI",
            "tirx.LowerTVMBuiltin",
            "tirx.SplitHostDevice",
        ]
    ),
    # InsertHoistedIf has no factory of its own; it is the first pass of the native HoistIf.
    "InsertHoistedIf": dict(
        sources=["tirx.InsertHoistedIf", "tirx.StmtSimplify"],
        native=lambda: tvm_tirx.tirx.transform.HoistIf().passes[0],
    ),
    "Filter[host]": dict(factory="Filter", sources=["tirx.Filter"], args=(_is_host_function,)),
    "Filter[device]": dict(
        factory="Filter", sources=["tirx.Filter"], args=(lambda f: not _is_host_function(f),)
    ),
    # The legalizers leave functions alone on a target with native bfloat16 / float8 support, so
    # the records of every pass around them in the host pipeline are inputs too.
    "BF16ComputeLegalize": dict(
        sources=["tirx.BF16ComputeLegalize", "tirx.BF16StorageLegalize", "tirx.FP8ComputeLegalize"]
    ),
    "BF16StorageLegalize": dict(
        sources=["tirx.BF16StorageLegalize", "tirx.BF16ComputeLegalize", "tirx.FP8StorageLegalize"]
    ),
    "FP8ComputeLegalize[float16]": dict(
        factory="FP8ComputeLegalize",
        sources=["tirx.FP8ComputeLegalize", "tirx.BF16ComputeLegalize"],
        args=("float16",),
    ),
    "FP8ComputeLegalize[float32]": dict(
        factory="FP8ComputeLegalize",
        sources=["tirx.FP8ComputeLegalize", "tirx.BF16ComputeLegalize"],
        args=("float32",),
    ),
    "FP8StorageLegalize": dict(
        sources=["tirx.FP8StorageLegalize", "tirx.FP8ComputeLegalize", "tirx.BF16StorageLegalize"]
    ),
    # Few inputs come to RenormalizeSplitPattern itself; the split patterns appear in the inputs of
    # many passes after it.
    "RenormalizeSplitPattern": dict(
        sources=[
            "tirx.RenormalizeSplitPattern",
            "tirx.StmtSimplify",
            "tirx.LowerTVMBuiltin",
            "tirx.LowerIntrin",
            "tirx.LowerTensorLayouts",
            "tirx.TileDispatch",
            "tirx.VectorizeLoop",
            "tirx.NarrowDataType",
            "tirx.MakePackedAPI",
        ]
    ),
    "NarrowDataType": dict(
        sources=[
            "tirx.NarrowDataType",
            "tirx.FlattenTensor",
            "tirx.VectorizeLoop",
            "tirx.UnrollLoop",
            "tirx.LowerThreadAllreduce",
        ],
        args=(32,),
    ),
    # A narrower target narrows the int32 indices, shapes and thread extents of the corpus too.
    "NarrowDataType[16]": dict(
        factory="NarrowDataType",
        sources=[
            "tirx.NarrowDataType",
            "tirx.BF16ComputeLegalize",
            "tirx.FlattenTensor",
            "tirx.VectorizeLoop",
            "tirx.LowerThreadAllreduce",
        ],
        args=(16,),
    ),
    # ForceNarrowIndexToInt32 is not in the default pipeline; it runs on NarrowDataType's inputs
    # and on the lowered modules around it.
    "ForceNarrowIndexToInt32": dict(
        sources=[
            "tirx.NarrowDataType",
            "tirx.FlattenTensor",
            "tirx.LowerTensorLayouts",
            "tirx.MakePackedAPI",
            "tirx.VectorizeLoop",
        ]
    ),
}


def _records(sources):
    for source in sources:
        yield from sorted((CORPUS / source).glob("*.json"))


def _run(pass_, text, target=None):
    """Run `pass_` on a fresh copy of the record: module passes may update their input.

    `target`, when given, is the current target around the run (`Target::Current`); the corpus
    does not record the one the pass ran under.
    """
    record = tvm_tirx.ir.load_json(text)
    with contextlib.ExitStack() as stack:
        if target is not None:
            stack.enter_context(tvm_tirx.target.Target(target))
        stack.enter_context(tvm_tirx.transform.PassContext(config=dict(record["config"])))
        return pass_(record["mod"])


@pytest.mark.parametrize("case", sorted(CASES))
def test_parity(case):
    spec = CASES[case]
    factory = spec.get("factory", case)
    args = spec.get("args", ())
    args = args() if callable(args) else args
    if "native" in spec:
        native = spec["native"]()
    else:
        native = tvm_ffi.get_global_func(f"tirx.transform.{factory}")(*args)
    rust = tvm_ffi.get_global_func(f"rust.tirx.transform.{factory}")(*args)
    compared = both_failed = unloadable = changed = 0
    failures = []
    for path in _records(spec["sources"]):
        text = path.read_text()
        try:
            original = tvm_tirx.ir.load_json(text)["mod"]
        except Exception:
            unloadable += 1
            continue
        outcome = []
        for pass_ in (native, rust):
            try:
                outcome.append(("ok", _run(pass_, text, spec.get("target"))))
            except Exception as error:
                outcome.append(("error", error))
        compared += 1
        (native_kind, expected), (rust_kind, actual) = outcome
        if native_kind == rust_kind == "error":
            both_failed += 1
        elif native_kind != rust_kind:
            detail = expected if native_kind == "error" else actual
            failures.append(f"{path.name}: native {native_kind}, rust {rust_kind}: {detail!s:.300}")
        else:
            changed += not tvm_ffi.structural_equal(original, expected, map_free_vars=True)
            if not tvm_ffi.structural_equal(expected, actual, map_free_vars=True):
                mismatch = tvm_ffi.get_first_structural_mismatch(
                    expected, actual, map_free_vars=True
                )
                failures.append(f"{path.name}: results differ at {mismatch}")
    print(
        f"{case}: {compared} compared ({changed} changed by the pass), {both_failed} failed "
        f"on both sides, {unloadable} unloadable, {len(failures)} disagree"
    )
    assert compared > 0, f"{case}: no loadable records"
    assert not failures, "\n".join(failures[:20])


def _verify_memory_modules():
    """A store to a CUDA tensor argument on the host side, and the same store inside a thread."""
    from tvm_tirx.script import tirx as T

    @T.function
    def host_access(A: T.Tensor((16,), "float32")):
        T.func_attr({"target": T.target("cuda"), "global_symbol": "host_access"})
        A[0] = T.float32(1)

    @T.function
    def thread_access(A: T.Tensor((16,), "float32")):
        T.func_attr({"target": T.target("cuda"), "global_symbol": "thread_access"})
        with T.launch_thread("threadIdx.x", 16) as tx:
            A[tx] = T.float32(1)

    return tvm_tirx.IRModule({"main": host_access}), tvm_tirx.IRModule({"main": thread_access})


def test_verify_memory_rejects_host_access():
    rejected, accepted = _verify_memory_modules()
    for factory in ("tirx.transform.VerifyMemory", "rust.tirx.transform.VerifyMemory"):
        pass_ = tvm_ffi.get_global_func(factory)()
        with pytest.raises(Exception, match="directly accessed by host memory"):
            pass_(rejected)
        pass_(accepted)


def _lower_intrin_module():
    """Every LowerIntrin rewrite: floordiv/floormod of unknown, constrained, and shiftable sign,
    power-of-two divisors, the max/eq patterns, CUDA and LLVM intrinsic rules, and fma fusion."""
    from tvm_tirx.script import tirx as T

    @T.function
    def cuda(A: T.Tensor((16,), "int32"), B: T.Tensor((16,), "float32"), x: T.int32, y: T.int32):
        T.func_attr({"target": T.target("cuda"), "global_symbol": "cuda"})
        with T.launch_thread("threadIdx.x", 16) as tx:
            A[0] = T.floordiv(x, y)
            A[1] = T.floormod(x, y)
            A[2] = T.floordiv(x, 8)
            A[3] = T.floormod(x, 8)
            A[4] = T.Select(y >= 0, T.floordiv(x, y), 0)
            A[5] = T.floordiv(tx - 4, 5)
            A[6] = T.Select(y >= 0, T.max(T.floordiv(x, y), 0), 0)
            A[7] = T.Select(T.floormod(x, y) == 0, 1, 0)
            A[8] = T.if_then_else(y > 0, T.floordiv(x, y), 0)
            A[9] = T.floormod(tx - 4, 5)
            B[tx] = B[tx] * B[0] + B[1]
            B[1] = T.exp(B[2])
            if y > 0:
                A[10] = T.floordiv(x, y)
            else:
                A[11] = T.floormod(x, y)

    @T.function
    def llvm(
        C: T.Tensor((16,), "float32"),
        D: T.Tensor((16,), "float16"),
        E: T.Tensor((4,), "float32x4"),
        n: T.int32,
    ):
        T.func_attr({"target": T.target("llvm"), "global_symbol": "llvm"})
        for i in range(n):
            C[i] = C[i] * C[0] + C[1]
            C[1] = T.exp(C[2]) + T.floor(C[3])
            E[0] = T.Broadcast(T.Cast("float32", D[0]), 4) * E[1] + E[2]
            C[2] = T.Cast("float32", T.floordiv(i - 3, 4))

    return tvm_tirx.IRModule({"cuda": cuda, "llvm": llvm})


def test_lower_intrin_branches():
    native = tvm_tirx.tirx.transform.LowerIntrin()(_lower_intrin_module())
    rust = tvm_ffi.get_global_func("rust.tirx.transform.LowerIntrin")()(_lower_intrin_module())
    assert not tvm_ffi.structural_equal(native, _lower_intrin_module(), map_free_vars=True)
    assert tvm_ffi.structural_equal(native, rust, map_free_vars=True)


def test_call_attrs_are_left_alone():
    """The engine's default recursion follows the native structural hooks, which skip Call.attrs
    like C++ StmtExprMutator: a floordiv in attrs stays while the one in args is lowered."""
    tvm_tirx.ir._ffi_api.RegisterOp("test.parity.call_attrs", "")
    op = tvm_tirx.ir.Op.get("test.parity.call_attrs")

    def module():
        x, y = tvm_tirx.tirx.Var("x", "int32"), tvm_tirx.tirx.Var("y", "int32")
        div = tvm_tirx.tirx.floordiv(x, y)
        call = tvm_tirx.ir.Call(op, [div], attrs={"k": div}, ty=tvm_tirx.ir.PrimType("int32"))
        func = tvm_tirx.tirx.Function([x, y], tvm_tirx.ir.Evaluate(call)).with_attr(
            {"target": tvm_tirx.target.Target("cuda"), "global_symbol": "main"}
        )
        return tvm_tirx.IRModule({"main": func})

    native = tvm_tirx.tirx.transform.LowerIntrin()(module())
    rust = tvm_ffi.get_global_func("rust.tirx.transform.LowerIntrin")()(module())
    call = rust["main"].body.seq[0].value
    assert not isinstance(call.args[0], tvm_tirx.tirx.FloorDiv)
    assert isinstance(call.attrs["k"], tvm_tirx.tirx.FloorDiv)
    assert tvm_ffi.structural_equal(native, rust, map_free_vars=True)


def _remove_no_op_module():
    """Every RemoveNoOp rewrite: empty and provably empty loops, no-op branches on either side,
    pure evaluates, and stores of the value already in place."""
    from tvm_tirx.script import tirx as T

    @T.function
    def main(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32"), n: T.int32, m: T.int32):
        T.func_attr({"global_symbol": "main"})
        for i in range(n):
            for j in range(i - n):
                A[j] = T.float32(1)
            for k in range(m):
                T.evaluate(0)
            if n > 4:
                A[i] = A[i]
            else:
                B[i] = T.float32(2)
            if m > 2:
                T.evaluate(n + 1)
            else:
                B[0] = A[0] + T.float32(0)
            if n > m:
                B[1] = T.float32(3)
            else:
                T.evaluate(0)
            T.evaluate(n * m)
        for z in range(0):
            A[0] = T.float32(4)

    return tvm_tirx.IRModule({"main": main})


def test_remove_no_op_branches():
    native = tvm_tirx.tirx.transform.RemoveNoOp()(_remove_no_op_module())
    rust = tvm_ffi.get_global_func("rust.tirx.transform.RemoveNoOp")()(_remove_no_op_module())
    assert not tvm_ffi.structural_equal(native, _remove_no_op_module(), map_free_vars=True)
    assert tvm_ffi.structural_equal(native, rust, map_free_vars=True)


def _stmt_simplify_module():
    """StmtSimplify facts: tensor shapes are non-negative, loop ranges, pure bindings substituted
    into branch conditions, branch constraints, and a store of the loaded value."""
    from tvm_tirx.script import tirx as T

    n = T.dynamic("n")

    @T.function(check_well_formed=False)
    def main(A: T.Tensor((n * 32 + 1 - 2,), "float32"), B: T.Tensor((16,), "float32"), m: T.int32):
        T.func_attr({"global_symbol": "main"})
        A[T.min(T.int64(1), n)] = T.float32(0)
        for i in range(16):
            if i < 16:
                B[i] = B[i] + T.float32(1)
            k: T.let = i * 2
            if k < 32:
                B[i] = B[i]
            if m > 4:
                if m > 2:
                    B[i] = T.Select(m > 1, T.float32(2), T.float32(3))
            B[T.floordiv(i, 16)] = T.float32(4)

    return tvm_tirx.IRModule({"main": main})


def test_stmt_simplify_facts():
    native = tvm_tirx.tirx.transform.StmtSimplify()(_stmt_simplify_module())
    rust = tvm_ffi.get_global_func("rust.tirx.transform.StmtSimplify")()(_stmt_simplify_module())
    assert not tvm_ffi.structural_equal(native, _stmt_simplify_module(), map_free_vars=True)
    assert tvm_ffi.structural_equal(native, rust, map_free_vars=True)


def _string_binding_module():
    from tvm_tirx.script import tirx as T

    @T.function
    def main(A: T.Tensor((4,), "int32")):
        s: T.let = T.StringImm("hello")
        T.evaluate(T.call_extern("printf", s, ty="int32"))
        A[0] = 1

    return tvm_tirx.IRModule({"main": main})


def test_stmt_simplify_inlines_string_bindings():
    native = tvm_tirx.tirx.transform.StmtSimplify()(_string_binding_module())
    rust = tvm_ffi.get_global_func("rust.tirx.transform.StmtSimplify")()(_string_binding_module())
    assert not tvm_ffi.structural_equal(native, _string_binding_module(), map_free_vars=True)
    assert tvm_ffi.structural_equal(native, rust, map_free_vars=True)


def test_common_subexpr_elim_unit_inputs(monkeypatch):
    """Every module the C++ CSE unit tests run the pass on: scopes of If, For and While, Let
    bodies, bool predicates, calls and loads, and a statement shared by unrolling."""
    import importlib.util

    path = HERE.parents[1] / "tests/python/tirx/transform/test_common_subexpr_elim.py"
    spec = importlib.util.spec_from_file_location("cse_unit_tests", path)
    tests = importlib.util.module_from_spec(spec)
    # TVMScript reads the source of the classes it parses through the module.
    monkeypatch.setitem(sys.modules, spec.name, tests)
    spec.loader.exec_module(tests)
    native_factory = tvm_tirx.tirx.transform.CommonSubexprElim
    rust_factory = tvm_ffi.get_global_func("rust.tirx.transform.CommonSubexprElim")
    compared = []

    def both():
        native, rust = native_factory(), rust_factory()

        def run(mod):
            # The Rust pass leaves its input alone; the native one may update it in place.
            actual = rust(mod)
            expected = native(mod)
            assert tvm_ffi.structural_equal(expected, actual, map_free_vars=True)
            compared.append(expected)
            return expected

        return run

    monkeypatch.setattr(tvm_tirx.tirx.transform, "CommonSubexprElim", both)
    for name in sorted(dir(tests)):
        if name.startswith("test_"):
            getattr(tests, name)()
    assert len(compared) >= 20


_DTYPE_LEGALIZE_PASSES = [
    "BF16ComputeLegalize",
    "BF16StorageLegalize",
    "FP8ComputeLegalize",
    "FP8StorageLegalize",
]


def _comparing_factory(name, compared):
    """A stand-in for `tvm_tirx.tirx.transform.<name>` whose passes run the native and the Rust pass
    on copies of the input, require the same result (or the same error), and return the native
    one."""
    native_factory = getattr(tvm_tirx.tirx.transform, name)
    rust_factory = tvm_ffi.get_global_func(f"rust.tirx.transform.{name}")

    def factory(*args):
        native, rust = native_factory(*args), rust_factory(*args)

        def run(mod):
            text = tvm_tirx.ir.save_json(mod)
            try:
                expected = native(tvm_tirx.ir.load_json(text))
            except Exception as error:
                with pytest.raises(type(error)):
                    rust(tvm_tirx.ir.load_json(text))
                raise
            actual = rust(tvm_tirx.ir.load_json(text))
            tvm_tirx.ir.assert_structural_equal(actual, expected, map_free_vars=True)
            compared.append(name)
            return expected

        return run

    return factory


@pytest.mark.parametrize("family", ["bf16", "fp8"])
def test_dtype_legalize_unit_inputs(family, monkeypatch):
    """Every input of the native bf16 / fp8 legalize tests, legalized by both passes."""
    import importlib.util
    import inspect

    path = HERE.parent.parent / "tests/python/tirx/transform" / (f"test_{family}_legalize.py")
    spec = importlib.util.spec_from_file_location(f"_native_{family}_legalize_tests", path)
    module = importlib.util.module_from_spec(spec)
    # TVMScript reads the source of the module a parsed function belongs to.
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    compared = []
    for name in _DTYPE_LEGALIZE_PASSES:
        monkeypatch.setattr(tvm_tirx.tirx.transform, name, _comparing_factory(name, compared))
    for test_name in sorted(dir(module)):
        test = getattr(module, test_name)
        if not test_name.startswith("test_") or not callable(test):
            continue
        if not inspect.signature(test).parameters:
            test()
            continue
        for dtype in ("float8_e4m3fn", "float8_e5m2"):
            for promote_dtype in ("float16", "float32"):
                test(dtype=dtype, promote_dtype=promote_dtype)
    print(f"{family}: {len(compared)} pass runs compared")
    assert compared


@pytest.mark.parametrize(
    "dtype",
    [
        "float8_e3m4",
        "float8_e4m3",
        "float8_e4m3b11fnuz",
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
    ],
)
@pytest.mark.parametrize("promote_dtype", ["float16", "float32", "float64", "bfloat16"])
def test_fp8_legalize_formats(dtype, promote_dtype, monkeypatch):
    """Compute then storage legalization of every float8 format, through each promoted type: the
    rounding, exponent and bias branches of the bit-level conversion."""
    import importlib.util

    path = HERE.parent.parent / "tests/python/tirx/transform/test_fp8_legalize.py"
    spec = importlib.util.spec_from_file_location("_native_fp8_formats", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    compared = []
    compute = _comparing_factory("FP8ComputeLegalize", compared)
    storage = _comparing_factory("FP8StorageLegalize", compared)
    target = tvm_tirx.target.Target("llvm")
    before = tvm_tirx.tirx.transform.BindTarget(target)(module.get_before(dtype))
    try:
        after = storage()(compute(promote_dtype)(before))
    except Exception:  # pylint: disable=broad-except
        # Both sides failed with the same kind of error (checked by the stand-in factory).
        return
    assert compared == ["FP8ComputeLegalize", "FP8StorageLegalize"]
    assert not tvm_ffi.structural_equal(after, before)


def _legalize_module(dtype):
    """Scalar and vector computation on `dtype`: bindings, Let, Select, min / max, comparisons,
    casts, constants, broadcasts and shuffles; one allocation is reached through a pointer, so
    it keeps its type, and one is not, so compute legalization promotes it."""
    from tvm_tirx.script import ir as I
    from tvm_tirx.script import tirx as T

    z = T.dynamic("z", dtype)

    @I.ir_module
    class Module:
        @T.function
        def scalars(Aptr: T.handle(dtype), Bptr: T.handle("float32")):
            T.func_attr({"global_symbol": "scalars"})
            A = T.decl_tensor(Aptr, ty_args=[T.Tensor((8,), dtype, data=Aptr)])
            B = T.decl_tensor(Bptr, ty_args=[T.Tensor((8,), "float32", data=Bptr)])
            C = T.alloc_tensor(ty_args=[T.Tensor((8,), dtype)])
            for i in range(8):
                x: T.let = A[i] * T.FloatImm(dtype, 2.0)
                y: T.let = T.max(x, T.min(A[i], T.FloatImm(dtype, 1.5)))
                C[i] = T.Select(x < y, x, T.Cast(dtype, B[i]))
                B[i] = T.Cast("float32", C[i]) + T.Cast("float32", T.Let(z + y, where={z: A[i]}))
            T.evaluate(T.call_extern("consume", C.data, ty="void"))

        @T.function
        def vectors(Aptr: T.handle(dtype), Bptr: T.handle(dtype)):
            T.func_attr({"global_symbol": "vectors"})
            A = T.decl_tensor(Aptr, ty_args=[T.Tensor((8,), dtype, data=Aptr)])
            B = T.decl_tensor(Bptr, ty_args=[T.Tensor((8,), dtype, data=Bptr)])
            D = T.alloc_tensor(ty_args=[T.Tensor((8,), dtype)])
            D[T.Ramp(0, 1, 4)] = A[T.Ramp(0, 1, 4)] + T.Broadcast(T.FloatImm(dtype, 1.0), 4)
            D[T.Ramp(4, 1, 4)] = T.Shuffle([A[T.Ramp(4, 1, 4)]], [3, 2, 1, 0])
            B[T.Ramp(0, 1, 4)] = T.max(D[T.Ramp(0, 1, 4)], D[T.Ramp(4, 1, 4)])
            for i in range(8):
                if A[i] >= B[i]:
                    B[i] = T.exp(A[i]) / B[i] - A[i]
                B[i] = T.Cast(dtype, T.Cast("float32", A[i]) * T.float32(0.5))

    return Module


@pytest.mark.parametrize(
    "dtype, compute, compute_args, storage",
    [
        ("bfloat16", "BF16ComputeLegalize", (), "BF16StorageLegalize"),
        ("float8_e4m3fn", "FP8ComputeLegalize", ("float16",), "FP8StorageLegalize"),
        ("float8_e5m2", "FP8ComputeLegalize", ("float32",), "FP8StorageLegalize"),
    ],
)
@pytest.mark.parametrize("target", [None, "llvm", "nvidia/geforce-rtx-2080-ti", "cuda"])
def test_dtype_legalize_branches(dtype, compute, compute_args, storage, target):
    compared = []
    module = _legalize_module(dtype)
    if target is not None:
        module = tvm_tirx.tirx.transform.BindTarget(tvm_tirx.target.Target(target))(module)
    after_compute = _comparing_factory(compute, compared)(*compute_args)(module)
    _comparing_factory(storage, compared)()(after_compute)
    assert compared == [compute, storage]


def _renormalize_split_pattern_module():
    """Every RenormalizeSplitPattern rule, matching and not: floordiv of floormod (scalar and
    broadcast, divisible or not), unsigned types, the factoring rules with two and three terms
    (scalar and broadcast, positive or not, with or without a common factor), `x < c`, the
    comparison rewrites, and the negation rules, inside and outside a loop."""
    t = tvm_tirx.tirx
    x, y, z = t.Var("x", "int32"), t.Var("y", "int32"), t.Var("z", "int32")
    u, w = t.Var("u", "uint32"), t.Var("w", "uint32")
    v, q = t.Var("v", "int32x4"), t.Var("q", "int32x4")
    i = t.Var("i", "int32")

    def c(value, dtype="int32"):
        return tvm_tirx.ir.IntImm(dtype, value)

    def bc(value, lanes=4):
        return tvm_tirx.ir.prim.Broadcast(c(value), lanes)

    exprs = [
        t.FloorDiv(t.FloorMod(x, c(16)), c(4)),
        t.FloorDiv(t.FloorMod(x, c(15)), c(4)),
        t.FloorDiv(t.FloorMod(v, bc(16)), bc(4)),
        t.FloorDiv(t.FloorMod(v, bc(15)), bc(4)),
        t.FloorDiv(t.FloorMod(u, c(16, "uint32")), c(4, "uint32")),
        t.FloorDiv(t.Add(t.Mul(u, c(6, "uint32")), w), c(4, "uint32")),
        t.FloorDiv(t.Add(t.Mul(x, c(6)), y), c(4)),
        t.FloorDiv(t.Add(t.Mul(x, c(3)), y), c(4)),
        t.FloorDiv(t.Add(t.Mul(x, c(-6)), y), c(4)),
        t.FloorDiv(t.Add(t.Mul(x, c(6)), y), c(-4)),
        t.FloorDiv(t.Add(t.Mul(v, bc(6)), q), bc(4)),
        t.FloorDiv(t.Add(t.Mul(v, bc(3)), q), bc(4)),
        t.FloorDiv(t.Add(t.Add(t.Mul(x, c(6)), y), z), c(4)),
        t.FloorDiv(t.Add(t.Add(t.Mul(x, c(5)), y), z), c(4)),
        t.FloorDiv(t.Add(t.Add(t.Mul(v, bc(6)), q), v), bc(4)),
        t.FloorDiv(t.Add(t.Mul(x, c(12)), y), c(8)),
        t.FloorDiv(t.Add(t.Mul(x, c(64)), y), c(32)),
        t.FloorDiv(t.FloorMod(t.Add(t.Mul(x, c(8)), y), c(64)), c(4)),
        t.FloorDiv(x, y),
        t.FloorDiv(c(17), c(4)),
        t.LT(t.Add(x, c(1)), c(8)),
        t.LT(x, c(8)),
        t.LT(t.Mul(x, c(4)), c(16)),
        t.LT(t.Add(x, c(1)), c(0)),
        t.LT(t.Add(x, c(1)), y),
        t.LE(x, t.Add(y, c(1))),
        t.GT(x, t.Mul(y, c(2))),
        t.GE(t.Add(x, c(3)), c(8)),
        t.Not(t.Not(t.LT(x, y))),
        t.Not(t.LE(x, y)),
        t.Not(t.GE(x, y)),
        t.Not(t.LT(x, y)),
        t.Not(t.GT(x, y)),
        t.Not(t.LT(t.Add(x, c(2)), c(6))),
        t.Select(t.LT(x, c(4)), t.FloorDiv(t.FloorMod(x, c(16)), c(4)), y),
    ]
    body = [tvm_tirx.ir.Evaluate(e) for e in exprs]
    loop_body = tvm_tirx.ir.SeqStmt(
        [
            tvm_tirx.ir.Evaluate(t.FloorDiv(t.FloorMod(t.Add(t.Mul(i, c(4)), x), c(32)), c(4))),
            tvm_tirx.ir.Evaluate(t.LT(t.Add(t.Mul(i, c(2)), c(1)), c(10))),
        ]
    )
    body.append(tvm_tirx.ir.For(i, c(0), c(8), c(1), tvm_tirx.ir.ForKind.DEFAULT, loop_body))
    func = t.Function([x, y, z, u, w, v, q], tvm_tirx.ir.SeqStmt(body)).with_attr(
        {"global_symbol": "main"}
    )
    return tvm_tirx.IRModule({"main": func})


def test_renormalize_split_pattern_rules():
    native = tvm_tirx.tirx.transform.RenormalizeSplitPattern()(_renormalize_split_pattern_module())
    rust = tvm_ffi.get_global_func("rust.tirx.transform.RenormalizeSplitPattern")()(
        _renormalize_split_pattern_module()
    )
    assert not tvm_ffi.structural_equal(
        native, _renormalize_split_pattern_module(), map_free_vars=True
    )
    assert tvm_ffi.structural_equal(native, rust, map_free_vars=True), (
        tvm_ffi.get_first_structural_mismatch(native, rust, map_free_vars=True)
    )


def _narrow_datatype_module():
    """NarrowDataType branches: casts, Lets (nested, with bodies analyzed under the variable's
    bound), Select/if_then_else, arithmetic and comparisons, clz, shifts and bitwise operations,
    vector lanes, unused loop variables, symbolic and overflowing extents, and launched threads."""
    from tvm_tirx.script import tirx as T

    x, y, z, w = (tvm_tirx.tirx.Var(name, "int64") for name in "xyzw")

    @T.function
    def exprs(A: T.Tensor((T.int64(64),), "float32"), B: T.Tensor((T.int64(64),), "int64")):
        T.func_attr({"global_symbol": "exprs"})
        for i in range(T.int64(8)):
            A[i * T.int64(8)] = A[T.Cast("int64", i)] + T.float32(1)
        for i in range(T.int64(8)):
            A[T.Let(x * T.int64(2), where={x: i})] = T.float32(2)
        for i in range(T.int64(8)):
            A[i] = A[T.Let(y * T.int64(2), where={y: T.int64(3)})]
        for i in range(T.int64(8)):
            A[T.Let(T.Let(z + w, where={w: z * T.int64(2)}), where={z: i})] = T.float32(2)
        for i, j in T.grid(T.int64(8), T.int64(8)):
            A[T.Select(i < j, i, j)] = A[T.if_then_else(i >= T.int64(1), i - T.int64(1), j)]
        for i, j in T.grid(T.int64(8), T.int64(8)):
            A[T.min(i, j) + T.max(i, j) + T.floordiv(i, T.int64(2)) + T.floormod(j, T.int64(3))] = (
                A[T.truncdiv(i, T.int64(2)) + T.truncmod(j, T.int64(3))]
            )
        for i in range(T.int64(8)):
            if (
                (i * T.int64(8) != T.int64(5) and i <= T.int64(6))
                or i > T.int64(7)
                or i == T.int64(1)
            ):
                A[i] = T.float32(3)
        for i in range(T.int64(8)):
            B[i] = T.bitwise_not(i) + T.Cast("int64", T.clz(i))

    @T.function
    def shifts(A: T.Tensor((T.int64(64),), "float32")):
        T.func_attr({"global_symbol": "shifts"})
        for i in range(T.int64(8)):
            A[T.shift_right(i, T.int64(1)) + T.shift_left(i, T.int64(2))] = T.float32(3)
        for i in range(T.int64(8)):
            A[T.bitwise_and(i, T.int64(7)) + T.bitwise_or(i, T.int64(1))] = T.float32(4)

    @T.function
    def vectors(A: T.Tensor((T.int64(64),), "float32"), C: T.Tensor((T.int64(64),), "float32")):
        T.func_attr({"global_symbol": "vectors"})
        for i in range(T.int64(8)):
            A[T.Ramp(i * T.int64(4), T.int64(1), 4)] = T.Broadcast(T.float32(4), 4)
            A[T.Broadcast(i, 4)] = T.Broadcast(T.float32(5), 4)
            C[i] = T.pow(C[i], T.float32(2))
        for i in range(T.int64(16)):
            A[i] = A[T.Shuffle([T.Ramp(i, T.int64(1), 2)], [T.int64(1)])]

    @T.function
    def extents(
        A: T.Tensor((T.int64(4096),), "float32"), B: T.Tensor((T.int64(4096),), "int64"), n: T.int64
    ):
        T.func_attr({"global_symbol": "extents"})
        for k in range(T.int64(8)):
            T.evaluate(0)
        for m in range(n):
            A[m] = T.float32(5)
        for v in range(T.int64(2**33)):
            B[T.floormod(v, T.int64(4096))] = v

    @T.function
    def threads(A: T.Tensor((T.int64(2048),), "float32")):
        T.func_attr({"global_symbol": "threads"})
        bx = T.launch_thread("blockIdx.x", T.int64(16))
        tx = T.launch_thread("threadIdx.x", T.int64(128))
        A[bx * T.int64(128) + tx] = A[tx] + T.float32(1)

    @T.function
    def small(A: T.Tensor((256,), "float32"), B: T.Tensor((70000,), "float32")):
        T.func_attr({"global_symbol": "small"})
        for i in range(16):
            for j in range(16):
                A[i * 16 + j] = A[j] * T.float32(2)
        for u in range(4):
            T.evaluate(0)
        for v in range(70000):
            B[v] = T.float32(1)

    functions = [exprs, shifts, vectors, extents, threads, small]
    return tvm_tirx.IRModule({f.attrs["global_symbol"]: f for f in functions})


@pytest.mark.parametrize("target_bits", [32, 16, 8])
def test_narrow_datatype_branches(target_bits):
    native = tvm_tirx.tirx.transform.NarrowDataType(target_bits)(_narrow_datatype_module())
    rust_pass = tvm_ffi.get_global_func("rust.tirx.transform.NarrowDataType")(target_bits)
    rust = rust_pass(_narrow_datatype_module())
    assert not tvm_ffi.structural_equal(native, _narrow_datatype_module(), map_free_vars=True)
    assert tvm_ffi.structural_equal(native, rust, map_free_vars=True)


def _force_narrow_module():
    """ForceNarrowIndexToInt32 branches: casts of launched threads, scalar parameters, Select and
    if_then_else (with an int64 local scalar on one side), clz, clamped shift amounts (scalar,
    dynamic and vector), Let, bitwise not, narrow tensors, and a symbolic shape."""
    from tvm_tirx.script import tirx as T

    n = T.dynamic("n", "int64")
    x = tvm_tirx.tirx.Var("x", "int64")

    @T.function
    def threads(A: T.Tensor((T.int64(64),), "float32"), B: T.Tensor((T.int64(64),), "float32")):
        T.func_attr({"global_symbol": "threads"})
        bx = T.launch_thread("blockIdx.x", T.int64(2))
        tx = T.launch_thread("threadIdx.x", T.int64(32))
        B[T.Cast("int64", bx) * T.int64(32) + T.Cast("int64", tx)] = A[
            T.Cast("int64", bx) * T.int64(32) + T.Cast("int64", tx)
        ] + T.float32(1)

    @T.function
    def exprs(
        A: T.Tensor((T.int64(16),), "float32"),
        B: T.Tensor((16,), "float32"),
        C: T.Tensor((T.int64(4),), "int32"),
        D: T.Tensor((16,), "int16"),
        m: T.int64,
        s: T.int64,
    ):
        T.func_attr({"global_symbol": "exprs"})
        for i in T.serial(T.int64(4)):
            B[i] = T.Select(T.int64(1) <= i, A[i + m], T.Cast("float32", i))
            B[i + T.int64(4)] = A[T.if_then_else(m < T.int64(0), m + T.int64(1), m)]
            C[i] = T.clz(i)
            B[i + T.int64(8)] = A[
                T.shift_right(T.truncmod(m - T.int64(8), T.int64(6)), T.int64(63))
            ]
            B[T.int64(12)] = A[T.shift_left(T.truncmod(m, T.int64(1)) + i, s)]
            B[T.Ramp(T.int64(1), T.int64(1), 4)] = A[
                T.shift_right(
                    T.Broadcast(T.truncmod(m - T.int64(8), T.int64(6)), 4),
                    T.Ramp(T.int64(30), T.int64(1), 4),
                )
            ]
            B[T.Let(x + T.int64(1), where={x: i})] = A[T.bitwise_not(i) + T.int64(8)]
            D[i] = D[i] + T.int16(1)

    @T.function
    def scalars(A: T.Tensor((T.int64(4),), "float32"), B: T.Tensor((4,), "float32"), m: T.int64):
        T.func_attr({"global_symbol": "scalars"})
        opaque_index: T.int64 = T.call_extern("opaque_index", m, ty="int64")
        B[0] = A[T.if_then_else(m < T.int64(0), opaque_index, m)]
        B[1] = A[T.Select(m < T.int64(0), m, opaque_index)]

    @T.function
    def dynamic(Buf: T.Tensor([n], "int32")):
        T.func_attr({"global_symbol": "dynamic"})
        ceil_log2: T.int64 = T.Cast("int64", T.ceil(T.log2(T.Cast("float32", n))))
        for i in T.serial(ceil_log2):
            Buf[i] = 0

    functions = [threads, exprs, scalars, dynamic]
    return tvm_tirx.IRModule({f.attrs["global_symbol"]: f for f in functions})


def test_force_narrow_index_to_i32_branches():
    native = tvm_tirx.tirx.transform.ForceNarrowIndexToInt32()(_force_narrow_module())
    rust_pass = tvm_ffi.get_global_func("rust.tirx.transform.ForceNarrowIndexToInt32")()
    rust = rust_pass(_force_narrow_module())
    assert not tvm_ffi.structural_equal(native, _force_narrow_module(), map_free_vars=True)
    assert tvm_ffi.structural_equal(native, rust, map_free_vars=True)


def _force_narrow_rejected():
    """An int64 tensor parameter, an int64 tensor allocated in the body, and a constant that
    does not fit in int32."""
    from tvm_tirx.script import tirx as T

    @T.function
    def param(A: T.Tensor((16,), "int64")):
        for i in range(16):
            A[i] = A[i] + T.int64(1)

    @T.function
    def internal(A: T.Tensor((16,), "int32")):
        C = T.alloc_tensor(ty_args=[T.Tensor((16,), "int64")])
        for i in range(16):
            C[i] = T.Cast("int64", A[i]) + T.int64(1)
            A[i] = T.Cast("int32", C[i])

    @T.function
    def large(A: T.Tensor((16,), "int32")):
        A[0] = T.Cast("int32", T.int64(2**40))

    return [tvm_tirx.IRModule({"main": f}) for f in (param, internal, large)]


def test_force_narrow_index_to_i32_rejects_wide_integers():
    for factory in (
        "tirx.transform.ForceNarrowIndexToInt32",
        "rust.tirx.transform.ForceNarrowIndexToInt32",
    ):
        pass_ = tvm_ffi.get_global_func(factory)()
        for module in _force_narrow_rejected():
            with pytest.raises(RuntimeError):
                pass_(module)
