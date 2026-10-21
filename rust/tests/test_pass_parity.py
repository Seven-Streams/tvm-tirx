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

import ast
import contextlib
import ctypes
import importlib.util
import itertools
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


REPO = HERE.parent.parent

# Upstream test suites run again with the C++ pass replaced by a run of both passes
# (`test_upstream_suite_parity`), by the factory they exercise.
SUITES = {
    "StorageRewrite": "tests/python/tirx/transform/test_storage_rewrite.py",
    "PointerValueTypeRewrite": ("tests/python/tirx/transform/test_pointer_value_type_rewrite.py"),
}
_SUITE_MODULES = {}


def _suite(factory):
    """The upstream test module of `factory`, imported once."""
    if factory not in _SUITE_MODULES:
        path = REPO / SUITES[factory]
        spec = importlib.util.spec_from_file_location(f"_parity_suite_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        # TVMScript finds the source of a parsed class through `sys.modules`.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _SUITE_MODULES[factory] = module
    return _SUITE_MODULES[factory]


# Importing the StorageRewrite suite registers the memory scopes `local.L0A` and `local.L0A2`
# (single allocation), which part of the recorded StorageRewrite corpus uses.
_suite("StorageRewrite")


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
    "LowerTVMBuiltin": dict(sources=["tirx.LowerTVMBuiltin", "tirx.MakePackedAPI"]),
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
    "MakePackedAPI": dict(sources=["tirx.MakePackedAPI"]),
    "SplitHostDevice": dict(
        sources=["tirx.SplitHostDevice", "tirx.BindTarget", "tirx.AnnotateEntryFunc", "tirx.Filter"]
    ),
    "FlattenTensor": dict(sources=["tirx.FlattenTensor"]),
    # The pass needs a current target when a function has none; any target gives the same result.
    "LowerTensorLayouts": dict(sources=["tirx.LowerTensorLayouts"], target="cuda"),
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
    # StorageRewrite runs after VectorizeLoop; the passes after it see its results.
    "StorageRewrite": dict(
        sources=[
            "tirx.StorageRewrite",
            "tirx.VectorizeLoop",
            "tirx.HoistIf",
            "tirx.UnrollLoop",
            "tirx.FlattenTensor",
        ]
    ),
    "PointerValueTypeRewrite": dict(
        sources=[
            "tirx.PointerValueTypeRewrite",
            "tirx.StorageRewrite",
            "tirx.HoistIf",
            "tirx.UnrollLoop",
            "tirx.LowerTVMBuiltin",
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


def _lower_tvm_builtin_modules():
    """LowerTVMBuiltin branches the corpus barely reaches: packed calls with every argument kind
    (int, float, string, handle, bool, tensor, a nested packed call), `call_cpacked`, shape stacks
    with and without extents, a parallel loop with its own stacks, a device allocation freed per
    branch and at the end of the device region, the disable annotation, CPU stack and workspace
    allocations, and an OpenCL image allocation."""
    from tvm_tirx.script import tirx as T

    float_ptr = tvm_tirx.ir.PtrType(tvm_tirx.ir.PrimType("float32"))
    image = tvm_tirx.ir.Call(
        "tirx.opencl.nd_mem_alloc_with_scope",
        [tvm_tirx.ir.StringImm("global.texture"), 2, tvm_tirx.tirx.const(16, "int64")],
        ty=float_ptr,
    )

    @T.function
    def gpu(A: T.Tensor((16,), "float32"), n: T.int32, x: T.float32, h: T.handle, flag: T.bool):
        T.func_attr(
            {
                "target": T.target("cuda", host="llvm"),
                "global_symbol": "gpu",
                "tirx.is_host_func": True,
            }
        )
        with T.device_context(2, 0):
            T.evaluate(T.call_packed("f", n, x, "text", h, flag, A, T.call_packed("g", n)))
            T.evaluate(T.call_cpacked("c", n, T.int64(3)))
            T.evaluate(T.call_packed("s", T.stack_make_shape(n, 4), T.stack_make_shape()))
            for i in T.parallel(4):
                T.evaluate(T.call_packed("p", i, A))
            if n > 0:
                B = T.alloc_tensor(ty_args=[T.Tensor((n,), "float32")])
                T.evaluate(T.call_packed("b", B.data))
            else:
                C = T.alloc_tensor(ty_args=[T.Tensor((4,), "float32")])
                T.evaluate(T.call_packed("c", C.data))
            D = T.alloc_tensor(
                annotations={"disable_lower_builtin": 1}, ty_args=[T.Tensor((8,), "float32")]
            )
            E = T.alloc_tensor(ty_args=[T.Tensor((8,), "float32")])
            T.evaluate(T.call_packed("e", D.data, E.data))

    @T.function
    def cpu(n: T.int32):
        T.func_attr({"target": T.target("llvm"), "global_symbol": "cpu"})
        with T.device_context(1, 0):
            small = T.alloc_tensor(ty_args=[T.Tensor((16,), "float32")])
            large = T.alloc_tensor(ty_args=[T.Tensor((1024,), "float32")])
            dynamic = T.alloc_tensor(ty_args=[T.Tensor((n,), "float32")])
            T.evaluate(T.call_packed("f", small.data, large.data, dynamic.data))

    @T.function
    def opencl(n: T.int32):
        T.func_attr(
            {
                "target": T.target("opencl", host="llvm"),
                "global_symbol": "opencl",
                "tirx.is_host_func": True,
            }
        )
        with T.device_context(4, 0):
            p: T.let = image
            T.evaluate(T.call_packed("use", p))

    @T.function
    def no_device(n: T.int32):
        T.func_attr({"target": T.target("cuda", host="llvm"), "tirx.is_host_func": True})
        B = T.alloc_tensor(ty_args=[T.Tensor((n,), "float32")])
        T.evaluate(T.call_packed("b", B.data))

    return {
        name: (lambda func=func: tvm_tirx.IRModule({name: func}))
        for name, func in [("gpu", gpu), ("cpu", cpu), ("opencl", opencl), ("no_device", no_device)]
    }


def test_lower_tvm_builtin_branches():
    native_pass = tvm_tirx.tirx.transform.LowerTVMBuiltin()
    rust_pass = tvm_ffi.get_global_func("rust.tirx.transform.LowerTVMBuiltin")()
    for name, module in _lower_tvm_builtin_modules().items():
        if name == "no_device":
            # A device allocation outside a device region has no device id: both reject it.
            with pytest.raises(Exception) as native_error:
                native_pass(module())
            with pytest.raises(Exception) as rust_error:
                rust_pass(module())
            assert type(native_error.value) is type(rust_error.value), name
            continue
        native = native_pass(module())
        rust = rust_pass(module())
        assert not tvm_ffi.structural_equal(native, module(), map_free_vars=True), name
        assert tvm_ffi.structural_equal(native, rust, map_free_vars=True), (
            name,
            tvm_ffi.get_first_structural_mismatch(native, rust, map_free_vars=True),
        )


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


def _make_packed_api_modules():
    """MakePackedAPI inputs beyond the corpus: every scalar parameter and return kind, symbolic
    shapes bound twice or only through later parameters, explicit strides, element offsets, sub-byte
    and vector dtypes, CUDA/Vulkan/OpenCL devices, subroutine calls, and the skipped functions."""
    from tvm_tirx.script import ir as I
    from tvm_tirx.script import tirx as T

    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int32")
    s = T.dynamic("s", "int32")
    off = T.dynamic("off", "int32")

    @I.ir_module
    class Scalars:
        @T.function
        def ret_int(a: T.int32, u: T.uint8) -> T.int32:
            T.func_attr({"target": T.target("llvm", host="llvm")})
            if a > 0:
                return a
            return T.int32(u)

        @T.function
        def ret_bool(b: T.bool) -> T.bool:
            T.func_attr({"target": T.target("llvm", host="llvm")})
            return b

        @T.function
        def ret_float(x: T.float32, h: T.bfloat16, d: T.float64) -> T.float32:
            T.func_attr({"target": T.target("llvm", host="llvm")})
            return x

        @T.function
        def ret_ptr(p: T.handle) -> T.handle:
            T.func_attr({"target": T.target("llvm", host="llvm")})
            return p

        @T.function
        def typed_ptr(p: T.handle("float32"), q: T.handle("int8")):
            T.func_attr({"target": T.target("llvm", host="llvm")})
            T.evaluate(p)

        @T.function
        def zero_args() -> T.int64:
            T.func_attr({"target": T.target("llvm", host="llvm")})
            for i in T.parallel(1):
                T.evaluate(i)
            return T.int64(42)

    @I.ir_module
    class Symbolic:
        @T.function
        def main(
            A: T.Tensor((n + 1,), "int32"),
            B: T.Tensor((n,), "int32"),
            C: T.Tensor((m, 4), "float32", layout=T.TileLayout(T.S[(m, 4) : (s, 1)])),
            D: T.Tensor((m * 2,), "float16", layout=T.TileLayout(T.S[(m * 2,)] + off)),
            E: T.Tensor((8,), "int4"),
            F: T.Tensor((4,), "float32", align=1),
            k: T.int64,
        ):
            T.func_attr({"target": T.target("llvm", host="llvm")})
            for i in range(n):
                B[i] = A[i] + A[i + 1]

    @I.ir_module
    class Tensors:
        @T.function
        def main(
            A: T.Tensor((8,), "int1"),
            B: T.Tensor((8,), "uint4"),
            C: T.Tensor((), "float32"),
            D: T.Tensor((4,), "float32", layout=T.TileLayout(T.S[(4,)] + 4)),
            E: T.Tensor((4,), "bool"),
            F: T.Tensor((4,), "float32x4"),
            G: T.Tensor((off,), "float16", layout=T.TileLayout(T.S[(off,)] + off * 2)),
            k: T.int64,
            h: T.float16,
        ):
            T.func_attr({"target": T.target("opencl", host="llvm"), "calling_conv": 0})
            T.evaluate(0)

    @I.ir_module
    class Cuda:
        @T.function
        def main(A: T.Tensor((m,), "float32"), B: T.Tensor((m,), "float32")):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            T.evaluate(0)

    @I.ir_module
    class Vulkan:
        @T.function
        def main(A: T.Tensor((m,), "float32"), B: T.Tensor((16,), "float32")):
            T.func_attr({"target": T.target("vulkan", host="llvm")})
            T.evaluate(0)

    @I.ir_module
    class Calls:
        @T.function
        def main(A: T.handle("float32")):
            T.func_attr({"target": T.target("llvm", host="llvm")})
            Calls.subroutine(A)
            Calls.internal(A)

        @T.function
        def subroutine(A_data: T.handle("float32")):
            T.func_attr({"target": T.target("llvm", host="llvm")})
            T.evaluate(A_data)

        @T.function(private=True)
        def internal(A_data: T.handle("float32")):
            T.func_attr({"target": T.target("llvm")})
            T.evaluate(A_data)

        @T.function
        def lowered(A_data: T.handle("float32")):
            T.func_attr({"target": T.target("llvm", host="llvm"), "calling_conv": 2})
            T.evaluate(A_data)

        @T.function
        def no_host(A_data: T.handle("float32")):
            T.func_attr({"target": T.target("llvm")})
            T.evaluate(A_data)

    x = tvm_tirx.tirx.Var("x", "int32")
    void_return = tvm_tirx.tirx.Function(
        [x], tvm_tirx.ir.Return(tvm_tirx.tirx.call_extern("f", x, ty="void"))
    )
    void_return = void_return.with_attr(
        {"target": tvm_tirx.target.Target("llvm", host="llvm"), "global_symbol": "void_return"}
    )
    return [
        Scalars,
        Symbolic,
        Tensors,
        Cuda,
        Vulkan,
        Calls,
        tvm_tirx.IRModule({"main": void_return}),
    ]


def test_make_packed_api_branches():
    native = tvm_tirx.tirx.transform.MakePackedAPI()
    rust = tvm_ffi.get_global_func("rust.tirx.transform.MakePackedAPI")()
    for before, expected, actual in zip(
        _make_packed_api_modules(),
        map(native, _make_packed_api_modules()),
        map(rust, _make_packed_api_modules()),
    ):
        assert not tvm_ffi.structural_equal(before, expected, map_free_vars=True)
        assert tvm_ffi.structural_equal(expected, actual, map_free_vars=True)


def _make_packed_api_rejected():
    """MakePackedAPI inputs both sides reject, with the message each must carry."""
    from tvm_tirx.script import ir as I
    from tvm_tirx.script import tirx as T

    @I.ir_module
    class NoTarget:
        @T.function
        def main(A: T.handle("float32")):
            T.evaluate(A)

    @I.ir_module
    class ParallelReturn:
        @T.function
        def main() -> T.int32:
            T.func_attr({"target": T.target("llvm", host="llvm")})
            for i in T.parallel(2):
                return i

    @I.ir_module
    class Bfloat16Return:
        @T.function
        def main(x: T.bfloat16) -> T.bfloat16:
            T.func_attr({"target": T.target("llvm", host="llvm")})
            return x

    @I.ir_module
    class Float8Param:
        @T.function
        def main(x: T.float8_e4m3fn):
            T.func_attr({"target": T.target("llvm", host="llvm")})
            T.evaluate(x)

    x, y = tvm_tirx.tirx.Var("x", "int32"), tvm_tirx.tirx.Var("y", "int32")
    undefined = tvm_tirx.tirx.Function([x], tvm_tirx.ir.Evaluate(x + y)).with_attr(
        {"target": tvm_tirx.target.Target("llvm", host="llvm"), "global_symbol": "main"}
    )
    return [
        (NoTarget, "required the function to be annotated with tvm::attr::kTarget"),
        (ParallelReturn, "Return cannot be used in parallel scope"),
        (Bfloat16Return, "data type bfloat16 not supported yet"),
        (Float8Param, "kDLFloat, DLDataTypeCode::kDLBfloat"),
        (tvm_tirx.IRModule({"main": undefined}), r"variables \(y,\) are used, but are not passed"),
    ]


def test_make_packed_api_rejects():
    for factory in ("tirx.transform.MakePackedAPI", "rust.tirx.transform.MakePackedAPI"):
        pass_ = tvm_ffi.get_global_func(factory)()
        for mod, message in _make_packed_api_rejected():
            with pytest.raises(RuntimeError, match=message):
                pass_(mod)


def _compare_on(native, rust, mod, compared):
    """Run both passes on separate copies of `mod`: the native one may update its input."""
    text = tvm_tirx.ir.save_json(mod)
    outcome = []
    for pass_ in (native, rust):
        try:
            outcome.append(("ok", pass_(tvm_tirx.ir.load_json(text))))
        except Exception as error:  # pylint: disable=broad-except
            outcome.append(("error", error))
    (native_kind, expected), (rust_kind, actual) = outcome
    if native_kind == rust_kind == "error":
        agree = type(expected) is type(actual)
    elif native_kind != rust_kind:
        agree = False
    else:
        agree = tvm_ffi.structural_equal(expected, actual, map_free_vars=True)
    compared.append((agree, native_kind, mod, expected, actual))
    return native_kind, expected


def _pipeline_comparing_factory(name, compared, stop=False):
    """A stand-in for `tvm_tirx.tirx.transform.<name>` that runs the native pass and the Rust one on
    each input and continues with the native result, or stops the pipeline after comparing."""
    native_factory = getattr(tvm_tirx.tirx.transform, name)
    rust_factory = tvm_ffi.get_global_func(f"rust.tirx.transform.{name}")

    def factory(*args):
        native, rust = native_factory(*args), rust_factory(*args)

        def apply(mod, _ctx):
            kind, result = _compare_on(native, rust, mod, compared)
            if stop:
                raise _StopPipeline()
            if kind == "error":
                raise result
            return result

        return tvm_tirx.transform.module_pass(apply, opt_level=0, name=f"tirx.{name}")

    return factory


class _StopPipeline(Exception):
    pass


def _assert_all_agree(compared, minimum):
    failures = []
    for agree, kind, _, expected, actual in compared:
        if agree:
            continue
        if kind == "ok" and not isinstance(actual, Exception):
            detail = tvm_ffi.get_first_structural_mismatch(expected, actual, map_free_vars=True)
        else:
            detail = (expected, actual)
        failures.append(f"native {kind}: {detail!s:.600}")
    assert not failures, "\n".join(failures)
    assert len(compared) >= minimum, f"only {len(compared)} inputs compared"


def _load_upstream(relative):
    import importlib.util  # pylint: disable=import-outside-toplevel

    path = HERE.parent.parent / relative
    spec = importlib.util.spec_from_file_location(f"upstream_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    # TVMScript finds the source of a class through its module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_split_host_device_upstream_inputs(monkeypatch):
    """Every input of the upstream SplitHostDevice tests, which still pass on the native results."""
    compared = []
    monkeypatch.setattr(
        tvm_tirx.tirx.transform,
        "SplitHostDevice",
        _pipeline_comparing_factory("SplitHostDevice", compared),
    )
    tests = [
        ("tests/python/tirx/transform/test_split_host_device.py", None),
        (
            "tests/python/tirx/test_cuda_launch.py",
            ["test_launch_region_roundtrip_and_host_only_operands"],
        ),
        (
            "tests/python/tirx/test_cuda_backend_config.py",
            ["test_device_helpers_are_cloned_per_configuration"],
        ),
    ]
    for path, names in tests:
        module = _load_upstream(path)
        for name in names or [n for n in vars(module) if n.startswith("test_")]:
            getattr(module, name)()
    _assert_all_agree(compared, 20)


def _split_host_device_kernels():
    """Device kernels whose SplitHostDevice input the compile pipeline produces."""
    from tvm_tirx.script import tirx as T  # pylint: disable=import-outside-toplevel

    @T.function
    def dynamic_bytes(A: T.Tensor((1,), "int32"), n: T.int32, bytes_: T.int32):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32, dynamic_smem_bytes=bytes_))
        scratch = T.alloc_tensor(ty_args=[T.Tensor((n,), "int32", scope="shared.dyn")])
        if T.cuda.thread_idx("x") == 0:
            scratch[n - 1] = n
            A[0] = scratch[n - 1]

    @T.function
    def too_few_bytes():
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32, dynamic_smem_bytes=64))
        scratch = T.alloc_tensor(ty_args=[T.Tensor((128,), "int32", scope="shared.dyn")])
        scratch[0] = T.cuda.thread_idx("x")

    @T.function
    def enough_bytes():
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32, dynamic_smem_bytes=1024))
        scratch = T.alloc_tensor(ty_args=[T.Tensor((128,), "int32", scope="shared.dyn")])
        scratch[0] = T.cuda.thread_idx("x")

    @T.function
    def undeclared_bytes(A: T.Tensor((64,), "float32"), n: T.int32):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(n, 2), block=(32, 2)))
        scratch = T.alloc_tensor(ty_args=[T.Tensor((n, 4), "float32", scope="shared.dyn")])
        scratch[0, 0] = A[T.cuda.thread_idx("x")]
        A[0] = scratch[0, 0]

    return [dynamic_bytes, too_few_bytes, enough_bytes, undeclared_bytes]


def test_split_host_device_cuda_launches(monkeypatch):
    """CUDA launches sizing their dynamic shared memory, as the compile pipeline hands them over."""
    compared = []
    monkeypatch.setattr(
        tvm_tirx.tirx.transform,
        "SplitHostDevice",
        _pipeline_comparing_factory("SplitHostDevice", compared, stop=True),
    )
    target = {"kind": "cuda", "backend_config": {"cuda": {"arch": "sm_80"}}}
    for kernel in _split_host_device_kernels():
        with pytest.raises(Exception):
            tvm_tirx.tirx.compile(kernel, target=target)
    _assert_all_agree(compared, 4)


def _split_host_device_modules():
    from tvm_tirx.script import ir as I  # pylint: disable=import-outside-toplevel
    from tvm_tirx.script import tirx as T  # pylint: disable=import-outside-toplevel

    @I.ir_module
    class Trainium:
        @T.function
        def main(B: T.Tensor((16,), "float32"), A: T.Tensor((16,), "float32")):
            T.func_attr({"target": T.target("trn", host="llvm")})
            T.region("tirx.device_scope", [], attrs={"target": T.target("trn")})
            A[0] = B[0]

    @I.ir_module
    class UntargetedScope:
        @T.function
        def main(A: T.Tensor((16,), "float32")):
            T.func_attr({"target": T.target("llvm")})
            T.region("tirx.device_scope", [])
            A[0] = T.float32(1)

    @I.ir_module
    class MetalReturn:
        @T.function
        def main() -> T.int32:
            T.func_attr({"global_symbol": "main", "target": T.target("metal", host="llvm")})
            T.region("tirx.device_scope", [], attrs={"target": T.target("metal")})
            T.launch_thread("threadIdx.x", 16)
            return 0

    @I.ir_module
    class CudaReturn:
        @T.function
        def main() -> T.int32:
            T.func_attr({"global_symbol": "main", "target": T.target("cuda", host="llvm")})
            T.region("tirx.device_scope", [], attrs={"target": T.target("cuda")})
            T.launch_thread("threadIdx.x", 16)
            return 0

    @I.ir_module
    class LegacySharedDyn:
        @T.function
        def main(A: T.Tensor((16,), "float32"), n: T.int32):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            T.region("tirx.device_scope", [], attrs={"target": T.target("cuda")})
            tx = T.launch_thread("threadIdx.x", 16)
            m: T.int32 = n * 2
            scratch = T.alloc_tensor(ty_args=[T.Tensor((m,), "float32", scope="shared.dyn")])
            scratch[tx] = A[tx]
            A[tx] = scratch[tx]

    @I.ir_module
    class Placeholder:
        @T.function
        def main(A: T.Tensor((16,), "float32")):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            T.region("tirx.device_scope", [], attrs={"target": T.target("cuda")})
            tx = T.launch_thread("threadIdx.x", 16)
            scratch = T.alloc_tensor(ty_args=[T.Tensor((0,), "float32", scope="shared.dyn")])
            A[tx] = scratch[tx]

    @I.ir_module
    class TwoSharedDyn:
        @T.function
        def main(A: T.Tensor((16,), "float32")):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            T.region("tirx.device_scope", [], attrs={"target": T.target("cuda")})
            tx = T.launch_thread("threadIdx.x", 16)
            first = T.alloc_tensor(ty_args=[T.Tensor((16,), "float32", scope="shared.dyn")])
            second = T.alloc_tensor(ty_args=[T.Tensor((16,), "float32", scope="shared.dyn")])
            A[tx] = first[tx] + second[tx]

    @I.ir_module
    class IncompatibleExtents:
        @T.function
        def main(A: T.Tensor((64,), "float32")):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            T.region("tirx.device_scope", [], attrs={"target": T.target("cuda")})
            tx = T.launch_thread("threadIdx.x", 16)
            with T.launch_thread("threadIdx.x", 32) as ty:
                A[tx + ty] = T.float32(1)

    def pool(launch_fields, values, required, alloc):
        @I.ir_module
        class Pool:
            @T.function
            def main(A: T.Tensor((16,), "int32")):
                T.func_attr({"target": T.target("cuda", host="llvm")})
                with T.region(
                    "tirx.device_scope",
                    values,
                    attrs={
                        "target": T.target("cuda"),
                        "cuda.launch_fields": launch_fields,
                        "cuda.kernel_attrs": {},
                        "cuda.smem_required": required,
                    },
                ):
                    if alloc:
                        scratch = T.alloc_tensor(
                            ty_args=[T.Tensor((16,), "int32", scope="shared.dyn")]
                        )
                        scratch[0] = A[0]
                        A[1] = scratch[0]
                    else:
                        A[1] = A[0]

        return Pool

    grid = ["grid.x", "grid.y", "grid.z", "block.x", "block.y", "block.z"]
    pools = [
        pool(grid, [1, 1, 1, 32, 1, 1], 256, True),
        pool(grid, [1, 1, 1, 32, 1, 1], 16, True),
        pool(grid, [1, 1, 1, 32, 1, 1], 128, False),
        pool([*grid, "dynamic_smem_bytes"], [1, 1, 1, 32, 1, 1, 32], 128, False),
        pool([*grid, "dynamic_smem_bytes"], [1, 1, 1, 32, 1, 1, 512], 128, True),
        pool([*grid, "dynamic_smem_bytes"], [1, 1, 1, 32, 1, 1, 0], 0, False),
    ]
    return [
        Trainium,
        UntargetedScope,
        MetalReturn,
        CudaReturn,
        LegacySharedDyn,
        Placeholder,
        TwoSharedDyn,
        IncompatibleExtents,
        *pools,
    ]


def test_split_host_device_branches():
    """Trainium parameter order, untargeted scopes, kernel returns, legacy dynamic shared memory,
    shared memory pools and their errors."""
    native = tvm_tirx.tirx.transform.SplitHostDevice()
    rust = tvm_ffi.get_global_func("rust.tirx.transform.SplitHostDevice")()
    compared = []
    for mod in _split_host_device_modules():
        _compare_on(native, rust, mod, compared)
    _assert_all_agree(compared, 14)
    changed = sum(
        kind == "ok" and not tvm_ffi.structural_equal(mod, expected, map_free_vars=True)
        for _, kind, mod, expected, _ in compared
    )
    failed = sum(kind == "error" for _, kind, *_ in compared)
    assert changed >= 8 and failed >= 3, (changed, failed)


def _flatten_tensor_module():
    """FlattenTensor rewrites the corpus does not reach: masked loads and stores through plain and
    strided parameters, a local tensor, and a view whose layout iterators load a local tensor."""
    from tvm_tirx.script import tirx as T

    tile = T.TileLayout.from_iters

    @T.function
    def main(
        A: T.Tensor((4, 8), "float32"),
        B: T.Tensor((4, 8), "float32", layout=T.TileLayout(T.S[(4, 8) : (16, 1)])),
        k: T.int32,
    ):
        T.func_attr({"global_symbol": "main"})
        mask = T.meta_var(T.Broadcast(T.bool(True), 4))
        n = T.alloc_local([1], "int32")
        n[0] = 4
        data = T.alloc_shared([64], "float32")
        view = T.decl_tensor(
            data.data,
            ty_args=[
                T.Tensor(
                    (n[0], 8),
                    "float32",
                    scope="shared",
                    layout=tile(shard=[T.Iter(n[0], 8, "m"), T.Iter(8, 1, "m")], replica=[]),
                )
            ],
        )
        view[1, k] = A[1, 2]
        loaded = T.meta_var(T.masked_load(A, k, T.Ramp(0, 1, 4), mask, ty="float32x4"))
        T.masked_store(B, loaded, 2, T.Ramp(4, 1, 4), mask)
        T.masked_store(
            data,
            T.masked_load(view, 1, T.Ramp(0, 1, 4), mask, ty="float32x4"),
            T.Ramp(k, 1, 4),
            mask,
        )

    return tvm_tirx.IRModule({"main": main})


def test_flatten_tensor_masked_accesses():
    native = tvm_tirx.tirx.transform.FlattenTensor()(_flatten_tensor_module())
    rust = tvm_ffi.get_global_func("rust.tirx.transform.FlattenTensor")()(_flatten_tensor_module())
    assert not tvm_ffi.structural_equal(native, _flatten_tensor_module(), map_free_vars=True)
    assert tvm_ffi.structural_equal(native, rust, map_free_vars=True)


def _lower_tensor_layouts_module():
    """LowerTensorLayouts rewrites the corpus does not reach: a tensor bound to a parameter with a
    layout, allocations spanning a tile layout's replica and offset and an unaligned swizzle over
    a gapped tile, and a declaration flattening leaves alone."""
    from tvm_tirx.script import tirx as T

    tile = T.TileLayout.from_iters

    @T.function
    def main(
        A: T.Tensor(
            (8, 16), "float32", layout=tile(shard=[T.Iter(8, 32, "m"), T.Iter(16, 1, "m")])
        ),
        B: T.Tensor((64,), "float32"),
        n: T.int32,
    ):
        T.func_attr({"target": T.target("cuda"), "global_symbol": "main"})
        V: T.let[
            T.Tensor(
                (8, 16), "float32", layout=tile(shard=[T.Iter(8, 32, "m"), T.Iter(16, 1, "m")])
            )
        ] = A
        S = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (4, 4),
                    "float32",
                    scope="shared",
                    layout=tile(
                        shard=[T.Iter(4, 8, "m"), T.Iter(4, 1, "m")],
                        replica=[T.Iter(2, 64, "m")],
                        offset={"m": 3},
                    ),
                )
            ]
        )
        W = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (8, 64),
                    "float16",
                    scope="shared",
                    layout=T.ComposeLayout(
                        3,
                        3,
                        3,
                        tile(shard=[T.Iter(8, 128, "m"), T.Iter(64, 1, "m")]),
                        swizzle_inner=False,
                    ),
                )
            ]
        )
        ptr: T.let[T.Ptr(T.float32, "shared")] = T.reinterpret(B[0], ty=T.Ptr(T.float32, "shared"))
        D = T.decl_tensor(ptr, ty_args=[T.Tensor((64,), "float32", scope="shared", layout=None)])
        S[1, 2] = A[n, 3]
        V[n, 2] = T.float32(1)
        W[2, 5] = T.float16(1)
        D[n] = B[n] + S[3, 3]
        T.evaluate(T.call_extern("use", V.data, ty="int32"))

    return tvm_tirx.IRModule({"main": main})


def test_lower_tensor_layouts_layouts():
    native = tvm_ffi.get_global_func("tirx.transform.LowerTensorLayouts")()(
        _lower_tensor_layouts_module()
    )
    rust = tvm_ffi.get_global_func("rust.tirx.transform.LowerTensorLayouts")()(
        _lower_tensor_layouts_module()
    )
    assert not tvm_ffi.structural_equal(native, _lower_tensor_layouts_module(), map_free_vars=True)
    assert tvm_ffi.structural_equal(native, rust, map_free_vars=True)


def _rejected_tensor_modules():
    """A direct access through a thread-axis layout, a projection of an undefined tensor, and a
    DeclTensor alias over one."""
    from tvm_tirx.script import tirx as T

    @T.function
    def thread_axis(n: T.int32):
        T.func_attr({"target": T.target("cuda"), "global_symbol": "thread_axis"})
        S = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (32, 4),
                    "float32",
                    scope="local",
                    layout=T.TileLayout.from_iters(
                        shard=[T.Iter(32, 1, "laneid"), T.Iter(4, 1, "m")]
                    ),
                )
            ]
        )
        S[0, n] = T.float32(1)

    undefined = tvm_tirx.tirx.Var("W", tvm_tirx.tirx.TensorType((4,), "float32"))
    target = tvm_tirx.target.Target("cuda")

    def function(stmt):
        func = tvm_tirx.tirx.Function([], stmt)
        return tvm_tirx.IRModule(
            {"main": func.with_attr({"target": target, "global_symbol": "main"})}
        )

    projection = function(
        tvm_tirx.ir.Evaluate(tvm_tirx.tirx.call_extern("use", undefined.data, ty="int32"))
    )
    alias = tvm_tirx.tirx.Var("V", tvm_tirx.tirx.TensorType((4,), "float32"))
    declaration = tvm_tirx.ir.Call(
        tvm_tirx.ir.Op.get("tirx.decl_tensor"), [undefined.data], ty=alias.ty, ty_args=[alias.ty]
    )
    aliased = function(
        tvm_tirx.ir.SeqStmt(
            [
                tvm_tirx.ir.Bind(alias, declaration),
                tvm_tirx.ir.Evaluate(tvm_tirx.tirx.call_extern("use", alias.data, ty="int32")),
            ]
        )
    )
    return tvm_tirx.IRModule({"main": thread_axis}), projection, aliased


@pytest.mark.parametrize("name", ["FlattenTensor", "LowerTensorLayouts"])
def test_rejected_tensor_accesses(name):
    thread_axis, projection, aliased = _rejected_tensor_modules()
    if name == "FlattenTensor":
        cases = [
            (projection, "Tensor W is used before its definition"),
            (aliased, "Tensor W is used before its definition"),
        ]
    else:
        cases = [
            (
                thread_axis,
                "Cannot lower direct TensorLoad/TensorStore on a tensor with thread-axis",
            ),
            (projection, "tensor_data_ptr projects W, which has no visible definition"),
            (aliased, "Tensor alias source W must be registered before its DeclTensor alias"),
        ]
    for prefix in ("", "rust."):
        pass_ = tvm_ffi.get_global_func(f"{prefix}tirx.transform.{name}")()
        for module, message in cases:
            with pytest.raises(Exception, match=message):
                pass_(module)


def test_lower_tensor_layouts_requires_a_target():
    from tvm_tirx.script import tirx as T

    @T.function
    def main(A: T.Tensor((4, 4), "float32")):
        A[1, 2] = T.float32(1)

    for prefix in ("", "rust."):
        pass_ = tvm_ffi.get_global_func(f"{prefix}tirx.transform.LowerTensorLayouts")()
        with pytest.raises(Exception, match="Target context required"):
            pass_(tvm_tirx.IRModule({"main": main}))
        with tvm_tirx.target.Target("cuda"):
            pass_(tvm_tirx.IRModule({"main": main}))


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


def _storage_rewrite_threads():
    """Thread regions: the hardware axes share the outer one's allocation scope, where shared
    storage is reused; a vthread opens a scope of its own."""
    from tvm_tirx.script import tirx as T

    @T.function
    def main(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
        T.func_attr({"target": T.target("cuda"), "global_symbol": "main"})
        with T.launch_thread("blockIdx.x", 2) as bx:
            with T.launch_thread("threadIdx.x", 32) as tx:
                S = T.alloc_tensor(ty_args=[T.Tensor((32,), "float32", scope="shared")])
                S[tx] = A[bx * 32 + tx]
                R = T.alloc_tensor(ty_args=[T.Tensor((32,), "float32", scope="shared")])
                R[tx] = S[tx] + T.float32(1)
                B[bx * 32 + tx] = R[tx]
                Q = T.alloc_tensor(ty_args=[T.Tensor((32,), "float32", scope="shared")])
                Q[tx] = B[tx] * T.float32(2)
                A[tx] = Q[tx]
        with T.launch_thread("vthread", 2) as vt:
            V = T.alloc_tensor(ty_args=[T.Tensor((16,), "float32")])
            for i in range(16):
                V[i] = A[vt * 16 + i]
            W = T.alloc_tensor(ty_args=[T.Tensor((16,), "float32")])
            for i in range(16):
                W[i] = V[i] * T.float32(3)
            for i in range(16):
                B[vt * 16 + i] = W[i]

    return tvm_tirx.IRModule({"main": main})


def _storage_rewrite_inplace():
    """In-place updates: `Y[i] = f(X[i])` writes over X; an indirect load, a shifted read, a
    reduction, and a volatile allocation do not."""
    from tvm_tirx.script import tirx as T

    @T.function
    def main(
        A: T.Tensor((64,), "float32"), Idx: T.Tensor((64,), "int32"), B: T.Tensor((64,), "float32")
    ):
        T.func_attr({"global_symbol": "main"})
        X = T.alloc_tensor(ty_args=[T.Tensor((64,), "float32")])
        for i in range(64):
            X[i] = A[i]
        Y = T.alloc_tensor(ty_args=[T.Tensor((64,), "float32")])
        for i in range(64):
            Y[i] = X[i] + T.float32(1)
        Z = T.alloc_tensor(ty_args=[T.Tensor((64,), "float32")])
        for i in range(64):
            Z[i] = Y[Idx[i]] * T.float32(2)
        U = T.alloc_tensor(ty_args=[T.Tensor((64,), "float32")])
        for i in range(63):
            U[i] = Z[i + 1]
        V = T.alloc_tensor(ty_args=[T.Tensor((64,), "float32")])
        for i in range(64):
            V[i] = U[i]
            V[i] = V[i] + U[i]
        W = T.alloc_tensor(ty_args=[T.Tensor((64,), "float32")])
        for i in range(64):
            if i < 32:
                W[i] = V[i] - T.float32(1)
        for i in range(64):
            B[i] = W[i]
        P = T.alloc_tensor(ty_args=[T.Tensor((64,), "float32")])
        P[0] = B[0]
        Q = T.alloc_tensor(
            ty_args=[T.Tensor((64,), "float32")], annotations={"tirx.volatile": True}
        )
        Q[0] = P[0]
        B[1] = Q[0]

    return tvm_tirx.IRModule({"main": main})


def _storage_rewrite_reuse():
    """Reuse: symbolic sizes round robin, a larger buffer takes a smaller freed one (a merged
    size), a large extent widens to int64, and branches and asserts as scopes."""
    from tvm_tirx.script import tirx as T

    @T.function
    def main(n: T.int32, A: T.Tensor((1024,), "float32"), B: T.Tensor((1024,), "float32")):
        T.func_attr({"global_symbol": "main"})
        S1 = T.alloc_tensor(ty_args=[T.Tensor((n,), "float32")])
        for i in range(n):
            S1[i] = A[i]
        B[0] = S1[0]
        S2 = T.alloc_tensor(ty_args=[T.Tensor((n,), "float32")])
        for i in range(n):
            S2[i] = A[i] + T.float32(1)
        B[1] = S2[0]
        X = T.alloc_tensor(ty_args=[T.Tensor((256,), "float32")])
        for i in range(256):
            X[i] = A[i]
        B[2] = X[255]
        Y = T.alloc_tensor(ty_args=[T.Tensor((1024,), "float32")])
        for i in range(1024):
            Y[i] = A[i] * T.float32(2)
        B[3] = Y[1023]
        L = T.alloc_tensor(ty_args=[T.Tensor((16777216,), "float32")])
        L[0] = A[0]
        B[4] = L[0]
        M = T.alloc_tensor(ty_args=[T.Tensor((268435456,), "float32")])
        M[0] = A[1]
        B[5] = M[0]
        if n > 4:
            C = T.alloc_tensor(ty_args=[T.Tensor((128,), "float32")])
            C[0] = A[0]
            B[6] = C[0]
        else:
            D = T.alloc_tensor(ty_args=[T.Tensor((128,), "float32")])
            D[0] = A[1]
            B[6] = D[0]
        assert n > 0, "positive"
        E = T.alloc_tensor(ty_args=[T.Tensor((128,), "float32")])
        E[0] = A[2]
        B[7] = E[0]

    return tvm_tirx.IRModule({"main": main})


def _storage_rewrite_dtypes(target):
    """Reuse across dtypes (Vulkan and WebGPU reuse only identical ones), and pointers into
    packed single-allocation storage (`local.L0A`, registered by the upstream suite)."""
    from tvm_tirx.script import tirx as T

    @T.function
    def main(A: T.Tensor((256,), "float32"), B: T.Tensor((256,), "int32")):
        T.func_attr({"target": T.target(target), "global_symbol": "main"})
        X = T.alloc_tensor(ty_args=[T.Tensor((256,), "int32")])
        for i in range(256):
            X[i] = B[i]
        B[0] = X[255]
        Y = T.alloc_tensor(ty_args=[T.Tensor((256,), "float32")])
        for i in range(256):
            Y[i] = A[i]
        A[0] = Y[255]
        P = T.alloc_tensor(ty_args=[T.Tensor((64,), "float16", "local.L0A")])
        Q = T.alloc_tensor(ty_args=[T.Tensor((64,), "int32", "local.L0A")])
        for i in range(64):
            P[i] = T.Cast("float16", A[i])
            Q[i] = B[i]
        T.evaluate(T.call_extern("consume", P.data, Q.data, ty="int32"))
        A[2] = T.Cast("float32", P[0]) + T.Cast("float32", Q[0])

    return tvm_tirx.IRModule({"main": main})


def _storage_rewrite_vector_merge():
    """A vector allocation reusing a scalar one is sized by the widest element type; the scalar
    accesses then fail the PointerValueTypeRewrite that StorageRewrite ends with."""
    from tvm_tirx.script import tirx as T

    @T.function
    def main(A: T.Tensor((256,), "float32")):
        T.func_attr({"global_symbol": "main"})
        Y = T.alloc_tensor(ty_args=[T.Tensor((256,), "float32")])
        for i in range(256):
            Y[i] = A[i]
        A[0] = Y[255]
        V = T.alloc_tensor(ty_args=[T.Tensor((64,), "float32x4")])
        for i in range(64):
            V[i] = A[T.Ramp(i * 4, 1, 4)]
        A[T.Ramp(0, 1, 4)] = V[63]

    return tvm_tirx.IRModule({"main": main})


def _storage_rewrite_masked():
    """Masked accesses to planned and to retyped tensors."""
    from tvm_tirx.script import tirx as T

    @T.function
    def main(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
        T.func_attr({"global_symbol": "main"})
        mask = T.Broadcast(T.bool(True), 4)
        X = T.alloc_tensor(ty_args=[T.Tensor((16,), "float32")])
        for i in range(4):
            T.evaluate(
                T.call_intrin(
                    "tirx.masked_store",
                    X,
                    T.call_intrin("tirx.masked_load", A, T.Ramp(i * 4, 1, 4), mask, ty="float32x4"),
                    T.Ramp(i * 4, 1, 4),
                    mask,
                    ty="void",
                )
            )
        B[0] = X[0]
        Y = T.alloc_tensor(ty_args=[T.Tensor((16,), "float32")])
        for i in range(4):
            Y[T.Ramp(i * 4, 1, 4)] = T.call_intrin(
                "tirx.masked_load", A, T.Ramp(i * 4, 1, 4), mask, ty="float32x4"
            )
        for i in range(4):
            T.evaluate(
                T.call_intrin(
                    "tirx.masked_store",
                    B,
                    T.call_intrin("tirx.masked_load", Y, T.Ramp(i * 4, 1, 4), mask, ty="float32x4"),
                    T.Ramp(i * 4, 1, 4),
                    mask,
                    ty="void",
                )
            )

    return tvm_tirx.IRModule({"main": main})


def _storage_rewrite_vectors():
    """Retyping: an allocation accessed as float32x4 whose data pointer is bound to a scalar
    pointer and passed on, a declaration over a pointer parameter, and an untyped pointer."""
    from tvm_tirx.script import tirx as T

    @T.function
    def main(
        A: T.Tensor((32,), "float32"),
        Bptr: T.handle("float32"),
        C: T.Tensor((32,), "float32"),
        D: T.handle(),
    ):
        T.func_attr({"global_symbol": "main"})
        B = T.decl_tensor(Bptr, ty_args=[T.Tensor((32,), "float32")])
        X = T.alloc_tensor(ty_args=[T.Tensor((32,), "float32")])
        for i in range(8):
            X[T.Ramp(i * 4, 1, 4)] = A[T.Ramp(i * 4, 1, 4)]
        for i in range(8):
            B[T.Ramp(i * 4, 1, 4)] = X[T.Ramp(i * 4, 1, 4)]
        p: T.let = X.data
        T.evaluate(T.call_extern("consume", p, X.data, D, ty="int32"))
        for i in range(8):
            C[T.Ramp(i * 4, 1, 4)] = A[T.Ramp(i * 4, 1, 4)] + B[T.Ramp(i * 4, 1, 4)]

    return tvm_tirx.IRModule({"main": main})


def _storage_rewrite_two_dimensional():
    """A two-dimensional allocation cannot be planned: both sides reject it."""
    from tvm_tirx.script import tirx as T

    @T.function
    def main(A: T.Tensor((16,), "float32")):
        T.func_attr({"global_symbol": "main"})
        X = T.alloc_tensor(ty_args=[T.Tensor((4, 4), "float32")])
        for i in range(4):
            X[i, 0] = A[i]
        A[0] = X[3, 0]

    return tvm_tirx.IRModule({"main": main})


_STORAGE_REWRITE_MODULES = {
    "threads": _storage_rewrite_threads,
    "inplace": _storage_rewrite_inplace,
    "reuse": _storage_rewrite_reuse,
    "dtypes[llvm]": lambda: _storage_rewrite_dtypes("llvm"),
    "dtypes[vulkan]": lambda: _storage_rewrite_dtypes("vulkan"),
    "dtypes[webgpu]": lambda: _storage_rewrite_dtypes("webgpu"),
    "vector_merge": _storage_rewrite_vector_merge,
    "masked": _storage_rewrite_masked,
    "vectors": _storage_rewrite_vectors,
    "two_dimensional": _storage_rewrite_two_dimensional,
}


@pytest.mark.parametrize("name", list(_STORAGE_REWRITE_MODULES))
@pytest.mark.parametrize(
    "factory,config",
    [
        ("StorageRewrite", {}),
        # MergeSharedMemoryAllocations takes over reuse.
        ("StorageRewrite", {"tirx.merge_static_smem": True}),
        ("PointerValueTypeRewrite", {}),
    ],
    ids=["StorageRewrite", "StorageRewrite[merge_static_smem]", "PointerValueTypeRewrite"],
)
def test_storage_rewrite_branches(name, factory, config):
    outcome = []
    for prefix in ("", "rust."):
        pass_ = tvm_ffi.get_global_func(f"{prefix}tirx.transform.{factory}")()
        try:
            with tvm_tirx.transform.PassContext(config=config):
                outcome.append(("ok", pass_(_STORAGE_REWRITE_MODULES[name]())))
        except Exception as error:
            outcome.append(("error", error))
    (native_kind, expected), (rust_kind, actual) = outcome
    assert native_kind == rust_kind, f"native {native_kind}, rust {rust_kind}: {expected or actual}"
    if native_kind == "ok":
        assert tvm_ffi.structural_equal(expected, actual, map_free_vars=True), (
            tvm_ffi.get_first_structural_mismatch(expected, actual, map_free_vars=True)
        )
    if factory == "StorageRewrite" and native_kind == "ok":
        # Every module has allocations to plan.
        original = _STORAGE_REWRITE_MODULES[name]()
        assert not tvm_ffi.structural_equal(original, expected, map_free_vars=True)


def _parametrizations(test):
    """Keyword arguments of every case of a test's `pytest.mark.parametrize` marks."""
    axes = []
    for mark in getattr(test, "pytestmark", []):
        if mark.name != "parametrize":
            continue
        names, values = mark.args[0], mark.args[1]
        if isinstance(names, str):
            names = [name.strip() for name in names.split(",")]
        axes.append([dict(zip(names, value if len(names) > 1 else (value,))) for value in values])
    for combination in itertools.product(*axes):
        yield {key: value for case in combination for key, value in case.items()}


def _upstream_tests():
    for factory, path in SUITES.items():
        for node in ast.parse((REPO / path).read_text()).body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                yield pytest.param(factory, node.name, id=f"{factory}::{node.name}")


def _dual_run(factory, runs):
    """A factory for `tirx.transform.<factory>` that runs the C++ pass and, on a copy of the
    module, the Rust one, requires them to agree, and returns the C++ result."""
    native_factory = getattr(tvm_tirx.tirx.transform, factory)
    rust_factory = tvm_ffi.get_global_func(f"rust.tirx.transform.{factory}")

    def make():
        native, rust = native_factory(), rust_factory()

        def run(mod, _ctx):
            copy = tvm_tirx.ir.load_json(tvm_tirx.ir.save_json(mod))
            try:
                expected = native(mod)
            except Exception:
                with pytest.raises(Exception):
                    rust(copy)
                runs.append("both failed")
                raise
            actual = rust(copy)
            assert tvm_ffi.structural_equal(expected, actual, map_free_vars=True), (
                tvm_ffi.get_first_structural_mismatch(expected, actual, map_free_vars=True)
            )
            runs.append("compared")
            return expected

        return tvm_tirx.transform.module_pass(run, opt_level=0, name=f"parity.{factory}")

    return make


@pytest.mark.parametrize("factory,test_name", list(_upstream_tests()))
def test_upstream_suite_parity(monkeypatch, factory, test_name):
    test = getattr(_suite(factory), test_name)
    runs = []
    monkeypatch.setattr(tvm_tirx.tirx.transform, factory, _dual_run(factory, runs))
    for kwargs in _parametrizations(test):
        test(**kwargs)
    assert runs, f"{test_name} never ran {factory}"
