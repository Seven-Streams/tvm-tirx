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
