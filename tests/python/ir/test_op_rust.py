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
"""Operators defined in Rust are registered when their library is loaded, and used as C++ ones are.

``rust/examples/define_ops.rs`` is built by cargo and loaded by a fresh interpreter, as
``import tvm_tirx`` loads libtvm_tirx_rust, which then uses its operators as it uses C++ ones:
their TVMScript builders are Python functions written for them.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

RUST_DIR = Path(__file__).resolve().parents[3] / "rust"


def find_cargo():
    """A cargo as new as the crate's rust-version, rustup's first, as the CMake build looks."""
    manifest = (RUST_DIR / "Cargo.toml").read_text()
    needed = re.search(r'^rust-version *= *"([0-9.]+)"', manifest, re.M).group(1)
    needed = tuple(int(part) for part in needed.split("."))
    homes = [Path(os.environ["CARGO_HOME"])] if "CARGO_HOME" in os.environ else []
    candidates = [str(home / "bin" / "cargo") for home in [*homes, Path.home() / ".cargo"]]
    for cargo in [*candidates, shutil.which("cargo")]:
        if not cargo or not os.access(cargo, os.X_OK):
            continue
        version = subprocess.run([cargo, "--version"], capture_output=True, text=True).stdout
        match = re.search(r"(\d+)\.(\d+)\.(\d+)", version)
        if match and tuple(int(part) for part in match.groups()) >= needed:
            return cargo
    return None


@pytest.fixture(scope="module")
def define_ops_library():
    """The example library of Rust operators, built by cargo."""
    cargo = find_cargo()
    if cargo is None or not shutil.which("tvm-ffi-config") or not shutil.which("tvm-tirx-config"):
        pytest.skip(
            "building rust/ needs a cargo as new as its rust-version, and tvm-ffi-config and "
            "tvm-tirx-config on PATH"
        )
    manifest = RUST_DIR / "Cargo.toml"
    subprocess.run(
        [cargo, "build", "--manifest-path", str(manifest), "--example", "define_ops"], check=True
    )
    target_dir = Path(os.environ.get("CARGO_TARGET_DIR", RUST_DIR / "target"))
    name = {"darwin": "libdefine_ops.dylib", "win32": "define_ops.dll"}.get(
        sys.platform, "libdefine_ops.so"
    )
    return target_dir / "debug" / "examples" / name


def run_python(code, library):
    """Run `code` in a fresh interpreter, with the path of `library` as its argument.

    The code runs from a file: TVMScript reads the source of the functions it parses, which Python
    keeps for code passed with `-c` only from 3.13 on.
    """
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "script.py"
        script.write_text(textwrap.dedent(code))
        result = subprocess.run(
            [sys.executable, str(script), str(library)],
            capture_output=True,
            text=True,
            check=False,
        )
    assert result.returncode == 0, result.stderr


def test_library_is_loaded_on_import():
    """``import tvm_tirx`` loads libtvm_tirx_rust, when the build has it, as a backend."""
    from tvm_ffi.libinfo import _find_library_by_basename  # pylint: disable=import-outside-toplevel

    from tvm_tirx.base import _LOADED_LIBS  # pylint: disable=import-outside-toplevel

    runtime_dir = Path(_LOADED_LIBS["tvm_runtime"]._name).resolve().parent
    try:
        _find_library_by_basename("tvm_tirx", "tvm_tirx_rust", extra_lib_paths=[runtime_dir])
    except RuntimeError:
        pytest.skip("the build has no libtvm_tirx_rust (TVM_TIRX_USE_RUST)")
    assert "tvm_tirx_rust" in _LOADED_LIBS


def test_rust_operators_are_registered_by_their_library(define_ops_library):
    run_python(
        """
        import sys

        import tvm_ffi
        import tvm_tirx
        from tvm_tirx import ir
        from tvm_tirx.script import tirx as T
        from tvm_tirx.script.ir_builder.base import auto_emit_void_call
        from tvm_tirx.tirx.script import ir_builder

        # Registered by the library, with the doc comment and signature the Rust definition gives.
        tvm_ffi.load_module(sys.argv[1]).tirx_register_ops()
        clamp = ir.Op.get("tirx.example_clamp")
        assert clamp.doc == "Clamp an index to the range `[0, bound)`."
        assert [arg.name for arg in clamp.args_info] == ["index", "bound"]
        mark = ir.Op.get("tirx.example_mark")
        assert mark.doc.startswith("Mark a point in the program.\\n\\nThe call has no value")
        hint = ir.Op.get("tirx.example_hint")
        loop = ir.Op.get("tirx.example_loop")

        # A Call built in Python runs the Rust type inference and validator.
        i, n = ir.Var("i", "int32"), ir.Var("n", "int32")
        call = ir.Call(clamp, [i, n])
        assert tvm_ffi.structural_equal(call.ty, i.ty)
        call.validate()
        try:
            ir.Call(clamp, [i, ir.Var("f", "float32")], ty="int32").validate()
        except TypeError as error:
            assert "`Call.args[1]` (`bound`) expected `ir.IntExpr`" in str(error), error
        else:
            raise AssertionError("a float bound passed validation")

        # The TVMScript builders, written in Python as those of the C++ operators are in
        # `tirx/script/ir_builder/op.py`, under the names the printer gives the calls.
        def example_clamp(index, bound):
            return ir.Call(clamp, [index, bound])

        @auto_emit_void_call
        def example_mark():
            return ir.Call(mark, [])

        def example_hint(value, attrs=None):
            return ir.Call(hint, [value], attrs=attrs)

        def example_loop(extent):
            return T.region(loop, [extent])

        for builder in (example_clamp, example_mark, example_hint, example_loop):
            setattr(ir_builder, builder.__name__, builder)

        # With them, TVMScript writes, prints and parses the calls as it does those of C++
        # operators: a call with a value, a call emitted as a statement, `Call.attrs`, and a region,
        # whose body parameter the Rust hook creates.
        @T.function
        def func(i: T.int32, n: T.int32):
            T.evaluate(T.example_clamp(i + 1, n))
            T.example_mark()
            T.evaluate(T.example_hint(i, attrs={"note": "hot"}))
            with T.example_loop(n) as j:
                T.evaluate(T.example_clamp(j, n))

        text = func.script()
        assert "T.example_clamp(i + 1, n)" in text, text
        assert "T.example_mark()" in text, text
        assert 'T.example_hint(i, attrs={"note": "hot"})' in text, text
        assert "with T.example_loop(n) as j:" in text, text
        parsed = tvm_tirx.script.parse(text, extra_vars={"T": T})
        assert tvm_ffi.structural_equal(parsed, func), parsed.script()
        assert parsed.script() == text
        """,
        define_ops_library,
    )
