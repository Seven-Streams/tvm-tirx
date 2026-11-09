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

# The parser consumes named IR bindings that Python sees as unused.
# ruff: noqa: F841
"""Tests for T.cuda.cta_reduce / cta_sum / cta_max / cta_min intrinsics."""

import numpy as np
import pytest

import tvm_tirx
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env

TARGET = tvm_tirx.target.Target("cuda")


def _build_and_run(func, n):
    mod = tvm_tirx.IRModule({"main": func})
    mod = tvm_tirx.compile(mod, target=TARGET, pipeline="tirx")
    out_np = np.zeros(n, dtype="float32")

    def run_and_check():
        dev = tvm_tirx.cuda(0)
        out = tvm_tirx.runtime.tensor(out_np, device=dev)
        mod(out)
        return out.numpy()

    return tvm_tirx.testing.run_with_gpu_lock(run_and_check), mod


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_cta_sum_4_warps():
    """CTA sum with 4 warps (128 threads): all threads get the same sum."""
    NUM_WARPS = 4
    N = NUM_WARPS * 32

    # fmt: off
    @T.function
    def func(out: T.Tensor((N,), 'float32')):

        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=N))
        cta_id = T.cuda.block_idx('x')
        warp_id = T.cuda.warp_id()
        lane_id = T.cuda.lane_id()
        tid = T.cuda.thread_idx('x')
        scratch = T.alloc_tensor(ty_args=[T.Tensor((NUM_WARPS,), "float32", scope="shared")])
        val: T.f32 = T.float32(tid + 1)
        val = T.cuda.cta_sum(val, NUM_WARPS, scratch.ptr_to([0]))
        out[tid] = val
        # fmt: on

    result, mod = _build_and_run(func, N)
    expected = np.float32(N * (N + 1) / 2)  # sum(1..128)
    np.testing.assert_allclose(result, np.full(N, expected))
    assert "cta_reduce_sum_4" in mod.mod.imports[0].inspect_source()


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_cta_sum_8_warps():
    """CTA sum with 8 warps (256 threads)."""
    NUM_WARPS = 8
    N = NUM_WARPS * 32

    # fmt: off
    @T.function
    def func(out: T.Tensor((N,), 'float32')):

        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=N))
        cta_id = T.cuda.block_idx('x')
        warp_id = T.cuda.warp_id()
        lane_id = T.cuda.lane_id()
        tid = T.cuda.thread_idx('x')
        scratch = T.alloc_tensor(ty_args=[T.Tensor((NUM_WARPS,), "float32", scope="shared")])
        val: T.f32 = T.float32(tid + 1)
        val = T.cuda.cta_sum(val, NUM_WARPS, scratch.ptr_to([0]))
        out[tid] = val
        # fmt: on

    result, _ = _build_and_run(func, N)
    expected = np.float32(N * (N + 1) / 2)
    np.testing.assert_allclose(result, np.full(N, expected))


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_cta_max_4_warps():
    """CTA max with 4 warps: all threads get the maximum value."""
    NUM_WARPS = 4
    N = NUM_WARPS * 32

    # fmt: off
    @T.function
    def func(out: T.Tensor((N,), 'float32')):

        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=N))
        cta_id = T.cuda.block_idx('x')
        warp_id = T.cuda.warp_id()
        lane_id = T.cuda.lane_id()
        tid = T.cuda.thread_idx('x')
        scratch = T.alloc_tensor(ty_args=[T.Tensor((NUM_WARPS,), "float32", scope="shared")])
        val: T.f32 = T.float32(tid + 1)
        val = T.cuda.cta_max(val, NUM_WARPS, scratch.ptr_to([0]))
        out[tid] = val
        # fmt: on

    result, _ = _build_and_run(func, N)
    np.testing.assert_allclose(result, np.full(N, float(N)))


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_cta_min_4_warps():
    """CTA min with 4 warps: all threads get the minimum value."""
    NUM_WARPS = 4
    N = NUM_WARPS * 32

    # fmt: off
    @T.function
    def func(out: T.Tensor((N,), 'float32')):

        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=N))
        cta_id = T.cuda.block_idx('x')
        warp_id = T.cuda.warp_id()
        lane_id = T.cuda.lane_id()
        tid = T.cuda.thread_idx('x')
        scratch = T.alloc_tensor(ty_args=[T.Tensor((NUM_WARPS,), "float32", scope="shared")])
        val: T.f32 = T.float32(tid + 1)
        val = T.cuda.cta_min(val, NUM_WARPS, scratch.ptr_to([0]))
        out[tid] = val
        # fmt: on

    result, _ = _build_and_run(func, N)
    np.testing.assert_allclose(result, np.full(N, 1.0))


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_cta_sum_1_warp():
    """CTA sum with 1 warp: degenerates to a pure warp reduce."""
    NUM_WARPS = 1
    N = 32

    # fmt: off
    @T.function
    def func(out: T.Tensor((N,), 'float32')):

        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=N))
        cta_id = T.cuda.block_idx('x')
        warp_id = T.cuda.warp_id()
        lane_id = T.cuda.lane_id()
        tid = T.cuda.thread_idx('x')
        scratch = T.alloc_tensor(ty_args=[T.Tensor((NUM_WARPS,), "float32", scope="shared")])
        val: T.f32 = T.float32(tid + 1)
        val = T.cuda.cta_sum(val, NUM_WARPS, scratch.ptr_to([0]))
        out[tid] = val
        # fmt: on

    result, _ = _build_and_run(func, N)
    expected = np.float32(32 * 33 / 2)
    np.testing.assert_allclose(result, np.full(N, expected))


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("num_warps", [1, 2, 4, 8, 16])
def test_cta_sum_all_warp_counts(num_warps):
    """Parametric test: cta_sum with various warp counts."""
    N = num_warps * 32

    # fmt: off
    @T.function
    def func(out: T.Tensor((N,), 'float32')):

        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=N))
        cta_id = T.cuda.block_idx('x')
        warp_id = T.cuda.warp_id()
        lane_id = T.cuda.lane_id()
        tid = T.cuda.thread_idx('x')
        scratch = T.alloc_tensor(ty_args=[T.Tensor((num_warps,), "float32", scope="shared")])
        val: T.f32 = T.float32(tid + 1)
        val = T.cuda.cta_sum(val, num_warps, scratch.ptr_to([0]))
        out[tid] = val
        # fmt: on

    result, _ = _build_and_run(func, N)
    expected = np.float32(N * (N + 1) / 2)
    np.testing.assert_allclose(result, np.full(N, expected))


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_cta_sum_beside_warp_sum():
    """cta_reduce embeds the 32-lane warp_reduce helper; calling both in one
    kernel must still define that helper once."""
    NUM_WARPS = 4
    N = NUM_WARPS * 32

    # fmt: off
    @T.function
    def func(out: T.Tensor((2 * N,), "float32")):

        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=N))
        tid = T.cuda.thread_idx("x")
        scratch = T.alloc_tensor(ty_args=[T.Tensor((NUM_WARPS,), "float32", scope="shared")])
        val: T.f32 = T.float32(tid + 1)
        warp_val: T.f32 = T.cuda.warp_sum(val)
        cta_val: T.f32 = T.cuda.cta_sum(val, NUM_WARPS, scratch.ptr_to([0]))
        out[tid] = cta_val
        out[N + tid] = warp_val
        # fmt: on

    result, _ = _build_and_run(func, 2 * N)
    vals = np.arange(1, N + 1, dtype="float32")
    warp_sums = np.repeat(vals.reshape(NUM_WARPS, 32).sum(axis=1), 32)
    np.testing.assert_allclose(result[:N], np.full(N, vals.sum()))
    np.testing.assert_allclose(result[N:], warp_sums)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_cta_sum_with_two_warp_counts_in_one_module():
    """The kernels of a module share one source, so cta_reduce helpers for two
    warp counts must not both define the warp_reduce helper they embed."""

    def make(num_warps):
        n = num_warps * 32

        # fmt: off
        @T.function
        def func(out: T.Tensor((n,), "float32")):

            T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=n))
            tid = T.cuda.thread_idx("x")
            scratch = T.alloc_tensor(ty_args=[T.Tensor((num_warps,), "float32", scope="shared")])
            val: T.f32 = T.float32(tid + 1)
            out[tid] = T.cuda.cta_sum(val, num_warps, scratch.ptr_to([0]))
            # fmt: on

        return func.with_attr("global_symbol", f"cta_sum_{num_warps}")

    warp_counts = (2, 4)
    mod = tvm_tirx.IRModule({f"cta_sum_{nw}": make(nw) for nw in warp_counts})
    mod = tvm_tirx.compile(mod, target=TARGET, pipeline="tirx")

    def run():
        dev = tvm_tirx.cuda(0)
        results = {}
        for nw in warp_counts:
            out = tvm_tirx.runtime.tensor(np.zeros(nw * 32, dtype="float32"), device=dev)
            mod[f"cta_sum_{nw}"](out)
            results[nw] = out.numpy()
        return results

    for nw, result in tvm_tirx.testing.run_with_gpu_lock(run).items():
        n = nw * 32
        np.testing.assert_allclose(result, np.full(n, n * (n + 1) / 2))


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("op", ["max", "min"])
@pytest.mark.parametrize("num_warps", [2, 4])
def test_cta_extremum_of_int32_above_float_precision(op, num_warps):
    """Integer max and min are exact across warps: the cross-warp step must not
    convert int32 values through float, which keeps only 24 bits."""
    N = num_warps * 32
    base = 16777217  # 2**24 + 1: neighbouring odd values share one float32
    sign = 1 if op == "max" else -1
    cta_reduce = T.cuda.cta_max if op == "max" else T.cuda.cta_min

    # fmt: off
    @T.function
    def func(out: T.Tensor((N,), "int32")):

        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=N))
        tid = T.cuda.thread_idx("x")
        scratch = T.alloc_tensor(ty_args=[T.Tensor((num_warps,), "int32", scope="shared")])
        val: T.int32 = T.int32(sign) * (T.int32(base) + tid * 2)
        out[tid] = cta_reduce(val, num_warps, scratch.ptr_to([0]))
        # fmt: on

    mod = tvm_tirx.compile(tvm_tirx.IRModule({"main": func}), target=TARGET, pipeline="tirx")
    assert "INFINITY" not in mod.mod.imports[0].inspect_source()

    def run():
        out = tvm_tirx.runtime.tensor(np.zeros(N, dtype="int32"), device=tvm_tirx.cuda(0))
        mod(out)
        return out.numpy()

    result = tvm_tirx.testing.run_with_gpu_lock(run)
    np.testing.assert_array_equal(result, np.full(N, sign * (base + 2 * (N - 1)), dtype="int32"))
