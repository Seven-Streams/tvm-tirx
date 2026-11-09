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
import ml_dtypes
import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.script
import tvm_tirx.testing
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target
from tvm_tirx.tirx.transform.transform import BindTarget

# pylint: disable=no-member,invalid-name,unused-variable


def get_before(dtype: str):
    @tvm_tirx.script.ir_module
    class Before:
        @T.function
        def main(Aptr: T.handle(dtype), Bptr: T.handle(dtype), Dptr: T.handle(dtype)):
            T.func_attr({"global_symbol": "main"})
            A = T.decl_tensor(Aptr, ty_args=[T.Tensor((100,), dtype, data=Aptr)])
            B = T.decl_tensor(Bptr, ty_args=[T.Tensor((100,), dtype, data=Bptr)])
            D = T.decl_tensor(Dptr, ty_args=[T.Tensor((100,), dtype, data=Dptr)])
            C = T.alloc_tensor(ty_args=[T.Tensor((100,), dtype)])
            for i in T.grid(100):
                C[i] = A[i] + B[i]
                D[i] = T.exp(C[i])

    return Before


def promote_f8(f8_dtype: str, promote_dtype: str, v):
    return promote_uint8(f8_dtype, promote_dtype, T.reinterpret(v, ty="uint8"))


def cast_to_f8(f8_dtype: str, promote_dtype: str, v):
    return T.reinterpret(cast_to_uint8(f8_dtype, promote_dtype, v), ty=f8_dtype)


def get_after_compute_legalize(dtype: str, promote_dtype: str):
    @tvm_tirx.script.ir_module
    class After:
        @T.function
        def main(Aptr: T.handle(dtype), Bptr: T.handle(dtype), Dptr: T.handle(dtype)):
            T.func_attr({"global_symbol": "main"})
            A = T.decl_tensor(Aptr, ty_args=[T.Tensor((100,), dtype, data=Aptr)])
            B = T.decl_tensor(Bptr, ty_args=[T.Tensor((100,), dtype, data=Bptr)])
            D = T.decl_tensor(Dptr, ty_args=[T.Tensor((100,), dtype, data=Dptr)])
            C = T.alloc_tensor(ty_args=[T.Tensor((100,), promote_dtype)])
            for i in T.grid(100):
                C[i] = promote_f8(dtype, promote_dtype, A[i]) + promote_f8(
                    dtype, promote_dtype, B[i]
                )
                D[i] = cast_to_f8(dtype, promote_dtype, T.exp(C[i]))

    return After


# The formats these tests convert between: (exponent bits, mantissa bits, bias, has infinity,
# NaN style), as FloatConfig in src/tirx/transform/dtype_conversion.h describes them.
_FORMATS = {
    "float16": (5, 10, 15, True, "ieee"),
    "float32": (8, 23, 127, True, "ieee"),
    "float8_e4m3fn": (4, 3, 7, False, "all_ones"),
    "float8_e5m2": (5, 2, 15, True, "ieee"),
}


def _max_finite(fmt):
    exponent, mantissa, bias, has_infinity, nan = _FORMATS[fmt]
    top_binade_finite = not has_infinity and nan != "ieee"
    max_exponent = (1 << exponent) - (1 if top_binade_finite else 2) - bias
    max_mantissa = (1 << mantissa) - 1 - (1 if top_binade_finite and nan == "all_ones" else 0)
    return (1 + max_mantissa / (1 << mantissa)) * 2.0**max_exponent


def convert_bits(src_bits, src: str, dst: str):
    """The bits of a ``dst`` value from the bits of a ``src`` value, built as DTypeConversion
    builds them when the exponent fields differ in width (ConvertAcrossExponentWidths in
    src/tirx/transform/dtype_conversion.cc)."""
    es, ms, bs, src_infinity_style, src_nan_style = _FORMATS[src]
    ed, md, bd, tgt_infinity_style, tgt_nan_style = _FORMATS[dst]
    src_uint = getattr(T, f"uint{1 + es + ms}")
    tgt_uint = f"uint{1 + ed + md}"
    work_name = "uint32" if ed + 1 + max(ms, md) + 2 <= 32 else "uint64"
    work, i32 = getattr(T, work_name), T.int32
    shift = ms - md

    def round_shift_right(value, amount):
        if isinstance(amount, int):
            half_minus_one = work((1 << (amount - 1)) - 1)
            dropped_lsb = T.bitwise_and(T.shift_right(value, work(amount)), work(1))
            return T.shift_right(value + half_minus_one + dropped_lsb, work(amount))
        one = work(1)
        half_minus_one = T.shift_left(one, amount - one) - one
        dropped_lsb = T.bitwise_and(T.shift_right(value, amount), one)
        return T.shift_right(value + half_minus_one + dropped_lsb, amount)

    magnitude_mask = (1 << (es + ms)) - 1
    src_infinity = ((1 << es) - 1) << ms
    magnitude = T.bitwise_and(src_bits, src_uint(magnitude_mask))
    sign = T.shift_left(
        T.shift_right(src_bits, src_uint(es + ms)).astype(tgt_uint), getattr(T, tgt_uint)(ed + md)
    )
    exponent = T.shift_right(magnitude, src_uint(ms)).astype("int32")
    fraction = T.bitwise_and(magnitude.astype(work_name), work((1 << ms) - 1))
    significand = T.bitwise_or(fraction, work(1 << ms))
    tgt_exponent = exponent + i32(bd - bs)
    normalize_subnormals = ms > 0 and bs < 1 + bd + md
    if normalize_subnormals:
        sub_significand = T.shift_left(fraction, work(ms))
        sub_exponent = i32(1 - bs + bd - ms)
        for k in range(1, ms):
            has_bit = fraction >= work(1 << k)
            sub_significand = T.Select(
                has_bit, T.shift_left(fraction, work(ms - k)), sub_significand
            )
            sub_exponent = T.Select(has_bit, i32(1 - bs + bd - (ms - k)), sub_exponent)
        is_subnormal = exponent == i32(0)
        significand = T.Select(is_subnormal, sub_significand, significand)
        tgt_exponent = T.Select(is_subnormal, sub_exponent, tgt_exponent)

    capped = T.max(T.min(tgt_exponent, i32(1 << ed)), i32(1))
    kept_fraction = fraction
    if normalize_subnormals:
        kept_fraction = T.bitwise_and(significand, work((1 << ms) - 1))
    result = T.bitwise_or(T.shift_left(capped.astype(work_name), work(ms)), kept_fraction)
    if shift > 0:
        result = round_shift_right(result, shift)
    elif shift < 0:
        result = T.shift_left(result, work(-shift))
    if 1 - bs + bd - (ms if normalize_subnormals else 0) < 1:
        right = i32(shift + 1) - tgt_exponent
        subnormal = round_shift_right(
            significand, T.min(T.max(right, i32(1)), i32(ms + 2)).astype(work_name)
        )
        if shift < 0:
            left = T.max(i32(0) - right, i32(0)).astype(work_name)
            subnormal = T.Select(right > i32(0), subnormal, T.shift_left(significand, left))
        result = T.Select(tgt_exponent >= i32(1), result, subnormal)

    all_ones = (1 << (ed + md)) - 1
    infinity = ((1 << ed) - 1) << md
    if tgt_nan_style == "ieee":
        nan = infinity | (1 << (md - 1))
    else:
        nan = all_ones if tgt_nan_style == "all_ones" else 0
    if tgt_infinity_style:
        max_finite, overflow = infinity - 1, infinity
    elif tgt_nan_style == "all_ones":
        max_finite, overflow = all_ones - 1, all_ones
    else:
        max_finite = overflow = all_ones
    if _max_finite(src) > _max_finite(dst):
        result = T.Select(result > work(max_finite), work(overflow), result)
    is_zero = magnitude == src_uint(0) if normalize_subnormals else exponent == i32(0)
    result = T.Select(is_zero, work(0), result)
    if src_infinity_style:
        result = T.Select(magnitude == src_uint(src_infinity), work(overflow), result)
    if src_nan_style == "ieee":
        result = T.Select(magnitude > src_uint(src_infinity), work(nan), result)
    elif src_nan_style == "all_ones":
        result = T.Select(magnitude == src_uint(magnitude_mask), work(nan), result)
    return T.bitwise_or(sign, result.astype(tgt_uint))


def promote_uint8(f8_dtype: str, promote_dtype: str, v):
    if f8_dtype == "float8_e5m2" and promote_dtype == "float16":
        # The exponent fields line up: the bits only move.
        return T.reinterpret(T.shift_left(T.Cast("uint16", v), T.uint16(8)), ty="float16")
    return T.reinterpret(convert_bits(v, f8_dtype, promote_dtype), ty=promote_dtype)


def cast_to_uint8(f8_dtype: str, promote_dtype: str, v):
    if f8_dtype == "float8_e4m3fn" or promote_dtype == "float32":
        src_bits = T.reinterpret(v, ty=f"uint{promote_dtype[len('float') :]}")
        return convert_bits(src_bits, promote_dtype, f8_dtype)
    # float8_e5m2 from float16: the exponent fields line up, so the bits are rounded in place.
    uint16_v = T.reinterpret(v, ty="uint16")
    lsb = T.bitwise_and(T.shift_right(uint16_v, T.uint16(8)), T.uint16(1))
    rounding_bias = lsb + T.uint16(0x7F)
    rounded = T.Cast("uint8", T.shift_right(uint16_v + rounding_bias, T.uint16(8)))
    # NaNs are truncated with the quiet bit set: rounding them could carry into the sign.
    is_nan = T.bitwise_and(uint16_v, T.uint16(0x7FFF)) > T.uint16(0x7C00)
    quiet_nan = T.bitwise_or(T.Cast("uint8", T.shift_right(uint16_v, T.uint16(8))), T.uint8(0x2))
    return T.Select(is_nan, quiet_nan, rounded)


def get_after_storage_legalize(dtype: str, promote_dtype: str):
    @tvm_tirx.script.ir_module
    class After:
        @T.function
        def main(Aptr: T.handle("uint8"), Bptr: T.handle("uint8"), Dptr: T.handle("uint8")):
            T.func_attr({"global_symbol": "main"})
            A = T.decl_tensor(Aptr, ty_args=[T.Tensor((100,), "uint8", data=Aptr)])
            B = T.decl_tensor(Bptr, ty_args=[T.Tensor((100,), "uint8", data=Bptr)])
            D = T.decl_tensor(Dptr, ty_args=[T.Tensor((100,), "uint8", data=Dptr)])
            C = T.alloc_tensor(ty_args=[T.Tensor((100,), promote_dtype)])
            for i in T.grid(100):
                C[i] = promote_uint8(dtype, promote_dtype, A[i]) + promote_uint8(
                    dtype, promote_dtype, B[i]
                )
                D[i] = cast_to_uint8(dtype, promote_dtype, T.exp(C[i]))

    return After


@pytest.fixture(params=["float8_e4m3fn", "float8_e5m2"], scope="session")
def dtype(request):
    return request.param


@pytest.fixture(params=["float16", "float32"], scope="session")
def promote_dtype(request):
    return request.param


def test_fp8_compute_legalize(dtype, promote_dtype):
    target = Target("nvidia/nvidia-a100")
    before = BindTarget(target)(get_before(dtype))
    expected = BindTarget(target)(get_after_compute_legalize(dtype, promote_dtype))
    # run the transform twice to ensure we can afford to deal
    # with this repeative optimizations
    after = tvm_tirx.tirx.transform.FP8ComputeLegalize(promote_dtype)(before)
    after = tvm_tirx.tirx.transform.FP8ComputeLegalize(promote_dtype)(after)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_fp8_compute_legalize_preserves_opaque_tensor_access(dtype, promote_dtype):
    @T.function
    def before():
        tensor = T.alloc_tensor(ty_args=[T.Tensor((16,), dtype)])
        T.evaluate(T.call_extern("consume", tensor.data, ty="void"))

    input_func = before

    @I.ir_module(check_well_formed=False)
    class InputModule:
        main = input_func

    before_mod = InputModule
    after = tvm_tirx.tirx.transform.FP8ComputeLegalize(promote_dtype)(before_mod)
    tvm_tirx.ir.assert_structural_equal(after, before_mod)


def test_fp8_storage_legalize(dtype, promote_dtype):
    target = Target("nvidia/nvidia-a100")
    before = BindTarget(target)(get_after_compute_legalize(dtype, promote_dtype))
    after = tvm_tirx.tirx.transform.FP8StorageLegalize()(before)
    expected = BindTarget(target)(get_after_storage_legalize(dtype, promote_dtype))
    tvm_tirx.ir.assert_structural_equal(after, expected)


def _convert_on_cpu(values, dst, lanes=1):
    """Cast every element of ``values`` to ``dst`` in a program compiled for LLVM, where float8
    has no native support and the cast is legalized to integer operations, ``lanes`` at a time."""
    n = values.size
    src = str(values.dtype)

    @T.function
    def convert(A: T.Tensor((n,), src), B: T.Tensor((n,), dst)):
        for i in range(n // lanes):
            for j in T.vectorized(lanes):
                B[i * lanes + j] = T.Cast(dst, A[i * lanes + j])

    func = tvm_tirx.compile(convert, target="llvm")
    out = tvm_tirx.runtime.empty((n,), dst)
    func(tvm_tirx.runtime.tensor(values), out)
    return out.numpy()


def _assert_same_floats(got, want):
    got32, want32 = got.astype("float32"), want.astype("float32")
    np.testing.assert_array_equal(np.isnan(got32), np.isnan(want32))
    finite = ~np.isnan(want32)
    # Bitwise on the non-NaN values, so that zeros keep their signs.
    np.testing.assert_array_equal(got32[finite].view("uint32"), want32[finite].view("uint32"))


@pytest.mark.parametrize("dst", ["float8_e4m3fn", "float8_e5m2"])
def test_fp8_from_every_float16(dst):
    """Every float16, subnormals, rounding ties, overflow, infinities and NaNs included, rounds
    to the float8 that ml_dtypes gives."""
    values = np.arange(1 << 16, dtype="uint16").view("float16")
    _assert_same_floats(_convert_on_cpu(values, dst), values.astype(getattr(ml_dtypes, dst)))


@pytest.mark.parametrize("dst", ["float8_e4m3fn", "float8_e5m2"])
def test_fp8_from_float32(dst):
    """Float32 values around the float8 range and its subnormals, special values, and random
    bit patterns round to the float8 that ml_dtypes gives."""
    fp8 = getattr(ml_dtypes, dst)
    grid = np.arange(256, dtype="uint8").view(fp8).astype("float32")
    grid = grid[np.isfinite(grid)]
    # Each float8 value, the midpoints next to it (ties), and the floats just beside those.
    steps = np.abs(np.diff(np.sort(grid))).min(initial=1.0)
    near = np.concatenate([grid, grid * 1.0625, grid * 0.9375, grid + steps / 2, grid * 2.0])
    near = np.concatenate(
        [near, np.nextafter(near, np.float32(np.inf)), np.nextafter(near, np.float32(-np.inf))]
    ).astype("float32")
    special = np.array(
        [
            0.0,
            -0.0,
            np.inf,
            -np.inf,
            np.nan,
            -np.nan,
            1e30,
            -1e30,
            1e-30,
            -1e-30,
            np.finfo("float32").tiny,
            np.finfo("float32").max,
        ],
        dtype="float32",
    )
    rng = np.random.default_rng(0)
    random_bits = rng.integers(0, 1 << 32, size=1 << 16, dtype=np.uint64).astype("uint32")
    values = np.concatenate([near, special, random_bits.view("float32")])
    _assert_same_floats(_convert_on_cpu(values, dst), values.astype(fp8))


@pytest.mark.parametrize("src", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("dst", ["float16", "float32"])
def test_fp8_to_wider_float(src, dst):
    """Every float8, zeros, subnormals, infinities and NaNs included, widens exactly."""
    values = np.arange(256, dtype="uint8").view(getattr(ml_dtypes, src))
    _assert_same_floats(_convert_on_cpu(values, dst), values.astype(dst))


@pytest.mark.parametrize("fp8", ["float8_e4m3fn", "float8_e5m2"])
def test_fp8_conversions_vectorize(fp8):
    """The legalized conversions take vector operands: their selects are elementwise."""
    narrow = np.arange(256, dtype="uint8").view(getattr(ml_dtypes, fp8))
    _assert_same_floats(_convert_on_cpu(narrow, "float32", lanes=4), narrow.astype("float32"))
    wide = np.concatenate([narrow.astype("float32"), [np.inf, -np.inf, 1e30, -1e-30]])
    wide = wide.astype("float32")
    _assert_same_floats(_convert_on_cpu(wide, fp8, lanes=4), wide.astype(getattr(ml_dtypes, fp8)))


if __name__ == "__main__":
    tvm_tirx.testing.main()
