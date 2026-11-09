/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file dtype_conversion.cc
 * \brief Header file of data type conversion routines.
 */
#include "dtype_conversion.h"

#include <tvm/tirx/op/memory.h>

#include <algorithm>
#include <cmath>

namespace tvm {
namespace tirx {
using namespace tvm::prim;

PrimExpr ReinterpretAsUInt(PrimExpr value) {
  return tirx::reinterpret(GetStorageUIntDType(value.ty()), value);
}

PrimType GetStorageUIntDType(PrimType dtype) {
  if (dtype.IsScalableVector()) {
    return PrimType::ScalableVector(DLDataTypeCode::kDLUInt, dtype.bits(), dtype.VScaleFactor());
  }
  return PrimType::UInt(dtype.bits(), dtype.lanes());
}

namespace {

/*! \brief The largest finite value of a format. */
double MaxFinite(const FloatConfig& fp) {
  // The top exponent is a finite binade unless it encodes infinity or IEEE NaNs; with all-ones
  // NaNs only its last mantissa pattern is NaN.
  bool top_binade_finite = fp.infty_style == FloatConfig::InftyStyle::kNone &&
                           fp.nan_style != FloatConfig::NaNStyle::kIEEE;
  int max_exponent = (1 << fp.exponent) - (top_binade_finite ? 1 : 2) - fp.bias;
  int64_t max_mantissa = (int64_t(1) << fp.mantissa) - 1;
  if (top_binade_finite && fp.nan_style == FloatConfig::NaNStyle::kAllOnes) max_mantissa -= 1;
  return std::ldexp(1.0 + std::ldexp(static_cast<double>(max_mantissa), -fp.mantissa),
                    max_exponent);
}

/*! \brief Round-half-to-even right shift of a non-negative integer by `shift` >= 1 bits. */
PrimExpr RoundShiftRight(PrimExpr value, int shift) {
  PrimExpr half_minus_one = MakeConst(value.ty(), (int64_t(1) << (shift - 1)) - 1);
  return (value + half_minus_one + ((value >> shift) & 1)) >> shift;
}

/*! \brief Round-half-to-even right shift by a run-time amount of at least one bit. */
PrimExpr RoundShiftRight(PrimExpr value, PrimExpr shift) {
  PrimType ty = value.ty();
  PrimExpr one = MakeConst(ty, 1);
  PrimExpr half_minus_one = (one << (shift - one)) - one;
  return (value + half_minus_one + ((value >> shift) & one)) >> shift;
}

/*!
 * \brief Convert the bits of a float to a format whose exponent field has another width.
 *
 * The source is split into sign, exponent and significand; zeros, subnormals, infinities and
 * NaNs are told apart, and the value is encoded in the target with round-half-to-even,
 * producing target subnormals where the value needs them. A magnitude above the target's largest
 * finite value becomes infinity, or NaN in all-ones formats without infinity, or the largest
 * finite value in formats with neither, as ml_dtypes converts. Infinities and NaNs map the same
 * way; NaNs become zero in formats without NaN. Signs are kept.
 */
PrimExpr ConvertAcrossExponentWidths(PrimExpr src_bits, const FloatConfig& src_fp,
                                     const FloatConfig& tgt_fp, PrimType src_uint,
                                     PrimType tgt_uint) {
  using NaNStyle = FloatConfig::NaNStyle;
  using InftyStyle = FloatConfig::InftyStyle;
  const int ms = src_fp.mantissa, es = src_fp.exponent;
  const int md = tgt_fp.mantissa, ed = tgt_fp.exponent;
  const int shift = ms - md;  // significand bits dropped (> 0) or added (< 0)
  TVM_FFI_ICHECK(tgt_fp.nan_style != NaNStyle::kIEEE || md > 0)
      << "An IEEE NaN needs a mantissa bit";
  // Unsigned work type: a capped target exponent above the source significand, the bits the
  // target adds, and a rounding carry. Exponents are signed.
  PrimType work = src_uint.WithBits(ed + 1 + std::max(ms, md) + 2 <= 32 ? 32 : 64);
  PrimType sint = src_uint.WithCode(DLDataTypeCode::kDLInt).WithBits(32);

  // Decode.
  int64_t magnitude_mask = (int64_t(1) << (es + ms)) - 1;
  int64_t src_infinity = ((int64_t(1) << es) - 1) << ms;
  PrimExpr magnitude = src_bits & MakeConst(src_uint, magnitude_mask);
  PrimExpr sign = cast(src_bits >> (es + ms), tgt_uint) << (ed + md);
  PrimExpr exponent = cast(magnitude >> ms, sint);
  PrimExpr fraction = cast(magnitude, work) & MakeConst(work, (int64_t(1) << ms) - 1);

  // The significand with its leading one at bit `ms`, and the target-biased exponent.
  PrimExpr significand = fraction | MakeConst(work, int64_t(1) << ms);
  PrimExpr tgt_exponent = exponent + MakeConst(sint, tgt_fp.bias - src_fp.bias);
  // Source subnormals are normalized unless every one of them rounds to zero in the target.
  bool normalize_subnormals = ms > 0 && src_fp.bias < 1 + tgt_fp.bias + md;
  if (normalize_subnormals) {
    // fraction * 2^(1 - bias - ms), with its leading one at bit k: shift that one to bit ms.
    PrimExpr sub_significand = fraction << ms;
    PrimExpr sub_exponent = MakeConst(sint, 1 - src_fp.bias + tgt_fp.bias - ms);
    for (int k = 1; k < ms; ++k) {
      PrimExpr has_bit = fraction >= MakeConst(work, int64_t(1) << k);
      sub_significand = Select(has_bit, fraction << (ms - k), sub_significand);
      sub_exponent =
          Select(has_bit, MakeConst(sint, 1 - src_fp.bias + tgt_fp.bias - (ms - k)), sub_exponent);
    }
    PrimExpr is_subnormal = exponent == MakeConst(sint, 0);
    significand = Select(is_subnormal, sub_significand, significand);
    tgt_exponent = Select(is_subnormal, sub_exponent, tgt_exponent);
  }

  // Encode a target normal: the exponent is capped one above the largest, which overflows.
  PrimExpr capped = max(min(tgt_exponent, MakeConst(sint, int64_t(1) << ed)), MakeConst(sint, 1));
  PrimExpr kept_fraction =
      normalize_subnormals ? (significand & MakeConst(work, (int64_t(1) << ms) - 1)) : fraction;
  PrimExpr result = (cast(capped, work) << ms) | kept_fraction;
  if (shift > 0) {
    result = RoundShiftRight(result, shift);
  } else if (shift < 0) {
    result = result << (-shift);
  }
  // Encode a target subnormal: the significand shifted right by shift + 1 - exponent bits.
  int min_tgt_exponent = 1 - src_fp.bias + tgt_fp.bias - (normalize_subnormals ? ms : 0);
  if (min_tgt_exponent < 1) {
    PrimExpr right = MakeConst(sint, shift + 1) - tgt_exponent;
    // Past ms + 2 bits every significand rounds to zero; capping keeps the shift in range.
    PrimExpr capped_right =
        cast(min(max(right, MakeConst(sint, 1)), MakeConst(sint, ms + 2)), work);
    PrimExpr subnormal = RoundShiftRight(significand, capped_right);
    if (shift < 0) {
      // Fewer bits are dropped than the target adds: close to the normal range it shifts left.
      PrimExpr left = cast(max(MakeConst(sint, 0) - right, MakeConst(sint, 0)), work);
      subnormal = Select(right > MakeConst(sint, 0), subnormal, significand << left);
    }
    result = Select(tgt_exponent >= MakeConst(sint, 1), result, subnormal);
  }

  // Out of range, and the special values.
  int64_t tgt_all_ones = (int64_t(1) << (ed + md)) - 1;
  int64_t tgt_infinity = ((int64_t(1) << ed) - 1) << md;
  int64_t tgt_nan = tgt_fp.nan_style == NaNStyle::kIEEE ? tgt_infinity | (int64_t(1) << (md - 1))
                    : tgt_fp.nan_style == NaNStyle::kAllOnes ? tgt_all_ones
                                                             : 0;
  int64_t tgt_max_finite = tgt_fp.infty_style == InftyStyle::kIEEE ? tgt_infinity - 1
                           : tgt_fp.nan_style == NaNStyle::kNone   ? tgt_all_ones
                                                                   : tgt_all_ones - 1;
  int64_t tgt_overflow = tgt_fp.infty_style == InftyStyle::kIEEE  ? tgt_infinity
                         : tgt_fp.nan_style == NaNStyle::kAllOnes ? tgt_all_ones
                                                                  : tgt_max_finite;
  if (MaxFinite(src_fp) > MaxFinite(tgt_fp)) {
    result =
        Select(result > MakeConst(work, tgt_max_finite), MakeConst(work, tgt_overflow), result);
  }
  // Zeros, and subnormals that round to zero.
  PrimExpr is_zero = normalize_subnormals ? (magnitude == MakeConst(src_uint, 0))
                                          : (exponent == MakeConst(sint, 0));
  result = Select(is_zero, MakeConst(work, 0), result);
  if (src_fp.infty_style == InftyStyle::kIEEE) {
    result = Select(magnitude == MakeConst(src_uint, src_infinity), MakeConst(work, tgt_overflow),
                    result);
  }
  if (src_fp.nan_style == NaNStyle::kIEEE) {
    result =
        Select(magnitude > MakeConst(src_uint, src_infinity), MakeConst(work, tgt_nan), result);
  } else if (src_fp.nan_style == NaNStyle::kAllOnes) {
    result =
        Select(magnitude == MakeConst(src_uint, magnitude_mask), MakeConst(work, tgt_nan), result);
  }
  return sign | cast(result, tgt_uint);
}

}  // namespace

PrimExpr DTypeConversion(PrimExpr src_value, PrimType tgt_dtype, RoundingMode round_mode) {
  PrimType src_dtype = src_value.ty();
  // Step 1: check dtype
  // The lanes of src dtype and target dtype must match.
  TVM_FFI_ICHECK_EQ(src_dtype->dtype.lanes, tgt_dtype->dtype.lanes)
      << "The lanes for data type for source value must matches the target datatype.";
  auto is_floating_point = [](PrimType dtype) {
    return dtype.MatchesCode(DLDataTypeCode::kDLFloat, DLDataTypeCode::kDLBfloat,
                             DLDataTypeCode::kDLFloat8_e3m4, DLDataTypeCode::kDLFloat8_e4m3,
                             DLDataTypeCode::kDLFloat8_e4m3b11fnuz,
                             DLDataTypeCode::kDLFloat8_e4m3fn, DLDataTypeCode::kDLFloat8_e4m3fnuz,
                             DLDataTypeCode::kDLFloat8_e5m2, DLDataTypeCode::kDLFloat8_e5m2fnuz,
                             DLDataTypeCode::kDLFloat8_e8m0fnu, DLDataTypeCode::kDLFloat6_e2m3fn,
                             DLDataTypeCode::kDLFloat6_e3m2fn, DLDataTypeCode::kDLFloat4_e2m1fn);
  };
  // Both source dtype and target dtype should be floating point.
  TVM_FFI_ICHECK(is_floating_point(src_dtype) && is_floating_point(tgt_dtype));
  FloatConfig src_fp = FloatConfig::FromDataType(src_dtype),
              tgt_fp = FloatConfig::FromDataType(tgt_dtype);
  int exponent_delta = tgt_fp.exponent - src_fp.exponent;
  int bias_delta = tgt_fp.bias - src_fp.bias;
  int mantissa_delta = tgt_fp.mantissa - src_fp.mantissa;
  PrimType src_uint = GetStorageUIntDType(src_dtype), tgt_uint = GetStorageUIntDType(tgt_dtype);
  PrimExpr src_uint_value = ReinterpretAsUInt(src_value);
  PrimExpr src_bits = src_uint_value;  // before rounding
  if (mantissa_delta < 0) {
    TVM_FFI_ICHECK(round_mode == RoundingMode::kHalfToEven)
        << "Currently we only support HalfToEven rounding mode.";
  }
  if (exponent_delta != 0) {
    return tirx::reinterpret(
        tgt_dtype, ConvertAcrossExponentWidths(src_bits, src_fp, tgt_fp, src_uint, tgt_uint));
  }
  if (mantissa_delta < 0) {
    // The exponent fields line up: rounding the bits rounds the value, and a carry out of the
    // mantissa steps the exponent.
    PrimExpr rounding_bias = ((src_uint_value >> (-mantissa_delta)) & 1) +
                             MakeConst(src_uint, (int64_t(1) << (-mantissa_delta - 1)) - 1);
    src_uint_value = src_uint_value + rounding_bias;
  }
  PrimExpr ret = src_uint_value;
  if (mantissa_delta >= 0) {
    ret = cast(ret, tgt_uint) << mantissa_delta;
  } else {  // mantissa_delta < 0
    ret = cast(ret >> (-mantissa_delta), tgt_uint);
  }
  if (bias_delta > 0) {
    ret = ret + (MakeConst(tgt_uint, bias_delta) << tgt_fp.mantissa);
  } else if (bias_delta < 0) {
    ret = ret - (MakeConst(tgt_uint, -bias_delta) << tgt_fp.mantissa);
  }
  if (mantissa_delta < 0 && src_fp.nan_style == FloatConfig::NaNStyle::kIEEE &&
      tgt_fp.nan_style == FloatConfig::NaNStyle::kIEEE && src_fp.bits() < 64) {
    // Rounding adds to the bits, which carries a NaN with a large payload into the sign
    // (float32 0x7fffffff would become bfloat16 0x8000, -0.0). Truncate NaNs instead and set
    // the quiet bit, so that they stay NaN.
    int64_t magnitude_mask = (int64_t(1) << (src_fp.exponent + src_fp.mantissa)) - 1;
    int64_t infinity = ((int64_t(1) << src_fp.exponent) - 1) << src_fp.mantissa;
    PrimExpr is_nan =
        (src_bits & MakeConst(src_uint, magnitude_mask)) > MakeConst(src_uint, infinity);
    PrimExpr quiet_nan = cast(src_bits >> (-mantissa_delta), tgt_uint) |
                         MakeConst(tgt_uint, int64_t(1) << (tgt_fp.mantissa - 1));
    ret = Select(is_nan, quiet_nan, ret);
  }
  return tirx::reinterpret(tgt_dtype, ret);
}

}  // namespace tirx
}  // namespace tvm
