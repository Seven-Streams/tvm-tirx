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
#![allow(dead_code, unused_imports)]

//! The primitive operators: arithmetic, comparisons, logic, casts and math intrinsics.
//!
//! The functions `python/tvm_tirx/ir/prim/op.py` defines, with the operators that
//! `python/tvm_tirx/ir/_prim_expr_overload.py` gives expressions (`a + b` is [`add`]). Unlike the
//! node constructors such as [`Add::new`](super::Add::new), which require operands of one type,
//! these promote their operands to a common type, fold constants and may return any [`Expr`]: a
//! node, a constant or an intrinsic call.

// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::Any
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): tvm_ffi::VALUE_ERROR
// tvm-ffi-stubgen(import-object): tvm_ffi::Error
// tvm-ffi-stubgen(import-object): crate::ir::expr::Expr
// tvm-ffi-stubgen(import-object): crate::ir::location::Location
// tvm-ffi-stubgen(import-object): crate::ir::r#type::PrimType
// tvm-ffi-stubgen(import-object): crate::ir::prim::expr::BitwiseNot
// tvm-ffi-stubgen(import-object): crate::ir::expr::IntImm
// tvm-ffi-stubgen(import-object): crate::ir::prim::vector_expr::Broadcast
// tvm-ffi-stubgen(import-object): crate::ir::prim::vector_expr::vscale_lanes
// tvm-ffi-stubgen(import-object): crate::ir::support
// tvm-ffi-stubgen(import-object): crate::ir::dtype::with_lanes
// tvm-ffi-stubgen(import-object): crate::ir::prim::vector_expr::ConstValue
// tvm-ffi-stubgen(import-object): crate::ir::prim::vector_expr::make_const_scalar
// tvm-ffi-stubgen(import-object): tvm_ffi::BigInt

// tvm-ffi-stubgen(begin): import-section
use crate::ir::dtype::with_lanes;
use crate::ir::expr::Expr;
use crate::ir::expr::IntImm;
use crate::ir::location::Location;
use crate::ir::prim::expr::BitwiseNot;
use crate::ir::prim::vector_expr::Broadcast;
use crate::ir::prim::vector_expr::ConstValue;
use crate::ir::prim::vector_expr::make_const_scalar;
use crate::ir::prim::vector_expr::vscale_lanes;
use crate::ir::r#type::PrimType;
use crate::ir::support;
use tvm_ffi::Any;
use tvm_ffi::BigInt;
use tvm_ffi::Error;
use tvm_ffi::Result;
use tvm_ffi::VALUE_ERROR;
// tvm-ffi-stubgen(end)

/// `value` as an expression: a Rust `bool`, integer or float becomes a constant of the default
/// type, and an expression is returned as is (`prim.convert`).
pub fn convert(value: impl Into<Any>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim.convert").call_tuple((value.into(),))?.try_into()
}

/// `a + b` (`prim._OpAdd`).
pub fn add(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim._OpAdd")
        .call_tuple((&a.into(), &b.into(), &loc.into()))?
        .try_into()
}

/// `a - b` (`prim._OpSub`).
pub fn sub(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim._OpSub")
        .call_tuple((&a.into(), &b.into(), &loc.into()))?
        .try_into()
}

/// `a * b` (`prim._OpMul`).
pub fn mul(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim._OpMul")
        .call_tuple((&a.into(), &b.into(), &loc.into()))?
        .try_into()
}

/// `a / b`; integer division rounds towards zero, as in C (`prim._OpDiv`).
pub fn div(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim._OpDiv")
        .call_tuple((&a.into(), &b.into(), &loc.into()))?
        .try_into()
}

/// `a / b` over non-negative indices, rounding down (`prim._OpIndexDiv`).
pub fn indexdiv(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim._OpIndexDiv")
        .call_tuple((&a.into(), &b.into(), &loc.into()))?
        .try_into()
}

/// `a % b` over non-negative indices (`prim._OpIndexMod`).
pub fn indexmod(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim._OpIndexMod")
        .call_tuple((&a.into(), &b.into(), &loc.into()))?
        .try_into()
}

/// `a / b`, rounding towards zero (`prim._OpTruncDiv`).
pub fn truncdiv(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim._OpTruncDiv")
        .call_tuple((&a.into(), &b.into(), &loc.into()))?
        .try_into()
}

/// The remainder of [`truncdiv`], with the sign of `a` (`prim._OpTruncMod`).
pub fn truncmod(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim._OpTruncMod")
        .call_tuple((&a.into(), &b.into(), &loc.into()))?
        .try_into()
}

/// `a / b`, rounding down (`prim._OpFloorDiv`).
pub fn floordiv(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim._OpFloorDiv")
        .call_tuple((&a.into(), &b.into(), &loc.into()))?
        .try_into()
}

/// The remainder of [`floordiv`], with the sign of `b` (`prim._OpFloorMod`).
pub fn floormod(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim._OpFloorMod")
        .call_tuple((&a.into(), &b.into(), &loc.into()))?
        .try_into()
}

/// `a / b`, rounding up (`prim._OpCeilDiv`).
pub fn ceildiv(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim._OpCeilDiv")
        .call_tuple((&a.into(), &b.into(), &loc.into()))?
        .try_into()
}

/// The smaller of `a` and `b` (`prim._OpMin`).
pub fn min(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim._OpMin")
        .call_tuple((&a.into(), &b.into(), &loc.into()))?
        .try_into()
}

/// The larger of `a` and `b` (`prim._OpMax`).
pub fn max(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim._OpMax")
        .call_tuple((&a.into(), &b.into(), &loc.into()))?
        .try_into()
}

/// `a` to the power `b` (`prim._OpPow`).
pub fn pow(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim._OpPow")
        .call_tuple((&a.into(), &b.into(), &loc.into()))?
        .try_into()
}

/// The boolean `a == b` (`prim._OpEQ`).
pub fn eq(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim._OpEQ")
        .call_tuple((&a.into(), &b.into(), &loc.into()))?
        .try_into()
}

/// The boolean `a != b` (`prim._OpNE`).
pub fn ne(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim._OpNE")
        .call_tuple((&a.into(), &b.into(), &loc.into()))?
        .try_into()
}

/// The boolean `a < b` (`prim._OpLT`).
pub fn lt(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim._OpLT")
        .call_tuple((&a.into(), &b.into(), &loc.into()))?
        .try_into()
}

/// The boolean `a <= b` (`prim._OpLE`).
pub fn le(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim._OpLE")
        .call_tuple((&a.into(), &b.into(), &loc.into()))?
        .try_into()
}

/// The boolean `a > b` (`prim._OpGT`).
pub fn gt(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim._OpGT")
        .call_tuple((&a.into(), &b.into(), &loc.into()))?
        .try_into()
}

/// The boolean `a >= b` (`prim._OpGE`).
pub fn ge(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim._OpGE")
        .call_tuple((&a.into(), &b.into(), &loc.into()))?
        .try_into()
}

/// `a & b` (`prim.bitwise_and`).
pub fn bitwise_and(
    a: impl Into<Expr>,
    b: impl Into<Expr>,
    loc: impl Into<Location>,
) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim.bitwise_and")
        .call_tuple((&a.into(), &b.into(), &loc.into()))?
        .try_into()
}

/// `a | b` (`prim.bitwise_or`).
pub fn bitwise_or(
    a: impl Into<Expr>,
    b: impl Into<Expr>,
    loc: impl Into<Location>,
) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim.bitwise_or")
        .call_tuple((&a.into(), &b.into(), &loc.into()))?
        .try_into()
}

/// `a ^ b` (`prim.bitwise_xor`).
pub fn bitwise_xor(
    a: impl Into<Expr>,
    b: impl Into<Expr>,
    loc: impl Into<Location>,
) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim.bitwise_xor")
        .call_tuple((&a.into(), &b.into(), &loc.into()))?
        .try_into()
}

/// `a << b` (`prim.left_shift`).
pub fn shift_left(
    a: impl Into<Expr>,
    b: impl Into<Expr>,
    loc: impl Into<Location>,
) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim.left_shift")
        .call_tuple((&a.into(), &b.into(), &loc.into()))?
        .try_into()
}

/// `a >> b` (`prim.right_shift`).
pub fn shift_right(
    a: impl Into<Expr>,
    b: impl Into<Expr>,
    loc: impl Into<Location>,
) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim.right_shift")
        .call_tuple((&a.into(), &b.into(), &loc.into()))?
        .try_into()
}

/// The node `~x`, which, unlike the other operators, neither promotes nor folds
/// ([`BitwiseNot::new`]).
pub fn bitwise_not(x: impl Into<Expr>, loc: impl Into<Location>) -> Result<Expr> {
    Ok(BitwiseNot::new(x, loc)?.into())
}

/// The conjunction of `conds`, which must not be empty (`prim._OpAnd`).
pub fn all(
    conds: impl IntoIterator<Item = impl Into<Expr>>,
    loc: impl Into<Location>,
) -> Result<Expr> {
    fold(conds, loc.into(), "all", |a, b, loc| {
        tvm_ffi::cached_global_func!("prim._OpAnd").call_tuple((a, b, loc))?.try_into()
    })
}

/// The disjunction of `conds`, which must not be empty (`prim._OpOr`).
pub fn any(
    conds: impl IntoIterator<Item = impl Into<Expr>>,
    loc: impl Into<Location>,
) -> Result<Expr> {
    fold(conds, loc.into(), "any", |a, b, loc| {
        tvm_ffi::cached_global_func!("prim._OpOr").call_tuple((a, b, loc))?.try_into()
    })
}

fn fold(
    conds: impl IntoIterator<Item = impl Into<Expr>>,
    loc: Location,
    name: &str,
    op: impl Fn(&Expr, &Expr, &Location) -> Result<Expr>,
) -> Result<Expr> {
    let mut conds = conds.into_iter().map(Into::into);
    let Some(first) = conds.next() else {
        let message = format!("{name} must take at least 1 argument");
        return Err(Error::new(VALUE_ERROR, &message, ""));
    };
    conds.try_fold(first, |acc, cond| op(&acc, &cond, &loc))
}

/// `true_value` where `cond` holds and `false_value` elsewhere, evaluating only the chosen one
/// (`prim._OpIfThenElse`).
pub fn if_then_else(
    cond: impl Into<Expr>,
    true_value: impl Into<Expr>,
    false_value: impl Into<Expr>,
    loc: impl Into<Location>,
) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim._OpIfThenElse")
        .call_tuple((&cond.into(), &true_value.into(), &false_value.into(), &loc.into()))?
        .try_into()
}

/// `value` converted to `ty` (`prim._cast`).
pub fn cast(
    value: impl Into<Expr>,
    ty: impl Into<PrimType>,
    loc: impl Into<Location>,
) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim._cast")
        .call_tuple((&value.into(), &ty.into(), &loc.into()))?
        .try_into()
}

/// The absolute value of `x` (`prim.abs`).
pub fn abs(x: impl Into<Expr>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim.abs").call_tuple((&x.into(), &loc.into()))?.try_into()
}

/// `x` rounded up (`prim.ceil`).
pub fn ceil(x: impl Into<Expr>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim.ceil").call_tuple((&x.into(), &loc.into()))?.try_into()
}

/// `x` rounded down (`prim.floor`).
pub fn floor(x: impl Into<Expr>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim.floor").call_tuple((&x.into(), &loc.into()))?.try_into()
}

/// `x` rounded to the nearest integer, halfway cases away from zero (`prim.round`).
pub fn round(x: impl Into<Expr>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim.round").call_tuple((&x.into(), &loc.into()))?.try_into()
}

/// `x` rounded to the nearest integer under the current rounding mode (`prim.nearbyint`).
pub fn nearbyint(x: impl Into<Expr>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim.nearbyint").call_tuple((&x.into(), &loc.into()))?.try_into()
}

/// `x` rounded towards zero (`prim.trunc`).
pub fn trunc(x: impl Into<Expr>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim.trunc").call_tuple((&x.into(), &loc.into()))?.try_into()
}

/// Whether `x` is NaN (`prim.isnan`).
pub fn isnan(x: impl Into<Expr>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim.isnan").call_tuple((&x.into(), &loc.into()))?.try_into()
}

/// Whether `x` is finite (`prim.isfinite`).
pub fn isfinite(x: impl Into<Expr>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim.isfinite").call_tuple((&x.into(), &loc.into()))?.try_into()
}

/// Whether `x` is infinite (`prim.isinf`).
pub fn isinf(x: impl Into<Expr>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim.isinf").call_tuple((&x.into(), &loc.into()))?.try_into()
}

/// `cond`, marked as likely to hold (`prim.likely`).
pub fn likely(cond: impl Into<Expr>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim.likely").call_tuple((&cond.into(), &loc.into()))?.try_into()
}

/// A statement that `cond` holds, for simplifications (`prim.assume`).
pub fn assume(cond: impl Into<Expr>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim.assume").call_tuple((&cond.into(), &loc.into()))?.try_into()
}

/// The smallest value of `ty` (`prim.min_value`).
pub fn min_value(ty: impl Into<PrimType>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim.min_value").call_tuple((&ty.into(), &loc.into()))?.try_into()
}

/// The largest value of `ty` (`prim.max_value`).
pub fn max_value(ty: impl Into<PrimType>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim.max_value").call_tuple((&ty.into(), &loc.into()))?.try_into()
}

/// The positive infinity of `ty` (`prim.infinity`).
pub fn infinity(ty: impl Into<PrimType>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim.infinity").call_tuple((&ty.into(), &loc.into()))?.try_into()
}

/// The constant `value` of type `ty`: an [`IntImm`] of an integer or boolean type, a [`FloatImm`]
/// of a floating-point type, and the scalar constant broadcast over the lanes of a vector type
/// (C++ `prim::MakeConst(ty, value, loc)`, which has no global function). An unsigned type takes
/// no negative value. Prefer [`IntImm::from_dtype`] or [`FloatImm::from_dtype`] when the kind of
/// the type is known.
///
/// [`FloatImm`]: crate::ir::FloatImm
/// [`FloatImm::from_dtype`]: crate::ir::FloatImm::from_dtype
pub fn make_const(ty: impl Into<PrimType>, value: i64, loc: impl Into<Location>) -> Result<Expr> {
    let (ty, loc) = (ty.into(), loc.into());
    let dtype = ty.dtype;
    let value = ConstValue::Int(BigInt::from(value));
    if !support::is_scalable_vector(dtype) && !support::is_fixed_length_vector(dtype) {
        return make_const_scalar(dtype, value, &loc);
    }
    let scalar = make_const_scalar(with_lanes(dtype, 1), value, &loc)?;
    let lanes = match support::is_fixed_length_vector(dtype) {
        true => IntImm::int32(support::lanes(dtype)?).into(),
        false => vscale_lanes(-i32::from(support::encoded_lanes(dtype)))?,
    };
    Ok(Broadcast::new(scalar, lanes, loc)?.into())
}
