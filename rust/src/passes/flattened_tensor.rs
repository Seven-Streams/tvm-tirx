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

//! The tensor geometry the flattening passes share: `TensorTypeNode::ElemOffset` and its index
//! simplification (`src/tirx/ir/tensor.cc`), `MatchShapeStridePattern`
//! (`include/tvm/tirx/type.h`), `FlattenedTensor` (`src/tirx/transform/flatten_tensor_utils.h`),
//! and small `TensorVar` helpers.

use tvm_ffi::{AnyView, Array, ObjectCore, ObjectRefCast, ObjectRefCore, Result, VALUE_ERROR};

use tvm_ffi::DLDataType;

use crate::ir::{Expr, Type, Var, VarObj};
use crate::ops::pattern::{PVar, Pattern, PatternOps};
use crate::ops::{deep_equal, int32, int_imm};
use crate::ir::prim::{AddObj, FloorDivObj, FloorModObj, MulObj};
use crate::sym::Analyzer;
use crate::tirx::{Axis, ComposeLayoutObj, TensorType, TensorTypeObj, TileLayout, TileLayoutObj};
use crate::ir::prim::op::{Ex, PrimOps};
use crate::ir::Location;

/// A `TensorVar`: a `Var`, not a subclass, whose type is a `TensorType` (C++ `as<TensorVar>()`).
pub(crate) fn as_tensor_var(expr: impl Into<Expr>) -> Option<(Var, TensorType)> {
    let expr: Expr = expr.into();
    if expr.as_node::<VarObj>().is_none()
        || AnyView::from(&expr).type_index() != VarObj::type_index()
    {
        return None;
    }
    let ty: TensorType = expr.ty.clone().try_cast().ok()?;
    Some((expr.try_cast().ok()?, ty))
}

/// The `TensorType` of a variable known to be a tensor.
pub(crate) fn tensor_type(var: &Var) -> Result<TensorType> {
    var.ty.clone().try_cast()
}

/// C++ `TensorTypeNode::DefaultIndexType()`: the type of the first extent, or int64 (TVM is built
/// with `TVM_INDEX_DEFAULT_I64`).
pub(crate) fn default_index_type(ty: &TensorTypeObj) -> Result<DLDataType> {
    match ty.shape.len() {
        0 => Ok(crate::ir::dtype::int(64)),
        _ => ty.shape.get(0)?.prim_dtype(),
    }
}

/// `(shape, strides, element offset)` of a shape-aligned strided tensor.
pub(crate) type ShapeStridePattern = (Array<Expr>, Array<Expr>, Expr);

/// C++ `MatchShapeStridePattern(tensor)`: `(shape, strides, element offset)` of a tensor whose
/// memory is one strided memory-axis iterator per dimension, with only a memory-axis offset. No
/// layout is row-major at offset 0; any other layout is `None`. Extents compare by
/// `ExprDeepEqual`.
pub(crate) fn match_shape_stride_pattern(ty: &TensorType) -> Result<Option<ShapeStridePattern>> {
    let index_ty = default_index_type(ty)?;
    let mut offset: Expr = int_imm(index_ty, 0)?.into();
    let mut strides = Vec::with_capacity(ty.shape.len());
    match &ty.layout {
        None => {
            let mut stride: Expr = int_imm(index_ty, 1)?.into();
            strides.resize(ty.shape.len(), stride.clone());
            for i in (0..ty.shape.len()).rev() {
                strides[i] = stride.clone();
                stride = (Ex::from(&stride) * &ty.shape.get(i)?).expr()?;
            }
        }
        Some(layout) => {
            let Some(tile) = layout.as_node::<TileLayoutObj>() else { return Ok(None) };
            let shard = tile.shard()?;
            if !tile.replica()?.is_empty() || shard.len() != ty.shape.len() {
                return Ok(None);
            }
            let memory_axis = Axis::get("m")?;
            for (iter, extent) in shard.iter().zip(ty.shape.iter()) {
                if !iter.axis.same_as(&memory_axis) || !deep_equal(&iter.extent, &extent)? {
                    return Ok(None);
                }
                strides.push(iter.stride.clone());
            }
            for (axis, value) in tile.offset()?.iter() {
                if !axis.same_as(&memory_axis) {
                    return Ok(None);
                }
                offset = value;
            }
        }
    }
    Ok(Some((ty.shape.clone(), Array::new(strides), offset)))
}

/// C++ `FlattenedTensor(tensor, analyzer, fold_origin)`: the 1-d physical view of `tensor`, which
/// is `tensor` itself when it is 1-d without a layout. Its extent covers the physical footprint:
/// stride gaps of a shape-aligned strided layout (with its memory origin when `fold_origin`), the
/// memory span of another tile layout, or the swizzle-rounded span of a composed one.
pub(crate) fn flattened_tensor(
    tensor: &Var,
    ty: &TensorType,
    analyzer: Option<&Analyzer>,
    fold_origin: bool,
) -> Result<(Var, TensorType)> {
    let fresh;
    let analyzer = match analyzer {
        Some(analyzer) => analyzer,
        None => {
            fresh = Analyzer::new();
            &fresh
        }
    };
    let shape = &ty.shape;
    let product = || -> Result<Ex> {
        let mut extent = Ex::from(int32(1));
        for dim in shape.iter() {
            extent *= &dim;
        }
        Ok(extent)
    };
    let tile = ty.layout.as_ref().and_then(|layout| layout.clone().try_cast::<TileLayout>().ok());
    let explicit_strides = tile.is_some() && match_shape_stride_pattern(ty)?.is_some();
    let extent: Expr = if let (true, Some(tile)) = (explicit_strides, &tile) {
        // Preserve the outer stride footprint and cover every addressed element.
        let shard = tile.shard()?;
        let mut span = Ex::from(int32(1));
        for iter in shard.iter() {
            span += (Ex::from(&iter.extent) - 1) * &iter.stride;
        }
        let mut extent = match tile.is_contiguous(Some(analyzer))? {
            true => product()?,
            false => (Ex::from(&shard.get(0)?.stride) * &shape.get(0)?).max(span),
        };
        if fold_origin {
            // Absolute flat indices include the origin.
            if let Some(origin) = tile.offset()?.get(&Axis::get("m")?)? {
                extent += &origin;
            }
        }
        extent.expr()?
    } else if let Some(tile) = &tile {
        analyzer.simplify(&crate::tirx::Layout::from(tile.clone()).span(Some("m"))?, 2)?
    } else if let Some(compose) = ty.layout.as_ref().and_then(|l| l.as_node::<ComposeLayoutObj>()) {
        let tile_layout = compose.tile_layout()?;
        let mut span = Ex::from(crate::tirx::Layout::from(tile_layout.clone()).span(Some("m"))?);
        if tile_layout.is_trivial()? {
            span = Ex::from(int_imm(default_index_type(ty)?, 1)?);
            for dim in shape.iter() {
                span *= &dim;
            }
        }
        if compose.swizzle_len()? != 0 {
            let mut changed_bits = compose.per_element()? + compose.swizzle_len()?;
            if !compose.swizzle_inner()? {
                changed_bits += compose.atom_len()?;
            }
            let span_ty = span.clone().expr()?.prim_dtype()?;
            let block: Expr = int_imm(span_ty, 1i128 << changed_bits)?.into();
            span = (span + &block - 1).floordiv(&block) * &block;
        }
        analyzer.simplify(&span.expr()?, 2)?
    } else {
        product()?.expr()?
    };
    if shape.len() == 1 && ty.layout.is_none() {
        return Ok((tensor.clone(), ty.clone()));
    }
    let flat_ty = TensorType::new(
        [extent],
        &ty.dtype,
        ty.scope.clone(),
        ty.data_alignment,
        None,
        Location::unknown(),
    )?;
    Ok((Var::new(&tensor.name, Type::from(&flat_ty), &tensor.loc), flat_ty))
}

/// C++ `TensorTypeNode::ElemOffset(indices)`: the offset of an element in elements, simplified
/// with a fresh analyzer. A layout maps the indices to its memory axis, which must be its only
/// output.
pub(crate) fn elem_offset(ty: &TensorType, indices: &Array<Expr>) -> Result<Array<Expr>> {
    let shape = &ty.shape;
    check_cmp!(
        shape.len(), ==, indices.len(), tvm_ffi::INTERNAL_ERROR,
        "shape.size() == input_indices.size()",
        "TensorType is {}-dimensional, cannot be indexed with the {}-dimensional indices \
         provided.",
        shape.len(),
        indices.len(),
    );
    let analyzer = Analyzer::new();
    if let Some(layout) = &ty.layout {
        let mapped = layout.apply_with_shape(indices, shape)?;
        if mapped.is_empty() && shape.is_empty() {
            return Ok(Array::new(vec![int_imm(default_index_type(ty)?, 0)?.into()]));
        }
        let memory = mapped.get(&tvm_ffi::String::from("m"))?;
        let offset = match (mapped.len(), memory) {
            (1, Some(offset)) => offset,
            _ => tvm_ffi::bail!(
                VALUE_ERROR,
                "Check failed: (mapped.size() == 1 && mapped.find(\"m\") != mapped.end()) is \
                 false: Tensor indexing requires a memory-only layout"
            ),
        };
        return Ok(Array::new(vec![analyzer.simplify(&offset, 2)?]));
    }
    let mut output = int32(0)?.into();
    for (i, index) in indices.iter().enumerate() {
        output = (Ex::from(&output) * &shape.get(i)? + &index).expr()?;
        if i > 0 {
            output = merge_mul_mod(&analyzer, output)?;
        }
    }
    Ok(Array::new(vec![analyzer.simplify(&output, 2)?]))
}

/// C++ `ExprSplitAddition`: the terms of a tree of `Add`, left to right.
fn split_addition(expr: &Expr) -> Vec<Expr> {
    let mut terms = Vec::new();
    let mut stack = vec![expr.clone()];
    while let Some(top) = stack.pop() {
        match top.as_node::<AddObj>() {
            Some(add) => {
                stack.push(add.b.clone());
                stack.push(add.a.clone());
            }
            None => terms.push(top),
        }
    }
    terms
}

/// `lhs + rhs`, or `rhs` when there is no `lhs` yet.
fn sum(lhs: Option<Expr>, rhs: &Expr) -> Result<Expr> {
    match lhs {
        Some(lhs) => (Ex::from(&lhs) + rhs).expr(),
        None => Ok(rhs.clone()),
    }
}

/// C++ `MergeMulModInner`: `(a1 + ... + c / k * k1 ...) * kt ...` merged with `c % k` into
/// `(a1 + ...) * kt ... + c`, when the factors match and `c` equals the modulus operand mod `k`.
fn merge_mul_mod_inner(
    analyzer: &Analyzer,
    mult_expr: &Expr,
    mod_lhs: &Expr,
    mod_rhs: &Expr,
) -> Result<Option<Expr>> {
    let Some(mult) = mult_expr.as_node::<MulObj>() else { return Ok(None) };
    // 1. The outer multiplier.
    let mut mult_outer = mult.b.clone();
    let mut inner = mult.a.clone();
    while let Some(mult) = inner.as_node::<MulObj>() {
        mult_outer = (Ex::from(&mult.b) * &mult_outer).expr()?;
        inner = mult.a.clone();
    }
    // 2. Search `c / (...) * (...) + c % (...)`.
    let mut search = inner;
    let mut mult_inner: Option<Expr> = None;
    let mut no_opt_sum: Option<Expr> = None;
    loop {
        if let Some(div) = search.as_node::<FloorDivObj>() {
            let overall = match &mult_inner {
                Some(mult_inner) => (Ex::from(mult_inner) * &mult_outer).expr()?,
                None => mult_outer.clone(),
            };
            let remainder = (Ex::from(&div.a) - mod_lhs).floormod(mod_rhs).expr()?;
            if deep_equal(&overall, &div.b)?
                && deep_equal(&overall, mod_rhs)?
                && analyzer.can_prove_equal(&remainder, int32(0)?)?
            {
                return Ok(Some(match no_opt_sum {
                    Some(sum) => (Ex::from(&sum) * &mult_outer + &div.a).expr()?,
                    None => div.a.clone(),
                }));
            }
            return Ok(None);
        } else if let Some(mult) = search.as_node::<MulObj>() {
            mult_inner = Some(match &mult_inner {
                Some(mult_inner) => (Ex::from(&mult.b) * mult_inner).expr()?,
                None => mult.b.clone(),
            });
            search = mult.a.clone();
        } else if let Some(add) = search.as_node::<AddObj>() {
            if mult_inner.is_some() {
                return Ok(None);
            }
            no_opt_sum = Some(sum(no_opt_sum, &add.a)?);
            search = add.b.clone();
        } else {
            return Ok(None);
        }
    }
}

/// The terms of a sum, sorted as C++ `MergeMulModInsertElements` sorts them.
#[derive(Default)]
struct Terms {
    mults: Vec<Expr>,
    mods: Vec<(Expr, Expr)>,
    no_opt_sum: Option<Expr>,
}

impl Terms {
    /// Returns whether a product and whether a modulus were among `terms`.
    fn insert(&mut self, terms: Vec<Expr>) -> Result<(bool, bool)> {
        let (mut has_mult, mut has_mod) = (false, false);
        for term in terms {
            if let Some(modulo) = term.as_node::<FloorModObj>() {
                has_mod = true;
                self.mods.push((modulo.a.clone(), modulo.b.clone()));
            } else if term.as_node::<MulObj>().is_some() {
                has_mult = true;
                self.mults.push(term);
            } else {
                self.no_opt_sum = Some(sum(self.no_opt_sum.take(), &term)?);
            }
        }
        Ok((has_mult, has_mod))
    }
}

/// C++ `MergeMulMod`: `x // y * y + x % y` becomes `x`, and every product that pairs with a
/// modulus as `MergeMulModInner` describes is merged, repeatedly.
fn merge_mul_mod(analyzer: &Analyzer, base: Expr) -> Result<Expr> {
    let (x, y) = (PVar::expr(), PVar::expr());
    let base = match (x.floordiv(&y) * &y + x.floormod(&y)).matches(&base) {
        true => x.get(),
        false => base,
    };
    let simplified = analyzer.simplify(&base, 2)?;
    let mut terms = Terms::default();
    terms.insert(split_addition(&simplified))?;
    let mut find_opt = false;
    // The C++ list iterator over `mods`, as an index; `mods.len()` is its end.
    let mut search = 0;
    while search < terms.mods.len() {
        let mut inner_find_opt = false;
        let mut mult = 0;
        while mult < terms.mults.len() {
            let (mod_lhs, mod_rhs) = terms.mods[search].clone();
            match merge_mul_mod_inner(analyzer, &terms.mults[mult], &mod_lhs, &mod_rhs)? {
                Some(merged) => {
                    inner_find_opt = true;
                    // Erasing the current modulus leaves the iterator on its successor.
                    terms.mods.remove(search);
                    terms.mults.remove(mult);
                    let was_end = search == terms.mods.len();
                    let (has_mult, has_mod) = terms.insert(split_addition(&merged))?;
                    if has_mult {
                        search = 0;
                    } else if has_mod && was_end {
                        search = terms.mods.len() - 1;
                    }
                    break;
                }
                None => mult += 1,
            }
        }
        find_opt = find_opt || inner_find_opt;
        if !inner_find_opt {
            search += 1;
        }
    }
    if !find_opt {
        return Ok(simplified);
    }
    let mut total = terms.no_opt_sum;
    for mult in &terms.mults {
        total = Some(sum(total, mult)?);
    }
    for (lhs, rhs) in &terms.mods {
        total = Some(sum(total, &lhs.indexmod(rhs).expr()?)?);
    }
    Ok(total.expect("a merge leaves a term"))
}
