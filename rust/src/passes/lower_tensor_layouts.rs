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

//! `tirx.LowerTensorLayouts` (`src/tirx/transform/lower_tensor_layouts.cc`): apply tensor
//! layouts to physical storage and accesses. Every tensor is flattened to its storage, an access
//! through a layout becomes the layout's element offset, and every `tensor_data_ptr` projects the
//! storage root. A flat view keeps a nonzero memory origin in a 1-d layout of its own.

use crate::fast_hash::HashMap;

use tvm_ffi::{
    dispatch, structural_mutate, Any, Array, Map, Mutator, ObjectIdentity, ObjectRefCast,
    ObjectRefCore, Result, INTERNAL_ERROR,
};

use super::analyzer::{AnalyzerBase, AnalyzerScope, AnalyzerState};
use super::analyzer::is_const_int;
use super::flattened_tensor::{
    as_tensor_var, default_index_type, elem_offset, flattened_tensor, match_shape_stride_pattern,
    tensor_type,
};
use super::util::{function_pass, get_attr, int_imm, MutatorExt};
use crate::ir::{Bind, Call, CallObj, Expr, SeqStmt, Stmt, TensorLoad, TensorStore, Type, Var};
use crate::ops::{int32, structural_equal};
use crate::sym::Analyzer;
use crate::target::Target;
use crate::tirx::op::{ALLOC_TENSOR, CUDA_DECL_TMEM, DECL_TENSOR, REINTERPRET, TENSOR_DATA_PTR};
use crate::tirx::{Axis, ComposeLayout, Iter, Layout, TensorType, TileLayout};
use crate::ir::transform::Pass;
use crate::ir::prim::op::{Ex, PrimOps};
use crate::ir::Location;

/// Where an expression of a tensor's geometry is rewritten: before the body, each in a traversal
/// of its own, or inside the body's traversal.
enum Site<'a> {
    Params,
    Body(&'a mut Mutator),
}

struct LayoutApplier {
    base: AnalyzerBase,
    /// `VarRemapSet(buf, flattened)`, and every definition the default descent rebuilt.
    remap: HashMap<ObjectIdentity, Var>,
    /// The storage root each tensor aliases, recorded at its definition.
    tensor_aliases: HashMap<ObjectIdentity, Var>,
}

impl AnalyzerState for LayoutApplier {
    fn base(&mut self) -> &mut AnalyzerBase {
        &mut self.base
    }
}

#[dispatch(mutate, policy = AnalyzerScope)]
impl LayoutApplier {
    /// A use of a remapped variable is its replacement; a definition the default descent rebuilds
    /// is recorded as the engine records it, so later lookups see it too.
    fn mutate_var(&mut self, var: Var, m: &mut Mutator) -> Result<Any> {
        if let Some(mapped) = self.remap.get(&ObjectIdentity::of(&var)) {
            return Ok(mapped.clone().into());
        }
        let result = m.default_mutate(self, &var)?;
        if let Ok(rebuilt) = Var::try_from(result.clone()) {
            if !rebuilt.same_as(&var) {
                self.remap.insert(ObjectIdentity::of(&var), rebuilt);
            }
        }
        Ok(result)
    }

    /// `tensor_data_ptr(t)` projects the storage root of `t`.
    fn mutate_call(&mut self, call: Call, m: &mut Mutator) -> Result<Any> {
        if TENSOR_DATA_PTR.is(&call.op) && call.args.len() == 1 {
            let arg = call.args.get(0)?;
            if let Ok(tensor) = arg.try_cast::<Var>() {
                if tensor_type(&tensor).is_ok() {
                    let Some(root) = self.tensor_aliases.get(&ObjectIdentity::of(&tensor)) else {
                        tvm_ffi::bail!(
                            INTERNAL_ERROR,
                            "Check failed: (root_opt.has_value()) is false: tensor_data_ptr \
                             projects {}, which has no visible definition \
                             (AllocTensor/DeclTensor/Function parameter) at this point",
                            tensor.name.as_str()
                        );
                    };
                    let root = self.remap.get(&ObjectIdentity::of(root)).unwrap_or(root);
                    return Ok(root.tensor_data()?.into());
                }
            }
        }
        m.default_mutate(self, &call)
    }

    fn mutate_bind(&mut self, op: Bind, m: &mut Mutator) -> Result<Any> {
        let var = op.var.clone();
        // `Bind(t, source)` of a tensor declares `t` over the source's data.
        if let Some((source, _)) = as_tensor_var(&op.value) {
            let declaration = declaration(&var, source.tensor_data()?)?;
            let binding = Bind::new(var, &declaration, &op.loc)?;
            let result = self.mutate_decl_tensor(&binding, &declaration, m)?;
            return Ok(result.unwrap_or(binding).into());
        }
        if op.value.as_node::<CallObj>().is_some() {
            let call: Call = op.value.clone().try_cast()?;
            if ALLOC_TENSOR.is(&call.op) {
                return self.mutate_alloc_tensor(&op, &call, m);
            }
            if DECL_TENSOR.is(&call.op) {
                return Ok(self.mutate_decl_tensor(&op, &call, m)?.unwrap_or(op).into());
            }
            if CUDA_DECL_TMEM.is(&call.op) {
                // TMEM is allocated by the pool; keep an explicit backing pointer so data
                // projections work for views and parameters too.
                let ty = tensor_type(&var)?;
                let pointer = Call::new(
                    Some(&Type::from(ty.data_ptr_type()?)),
                    REINTERPRET.get()?,
                    vec![call.args.get(0)?],
                    None,
                    Vec::<Type>::new(),
                    Location::unknown(),
                )?;
                let declaration = declaration(&var, pointer.into())?;
                let binding = Bind::new(var, &declaration, &op.loc)?;
                let result = self.mutate_decl_tensor(&binding, &declaration, m)?;
                return Ok(result.unwrap_or(binding).into());
            }
        }
        m.default_mutate(self, &op)
    }

    /// The logical tensor stays until the access linearizes its indices.
    fn mutate_store(&mut self, op: TensorStore, m: &mut Mutator) -> Result<Any> {
        let value: Expr = m.mutate_as(self, &op.value)?;
        let indices: Array<Expr> = m.mutate_as(self, &op.indices)?;
        let tensor: Var = op.dest.clone().try_cast()?;
        let indices = self.simplified_elem_offset(&mut Site::Body(&mut *m), &tensor, &indices)?;
        let flattened = self.flattened(&mut Site::Body(m), &tensor, false)?;
        Ok(TensorStore::from_complete_fields(op.loc.clone(), flattened.into(), indices, value)
            .into())
    }

    fn mutate_load(&mut self, op: TensorLoad, m: &mut Mutator) -> Result<Any> {
        let indices: Array<Expr> = m.mutate_as(self, &op.indices)?;
        let tensor: Var = op.source.clone().try_cast()?;
        let indices = self.simplified_elem_offset(&mut Site::Body(&mut *m), &tensor, &indices)?;
        let flattened = self.flattened(&mut Site::Body(m), &tensor, false)?;
        Ok(TensorLoad::new(flattened, indices.iter(), &op.loc)?.into())
    }
}

impl LayoutApplier {
    fn new(analyzer: Analyzer) -> Result<Self> {
        Ok(Self {
            base: AnalyzerBase::new(analyzer)?,
            remap: HashMap::default(),
            tensor_aliases: HashMap::default(),
        })
    }

    /// `Mutate(expr)` at `site`.
    fn mutate_expr(&mut self, site: &mut Site, expr: &Expr) -> Result<Expr> {
        match site {
            Site::Params => structural_mutate(expr.clone(), &mut *self)?.try_into(),
            Site::Body(m) => m.mutate_as(self, expr),
        }
    }

    /// `Mutate(AnyView(layout))` at `site`: every expression of the layout rewritten.
    fn mutate_layout(&mut self, site: &mut Site, layout: &Layout) -> Result<Layout> {
        match site {
            Site::Params => structural_mutate(layout.clone(), &mut *self)?.try_into(),
            Site::Body(m) => m.mutate_as(self, layout),
        }
    }

    fn mutate_alloc_tensor(&mut self, op: &Bind, call: &Call, m: &mut Mutator) -> Result<Any> {
        self.tensor_aliases.insert(ObjectIdentity::of(&op.var), op.var.clone());
        let tensor = self.flattened(&mut Site::Body(&mut *m), &op.var, true)?;
        let mut args: Vec<Expr> = call.args.iter().collect();
        if let Some(first) = args.first_mut() {
            *first = m.mutate_as(self, &*first)?;
        }
        let ty = tensor.ty.clone();
        let value = Call::new(
            Some(&ty),
            ALLOC_TENSOR.get()?,
            &Array::new(args),
            call.attrs.as_ref(),
            &Array::new(vec![ty.clone()]),
            &call.loc,
        )?;
        Ok(Bind::new(tensor, value, &op.loc)?.into())
    }

    /// The rebuilt binding, or `None` when nothing changed.
    fn mutate_decl_tensor(
        &mut self,
        op: &Bind,
        call: &Call,
        m: &mut Mutator,
    ) -> Result<Option<Bind>> {
        let source = call.args.get(0)?;
        self.register_alias(&op.var, &source)?;
        let data: Expr = m.mutate_as(self, &source)?;
        let tensor = self.flattened(&mut Site::Body(&mut *m), &op.var, false)?;
        if tensor.same_as(&op.var) && data.same_as(&source) {
            return Ok(None);
        }
        let ty = tensor.ty.clone();
        let value = Call::new(
            Some(&ty),
            DECL_TENSOR.get()?,
            &Array::new(vec![data]),
            call.attrs.as_ref(),
            &Array::new(vec![ty.clone()]),
            &call.loc,
        )?;
        Ok(Some(Bind::new(tensor, value, &op.loc)?))
    }

    /// `RegisterTensorAlias`: a tensor declared over another tensor's data shares its root.
    fn register_alias(&mut self, tensor: &Var, data: &Expr) -> Result<()> {
        let mut root = tensor.clone();
        if let Some(call) = data.as_node::<CallObj>() {
            if TENSOR_DATA_PTR.is(&call.op) && call.args.len() == 1 {
                if let Ok(source) = call.args.get(0)?.try_cast::<Var>() {
                    if tensor_type(&source).is_ok() {
                        let Some(source_root) =
                            self.tensor_aliases.get(&ObjectIdentity::of(&source))
                        else {
                            tvm_ffi::bail!(
                                INTERNAL_ERROR,
                                "Check failed: (source_root.has_value()) is false: Tensor alias \
                                 source {} must be registered before its DeclTensor alias",
                                source.name.as_str()
                            );
                        };
                        root = source_root.clone();
                    }
                }
            }
        }
        self.tensor_aliases.insert(ObjectIdentity::of(tensor), root);
        Ok(())
    }

    /// `GetFlattenedTensor(buf, is_alloc)`: the storage tensor of `buf`. An allocation through a
    /// layout covers the layout's memory span; a view keeps a nonzero origin of a regular layout
    /// as a 1-d layout.
    fn flattened(&mut self, site: &mut Site, buf: &Var, is_alloc: bool) -> Result<Var> {
        if let Some(mapped) = self.remap.get(&ObjectIdentity::of(buf)) {
            return Ok(mapped.clone());
        }
        let buf_ty = tensor_type(buf)?;
        let layout = &buf_ty.layout;
        let tile = layout.as_ref().and_then(|l| l.clone().try_cast::<TileLayout>().ok());
        let compose = layout.as_ref().and_then(|l| l.clone().try_cast::<ComposeLayout>().ok());
        let (flattened, base_ty, shape) = match (is_alloc, tile, compose) {
            (true, Some(tile), _) => {
                // The allocation covers the memory-axis span, gaps included.
                let analyzer = Analyzer::new();
                let mut span = Ex::from(int32(1));
                for iter in tile.shard()?.iter().chain(tile.replica()?.iter()) {
                    if iter.axis.is_memory()? {
                        span += (Ex::from(&iter.extent) - 1) * &iter.stride;
                    }
                }
                let offset: Map<Axis, Expr> = tile.offset()?;
                for (axis, offset) in offset.iter() {
                    if axis.is_memory()? {
                        span += &offset;
                    }
                }
                let span = analyzer.simplify(&span.expr()?, 2)?;
                (buf.clone(), buf_ty.clone(), Array::new(vec![span]))
            }
            (true, None, Some(compose)) => {
                // A swizzle permutes addresses: round the memory span to a block whose upper bits
                // the swizzle cannot change.
                let tile_layout = compose.tile_layout()?;
                let mut span: Expr = if tile_layout.is_trivial()? {
                    let mut span = Ex::from(int32(1));
                    for extent in buf_ty.shape.iter() {
                        span *= &extent;
                    }
                    span.expr()?
                } else {
                    Layout::from(tile_layout).span(Some("m"))?
                };
                let swizzle_len = compose.swizzle_len()?;
                if swizzle_len != 0 {
                    let mut changed_bits = compose.per_element()? + swizzle_len;
                    if !compose.swizzle_inner()? {
                        changed_bits += compose.atom_len()?;
                    }
                    let block: Expr = int_imm(span.prim_dtype()?, 1i128 << changed_bits)?.into();
                    span = ((Ex::from(&span) + &block - 1).floordiv(&block) * &block).expr()?;
                }
                let span = self.base.analyzer.simplify(&span, 2)?;
                (buf.clone(), buf_ty.clone(), Array::new(vec![span]))
            }
            _ => {
                let analyzer = self.base.analyzer.clone();
                let (flattened, flat_ty) = flattened_tensor(buf, &buf_ty, Some(&analyzer), false)?;
                let shape = flat_ty.shape.clone();
                (flattened, flat_ty, shape)
            }
        };
        // Variables the pass already rebuilt are remapped (a shape may load from another local
        // tensor), then the shape is canonicalized.
        let mut new_shape = Vec::with_capacity(shape.len());
        for extent in shape.iter() {
            let extent = self.mutate_expr(site, &extent)?;
            new_shape.push(self.base.analyzer.canonical_simplify(&extent)?);
        }
        // A regular view keeps its origin in its flat memory layout; an allocation has the origin
        // in its span already and uses absolute indices.
        let mut layout = None;
        if !is_alloc {
            if let Some((_, _, origin)) = match_shape_stride_pattern(&buf_ty)? {
                let origin = self.mutate_expr(site, &origin)?;
                let origin = self.base.analyzer.canonical_simplify(&origin)?;
                if !is_const_int(&origin, 0) {
                    let memory_axis = Axis::get("m")?;
                    let shard =
                        Iter::new(new_shape[0].clone(), int32(1)?.into(), memory_axis.clone());
                    let offset: Map<Axis, Expr> = [(memory_axis, origin)].into_iter().collect();
                    layout = Some(TileLayout::new([shard], Vec::<Iter>::new(), &offset).into());
                }
            }
        }
        let ty = TensorType::from_complete_fields(
            base_ty.loc.clone(),
            base_ty.dtype.clone(),
            base_ty.scope.clone(),
            Array::new(new_shape),
            base_ty.data_alignment,
            layout,
        );
        if structural_equal(&buf_ty, &ty)? {
            return Ok(buf.clone());
        }
        let flattened = Var::new(&flattened.name, Type::from(ty), &flattened.loc);
        self.remap.insert(ObjectIdentity::of(buf), flattened.clone());
        Ok(flattened)
    }

    /// `GetSimplifiedElemOffset`: the element offset of an access, through the tensor's layout
    /// when it has one. A layout whose origin the flat view keeps lowers only the relative
    /// indices.
    fn simplified_elem_offset(
        &mut self,
        site: &mut Site,
        tensor: &Var,
        indices: &Array<Expr>,
    ) -> Result<Array<Expr>> {
        let ty = tensor_type(tensor)?;
        if let Some(layout) = &ty.layout {
            if let Ok(tile) = layout.clone().try_cast::<TileLayout>() {
                for iter in tile.shard()?.iter() {
                    if iter.axis.is_thread()? {
                        tvm_ffi::bail!(
                            INTERNAL_ERROR,
                            "Cannot lower direct TensorLoad/TensorStore on a tensor with \
                             thread-axis layout: unable to verify that the coordinate matches the \
                             current thread. Use .view() + .local() to decompose thread and \
                             memory axes."
                        );
                    }
                }
            }
            let mut layout = self.mutate_layout(site, layout)?;
            let flat = self.flattened(site, tensor, false)?;
            let Some((_, _, origin)) = match_shape_stride_pattern(&tensor_type(&flat)?)? else {
                tvm_ffi::bail!(INTERNAL_ERROR, "bad optional access");
            };
            if !is_const_int(&origin, 0) {
                // The flat view retains this origin, so only lower its relative indices.
                let tile: TileLayout = layout.try_cast()?;
                layout = TileLayout::new(tile.shard()?.iter(), tile.replica()?.iter(), &Map::new())
                    .into();
            }
            let offsets = layout.canonicalize()?.apply_with_shape(indices, &ty.shape)?;
            if offsets.is_empty() && ty.shape.is_empty() {
                return Ok(Array::new(vec![int_imm(default_index_type(&ty)?, 0)?.into()]));
            }
            check_cmp!(
                offsets.len(), ==, 1, INTERNAL_ERROR, "res.size() == 1",
                "Expected a single element offset"
            );
            let (_, offset) = offsets.iter().next().expect("one offset");
            return Ok(Array::new(vec![self.base.analyzer.simplify(&offset, 2)?]));
        }
        let flattened = elem_offset(&ty, indices)?;
        Ok(Array::new(vec![self.base.analyzer.simplify(&flattened.get(0)?, 2)?]))
    }
}

/// `Call(t.ty, decl_tensor, {data}, {}, {t.ty})`.
fn declaration(tensor: &Var, data: Expr) -> Result<Call> {
    let ty = tensor.ty.clone();
    Call::new(
        Some(&ty),
        DECL_TENSOR.get()?,
        &Array::new(vec![data]),
        None,
        &Array::new(vec![ty.clone()]),
        Location::unknown(),
    )
}

/// C++ `LayoutApplier::Flatten(body, params, target)`: the flattened body and the parameters.
fn flatten(body: SeqStmt, params: &Array<Var>) -> Result<(Stmt, Array<Var>)> {
    let mut applier = LayoutApplier::new(Analyzer::new())?;
    let mut new_params = Vec::with_capacity(params.len());
    let mut aliased: Vec<(Var, Var)> = Vec::new();
    for param in params.iter() {
        let Some((tensor, ty)) = as_tensor_var(&param) else {
            new_params.push(param);
            continue;
        };
        applier.tensor_aliases.insert(ObjectIdentity::of(&tensor), tensor.clone());
        let flattened = applier.flattened(&mut Site::Params, &tensor, false)?;
        let mut source = tensor.clone();
        if ty.layout.is_some() && ty.has_default_layout(Some(&applier.base.analyzer))? {
            let without_layout = TensorType::from_complete_fields(
                ty.loc.clone(),
                ty.dtype.clone(),
                ty.scope.clone(),
                ty.shape.clone(),
                ty.data_alignment,
                None,
            );
            source = Var::new(&tensor.name, Type::from(without_layout), &tensor.loc);
        }
        if !flattened.same_as(&source) {
            aliased.push((flattened, source.clone()));
        }
        new_params.push(source);
    }
    let mut stmt: Stmt = structural_mutate(body, &mut applier)?.try_into()?;
    for (buf, source) in aliased {
        let declaration = declaration(&buf, source.tensor_data()?)?;
        stmt = SeqStmt::new(
            vec![Stmt::from(Bind::new(buf, declaration, Location::unknown())?), stmt],
            Location::unknown(),
        )
        .into();
    }
    Ok((stmt, Array::new(new_params)))
}

pub fn lower_tensor_layouts() -> Result<Pass> {
    function_pass("tirx.LowerTensorLayouts", 0, |func, _module, _context| {
        let Some(body) = func.body.clone() else { return Ok(Some(func)) };
        // `ResolveTarget`: the function's target, or the current one, which must exist.
        if get_attr(&func.attrs, "target")?.is_none() {
            Target::current(false)?;
        }
        let (body, params) = flatten(body, &func.params)?;
        let body = SeqStmt::from_stmt(body)?;
        Ok(Some(func.with(|node| {
            node.body = Some(body);
            node.params = params;
        })))
    })
}
