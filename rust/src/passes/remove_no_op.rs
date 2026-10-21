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

//! `tirx.RemoveNoOp` (`src/tirx/transform/remove_no_op.cc`): drop statements without effect,
//! empty loops and branches, and stores of the value already in place.

use crate::fast_hash::HashMap;

use tvm_ffi::{
    dispatch, Any, Array, Map, Mutator, ObjectIdentity, ObjectRefCast, ObjectRefCore, Result,
    String,
};

use super::analyzer::{is_const_int, AnalyzerBase, AnalyzerScope, AnalyzerState};
use super::side_effect::{side_effect, READ_STATE};
use super::util::{function_pass, MutatorExt};
use crate::ir::TensorLoad;

use crate::ir::{
    CallObj, Evaluate, EvaluateObj, Expr, For, If, IfObj, IntImmObj, Op, PrimTypeObj, Range,
    SeqStmt, SeqStmtObj, Stmt, TensorLoadObj, TensorStore, Var,
};
use crate::sym::{neg_inf, pos_inf, Analyzer, IntSet};
use crate::tirx::transform::RemoveNoOpConfig;
use crate::ir::transform::Pass;
use crate::ir::prim::op::PrimOps;
use crate::ir::Location;

struct NoOpRemover {
    base: AnalyzerBase,
    /// The range of each enclosing loop's variable (`var_range_map_`).
    loop_ranges: HashMap<ObjectIdentity, (Var, IntSet)>,
    /// Profiler calls, which count as no-ops when `ignore_profiler_call` is set.
    profiler_ops: Vec<Op>,
}

impl AnalyzerState for NoOpRemover {
    fn base(&mut self) -> &mut AnalyzerBase {
        &mut self.base
    }
}

#[dispatch(mutate, policy = AnalyzerScope)]
impl NoOpRemover {
    /// A branch whose arms do nothing; an empty arm is dropped, negating the condition if needed.
    fn mutate_if(&mut self, op: If, m: &mut Mutator) -> Result<Stmt> {
        let stmt: Stmt = m.default_mutate_as(self, &Stmt::from(op))?;
        let Some(op) = stmt.as_node::<IfObj>() else { return Ok(stmt) };
        let then_is_no_op = is_no_op(&op.then_case);
        let else_is_no_op = op.else_case.as_ref().is_none_or(is_no_op);
        let rewritten = match (then_is_no_op, else_is_no_op, &op.else_case) {
            (true, true, _) => Some(evaluate(&op.condition)?),
            (false, true, Some(_)) => {
                Some(If::new(&op.condition, &op.then_case, None, Location::unknown())?.into())
            }
            (true, false, Some(else_case)) => Some(
                If::new((!&op.condition).expr()?, else_case, None, Location::unknown())?.into(),
            ),
            _ => None,
        };
        Ok(rewritten.unwrap_or(stmt))
    }

    /// A loop that provably runs zero times, or whose body does nothing.
    fn mutate_for(&mut self, op: For, m: &mut Mutator) -> Result<Stmt> {
        let max = self.extent_range(&op.extent()?)?.max()?;
        if !max.same_as(&pos_inf()?) && !max.same_as(&neg_inf()?) {
            let empty = max.less_equal(0).expr()?;
            if self.base.analyzer.can_prove(&empty, crate::sym::ProofStrength::Default)? {
                return evaluate_zero();
            }
        }
        let key = ObjectIdentity::of(&op.var);
        self.loop_ranges.insert(key.clone(), (op.var.clone(), loop_range(&op)?));
        let stmt = m.default_mutate_as(self, &Stmt::from(op));
        self.loop_ranges.remove(&key);
        let stmt: Stmt = stmt?;
        let result: For = stmt.clone().try_cast()?;
        if is_const_int(&result.extent()?, 0) {
            return evaluate_zero();
        }
        if is_no_op(&result.body) {
            return evaluate_effects(&[
                result.begin.clone(),
                result.end.clone(),
                result.step.clone(),
            ]);
        }
        Ok(stmt)
    }

    /// An evaluation whose value has no effect.
    fn mutate_evaluate(&mut self, op: Evaluate) -> Result<Stmt> {
        if self.has_side_effect(&op.value)? {
            return Ok(op.into());
        }
        evaluate_zero()
    }

    /// A store of the value already in place keeps only the effects of its operands.
    fn mutate_store(&mut self, store: TensorStore, m: &mut Mutator) -> Result<Any> {
        if !self.stores_existing_value(&store)? {
            return Ok(store.into());
        }
        let mut effects = vec![evaluate(&store.value)?];
        for index in store.indices.iter() {
            effects.push(evaluate(&index)?);
        }
        m.mutate(self, &seq_stmt(effects)?)
    }
}

impl NoOpRemover {
    /// `sym::EvalSet(extent, var_range_map_)`.
    fn extent_range(&self, extent: &Expr) -> Result<IntSet> {
        let dom_map: Map<Var, IntSet> = self.loop_ranges.values().cloned().collect();
        Analyzer::new().int_set(extent, Some(&dom_map))
    }

    fn has_side_effect(&self, value: &Expr) -> Result<bool> {
        if value.ty.as_node::<PrimTypeObj>().is_none() {
            return Ok(value.as_node::<CallObj>().is_some());
        }
        if let Some(call) = value.as_node::<CallObj>() {
            if self.profiler_ops.iter().any(|op| call.op.same_as(op)) {
                return Ok(false);
            }
        }
        Ok(side_effect(value)? > READ_STATE)
    }

    fn stores_existing_value(&self, store: &TensorStore) -> Result<bool> {
        let dest: Var = store.dest.clone().try_cast()?;
        let load = TensorLoad::new(&dest, store.indices.iter(), Location::unknown())?;
        let unchanged = (&store.value - load).equal(0).expr()?;
        if is_const_int(&self.base.analyzer.simplify(&unchanged, 2)?, 1) {
            return Ok(true);
        }
        let Some(load) = store.value.as_node::<TensorLoadObj>() else { return Ok(false) };
        if !load.source.same_as(&dest) {
            return Ok(false);
        }
        // The same tensor on both sides: compare the indices.
        self.all_equal(&load.indices, &store.indices)
    }

    fn all_equal(&self, a: &Array<Expr>, b: &Array<Expr>) -> Result<bool> {
        if a.len() != b.len() {
            return Ok(false);
        }
        for (x, y) in a.iter().zip(b.iter()) {
            if !self.base.analyzer.can_prove_equal(&x, &y)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// `IntSet::FromRange(Range(begin, end))` for a loop.
fn loop_range(op: &For) -> Result<IntSet> {
    let range = Range::new(&op.begin, &op.end, Location::unknown())?;
    if is_const_int(&range.extent, 1) {
        return IntSet::single_point(&range.begin);
    }
    // C++ `IntervalSet(begin, extent + begin - 1)`.
    let max = (&range.extent + &range.begin - 1).expr()?;
    Ok(IntSet::from(crate::sym::IntervalSet::new(&range.begin, &max)))
}

/// `tirx::IsNoOp`: an integer `Evaluate` or an empty sequence.
fn is_no_op<S: ObjectRefCore>(stmt: &S) -> bool {
    if let Some(eval) = stmt.as_node::<EvaluateObj>() {
        return eval.value.as_node::<IntImmObj>().is_some();
    }
    matches!(stmt.as_node::<SeqStmtObj>(), Some(seq) if seq.seq.is_empty())
}

fn evaluate_zero() -> Result<Stmt> {
    Ok(Evaluate::new(0, Location::unknown()).into())
}

/// `MakeEvaluate(value)`: the evaluation if it has an effect, else a no-op.
fn evaluate(value: &Expr) -> Result<Stmt> {
    if side_effect(value)? > READ_STATE {
        return Ok(Evaluate::new(value, Location::unknown()).into());
    }
    evaluate_zero()
}

/// `MakeEvaluate(values)`: the evaluations that have an effect, in sequence.
fn evaluate_effects(values: &[Expr]) -> Result<Stmt> {
    let mut effects = Vec::new();
    for value in values {
        if side_effect(value)? > READ_STATE {
            effects.push(Evaluate::new(value, Location::unknown()).into());
        }
    }
    Ok(seq_stmt(effects)?.into())
}

/// `SeqStmt(stmts)`, which flattens and drops no-ops.
fn seq_stmt(stmts: Vec<Stmt>) -> Result<SeqStmt> {
    Ok(SeqStmt::new(stmts, Location::unknown()))
}

pub fn remove_no_op() -> Result<Pass> {
    function_pass("tirx.RemoveNoOp", 0, |func, _module, context| {
        let Some(body) = func.body.clone() else { return Ok(Some(func)) };
        // `GetConfigOrDefault<RemoveNoOpConfig>`: the reflection defaults are 0 and false.
        let (max_steps, ignore_profiler_call) =
            match context.config.get(&String::from("tirx.RemoveNoOp"))? {
                Some(value) => {
                    let config: RemoveNoOpConfig = value.try_into()?;
                    (config.max_simplification_steps, config.ignore_profiler_call)
                }
                None => (0, false),
            };
        let analyzer = Analyzer::new();
        analyzer.set_maximum_rewrite_steps(max_steps)?;
        let profiler_ops = if ignore_profiler_call {
            [
                "tirx.cuda.timer_init",
                "tirx.cuda.timer_start",
                "tirx.cuda.timer_end",
                "tirx.cuda.timer_finalize",
            ]
            .into_iter()
            .map(Op::get)
            .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let mut remover = NoOpRemover {
            base: AnalyzerBase::new(analyzer)?,
            loop_ranges: HashMap::default(),
            profiler_ops,
        };
        let body: Stmt = tvm_ffi::structural_mutate(body, &mut remover)?.try_into()?;
        Ok(Some(func.with_body(Some(SeqStmt::from_stmt(body)?))))
    })
}
