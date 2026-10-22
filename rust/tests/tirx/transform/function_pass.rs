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

//! Tests of `tirx::transform::function_pass`.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tvm_tirx::ir::transform::{Pass, PassInfo, PassObj};
use tvm_tirx::ir::{BaseFunc, Evaluate, GlobalInfo, GlobalVar, IRModule, IntImm, SeqStmt, Stmt};
use tvm_tirx::tirx;
use tvm_tirx::tirx::transform::{FunctionPass, FunctionPassObj};
use tvm_tirx::tvm_ffi::{
    Any, AnyView, Array, BigInt, Function as FfiFunction, Map, ObjectRefCast, RValueRef, Result,
    String as FfiString,
};

use crate::common::{assert_type_contracts, global, serial, structural_equal};
use crate::{evaluate_body, function, ir, module_of, s, void_function};

#[test]
fn generated_types_match_the_registry() {
    let _serial = serial();
    // Opaque bindings embed only their parent; the native object is larger.
    assert_type_contracts!(false; FunctionPassObj: PassObj);
}

#[test]
fn function_passes() {
    let _serial = serial();
    let func = void_function(vec![]);
    let runs = Arc::new(AtomicUsize::new(0));
    let counter = runs.clone();
    let counting = FunctionPass::new(
        move |func, _module, _ctx| {
            counter.fetch_add(1, Ordering::Relaxed);
            Ok(Some(func))
        },
        PassInfo::new(0, "rust.Count".into()),
    );
    let out = Pass::from(&counting).run(module_of("main", &func)).unwrap();
    assert_eq!(runs.load(Ordering::Relaxed), 1);
    assert!(out.contains("main").unwrap());
    let dropping =
        FunctionPass::new(|_func, _module, _ctx| Ok(None), PassInfo::new(0, "rust.Drop".into()));
    assert!(!Pass::from(&dropping)
        .run(module_of("main", &func))
        .unwrap()
        .contains("main")
        .unwrap());
}

#[test]
fn a_rust_closure_runs_as_a_function_pass() {
    let _serial = serial();
    let ir = ir();

    // The pass: rebuild the function with `evaluate(n)` turned into `evaluate(n + 1)`.
    let calls = Rc::new(Cell::new(0));
    let seen = calls.clone();
    let pass_func = FfiFunction::from_packed(move |args: &[AnyView]| -> Result<Any> {
        let func: tirx::Function = RValueRef::<tirx::Function>::try_from(args[0])?.into_inner();
        seen.set(seen.get() + 1);
        let body = func.body.clone().expect("the function has a body");
        let stmt: Evaluate = body.seq.get(0)?.try_cast()?;
        let value: IntImm = stmt.value.clone().try_cast()?;
        let bumped = IntImm::new(
            value.loc.clone(),
            value.ty.clone(),
            BigInt::from(value.value.to_i64().expect("a small constant") + 1),
        );
        let stmt = Evaluate::from_complete_fields(stmt.loc.clone(), bumped.into());
        let body =
            SeqStmt::from_complete_fields(body.loc.clone(), Array::<Stmt>::new(vec![stmt.into()]));
        let rewritten = tirx::Function::from_complete_fields(
            func.loc.clone(),
            func.ty.clone(),
            func.attrs.clone(),
            func.params.clone(),
            func.ret_type.clone(),
            Some(body),
        );
        Ok(Any::from(rewritten))
    });

    // A Rust-allocated PassInfo names the pass; the native factory wraps the closure.
    let pass: FunctionPass = global("tirx.transform.CreateFunctionPass")
        .call_tuple((pass_func, PassInfo::new(0, s("rust.BumpEvaluate"))))
        .unwrap()
        .try_into()
        .unwrap();
    let info = pass.pass_info().unwrap();
    assert_eq!(info.name.as_str(), "rust.BumpEvaluate");
    assert_eq!(info.opt_level, 0);

    let main: GlobalVar =
        global("ir.GlobalVar").call_tuple((s("main"), ir.loc.clone())).unwrap().try_into().unwrap();
    let functions: Map<GlobalVar, BaseFunc> =
        [(main.clone(), BaseFunc::from(function(&ir, &evaluate_body(&ir, 41))))]
            .into_iter()
            .collect();
    let module: IRModule = global("ir.IRModule")
        .call_packed(&[
            AnyView::from(&functions),
            AnyView::new(),
            AnyView::from(&Map::<FfiString, Array<GlobalInfo>>::new()),
        ])
        .unwrap()
        .try_into()
        .unwrap();

    let lowered: IRModule =
        global("transform.RunPass").call_tuple((pass, module)).unwrap().try_into().unwrap();
    assert_eq!(calls.get(), 1);
    let result: tirx::Function =
        lowered.functions.get(&main).unwrap().expect("main survives").try_cast().unwrap();
    assert!(structural_equal(&result, &function(&ir, &evaluate_body(&ir, 42))));
    assert!(!structural_equal(&result, &function(&ir, &evaluate_body(&ir, 41))));
}
