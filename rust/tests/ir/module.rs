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

//! Tests of `ir::module`.

use tvm_tirx::ir::{BaseFunc, Expr, GlobalInfo, GlobalVar, IRModule};
use tvm_tirx::tvm_ffi::{Any, Map, String as FfiString};

use super::{function, int32, loc};
use crate::common::{load_tirx_compiler, serial, structural_equal};

#[test]
fn modules() {
    load_tirx_compiler();
    let _serial = serial();
    let func = function(0);
    let main = GlobalVar::new("main", loc()).unwrap();
    let functions: Map<GlobalVar, BaseFunc> = [(main.clone(), func.clone())].into_iter().collect();
    let module = IRModule::from_functions(&functions, None, &Map::new()).unwrap();
    assert!(module.contains("main").unwrap());
    assert!(module.contains_global_var(&main).unwrap());
    assert!(module.has_global_var("main").unwrap());
    assert!(structural_equal(&module.get_global_var("main").unwrap(), &main));
    assert_eq!(module.get_global_vars().unwrap().len(), 1);
    assert!(structural_equal(&module.lookup(&main).unwrap(), &func));
    assert!(structural_equal(&module.lookup_name("main").unwrap(), &func));

    let other = GlobalVar::new("other", loc()).unwrap();
    module.add(&other, function(1), false).unwrap();
    module.update_func(&other, function(3)).unwrap();
    assert!(structural_equal(&module.lookup(&other).unwrap(), &function(3)));
    module.remove("other").unwrap();
    assert!(!module.contains("other").unwrap());
    module.add(&other, function(1), false).unwrap();
    module.remove_global_var(&other).unwrap();
    assert!(!module.contains_global_var(&other).unwrap());

    let extra_var = GlobalVar::new("extra", loc()).unwrap();
    let extra_functions: Map<GlobalVar, BaseFunc> =
        [(extra_var.clone(), function(4))].into_iter().collect();
    module.update(&IRModule::from_functions(&extra_functions, None, &Map::new()).unwrap()).unwrap();
    assert!(module.contains("extra").unwrap());
    module.update_global_info("info", Vec::<GlobalInfo>::new()).unwrap();
    assert!(module.global_infos.get(&FfiString::from("info")).is_ok());

    let copy = module.shallow_copy().unwrap();
    copy.remove("extra").unwrap();
    assert!(module.contains("extra").unwrap());

    assert!(module.get_attr("note").unwrap().is_none());
    let module = module.with_attr("note", int32(7)).unwrap();
    let note = module.get_attr("note").unwrap().unwrap();
    assert!(structural_equal(&Expr::try_from(note).unwrap(), &int32(7)));
    let attrs: Map<FfiString, Any> =
        [(FfiString::from("other"), Any::from(int32(1)))].into_iter().collect();
    let module = module.with_attrs(&attrs).unwrap();
    assert!(module.get_attr("other").unwrap().is_some());
    let module = module.without_attr("note").unwrap();
    assert!(module.get_attr("note").unwrap().is_none());

    let from_expr = IRModule::from_expr(function(5), &Map::new()).unwrap();
    assert!(from_expr.contains("main").unwrap());
}
