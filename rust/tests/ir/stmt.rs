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

//! Tests of `ir::stmt`.

use tvm_tirx::ir::*;
use tvm_tirx::tvm_ffi::{Any, Map, String as FfiString};

use super::{dtype, int32, int32_var, loc, op, seq, tensor_var};
use crate::common::{serial, structural_equal};

#[test]
fn statements() {
    let _serial = serial();
    let x = int32_var("x");
    let y = int32_var("y");
    let bind = Bind::new(&y, op("prim._OpAdd", &x, int32(1)), loc()).unwrap();
    assert!(structural_equal(&bind.var, &y));
    let evaluate = Evaluate::new(&x, loc()).unwrap();
    let ret = Return::new(&x, loc()).unwrap();
    assert!(structural_equal(&ret.value, &x));

    let body = seq([Stmt::from(&bind), Stmt::from(&evaluate)]);
    assert_eq!(body.seq.len(), 2);
    let nested = SeqStmt::new([Stmt::from(&body), Stmt::from(&ret)], Some(&loc())).unwrap();
    assert_eq!(nested.seq.len(), 3);

    let kind = ForKind::Unrolled;
    let annotations: Map<FfiString, Any> =
        [(FfiString::from("note"), Any::from(1i64))].into_iter().collect();
    let for_ =
        For::new(&x, int32(0), int32(16), int32(1), kind, &evaluate, Some(&annotations), loc())
            .unwrap();
    assert_eq!(for_.kind, kind as i32);
    assert_eq!(for_.body.seq.len(), 1);
    let less = op("prim._OpLT", &x, int32(4));
    let while_ = While::checked(&less, &body, loc()).unwrap();
    assert_eq!(while_.body.seq.len(), 2);
    let if_ = If::new(&less, &evaluate, Some(&Stmt::from(&ret)), loc()).unwrap();
    assert!(if_.else_case.is_some());
    assert!(If::new(&less, &evaluate, None, loc()).unwrap().else_case.is_none());

    let message = [StringImm::new("boom", loc()).unwrap()];
    let kind_name = StringImm::new("RuntimeError", loc()).unwrap();
    let assert_ = AssertStmt::new(&less, &kind_name, &message, loc()).unwrap();
    assert_eq!(assert_.message_parts.len(), 1);

    let a = tensor_var("A");
    let one = FloatImm::from_dtype(dtype("float32"), 1.0, loc()).unwrap();
    let store = TensorStore::new(&a, [&x], &one, loc()).unwrap();
    assert_eq!(store.indices.len(), 1);

    let attrs = DictAttrs::new(Map::new());
    let region = RegionStmt::new(
        Op::get("tirx.device_scope").unwrap(),
        Vec::<Expr>::new(),
        Vec::<Var>::new(),
        &attrs,
        &evaluate,
        Vec::<Var>::new(),
        loc(),
    )
    .unwrap();
    assert_eq!(region.body.seq.len(), 1);
}
