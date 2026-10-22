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

//! Tests of `ir::op`.

use tvm_tirx::ir::{ArgumentInfo, ArgumentInfoObj, ExprObj, Op, OpObj};
use tvm_tirx::tvm_ffi::tvm_ffi_sys::{TVMFFIByteArray, TVMFFITypeKeyToIndex};
use tvm_tirx::tvm_ffi::{Any, Array, FieldGetter, Object, ObjectCore, String as FfiString, TypeIndex};

use crate::common::{assert_type_contracts, global, serial};

#[test]
fn generated_types_match_the_registry() {
    let _serial = serial();
    assert_type_contracts!(true; ArgumentInfoObj: Object, OpObj: ExprObj);
}

#[test]
fn ops() {
    let _serial = serial();
    let op_names = Op::list_op_names().unwrap();
    assert!(op_names.iter().any(|name| name.as_str() == "prim.exp"));
    assert!(!Op::list_attr_names().unwrap().is_empty());
    let effect_kinds = Op::attr_map::<i64>("TCallEffectKind").unwrap().unwrap();
    assert!(effect_kinds.get(&Op::get("prim.exp").unwrap()).unwrap().is_some());
    assert!(Op::attr_map::<i64>("no_such_attr").unwrap().is_none());
    Op::register("rust.tests.ir.op", "An operator the tests register.").unwrap();
    assert_eq!(Op::get("rust.tests.ir.op").unwrap().name.as_str(), "rust.tests.ir.op");
    assert!(Op::register("rust.tests.ir.op", "Twice.").is_err());
    // The map is live: it sees the operator registered after it was read, which has no value.
    assert!(effect_kinds.get(&Op::get("rust.tests.ir.op").unwrap()).unwrap().is_none());
}

/// Read a reflected field of `op` through the C ABI getter, bypassing the Rust layout.
fn reflected<T>(op: &Op, field: &str) -> T
where
    T: TryFrom<Any>,
    T::Error: std::fmt::Debug,
{
    let value = FieldGetter::new(OpObj::type_index(), field).unwrap().get_any(&**op).unwrap();
    T::try_from(value).unwrap()
}

fn argument_names(arguments: &Array<ArgumentInfo>) -> Vec<std::string::String> {
    arguments.iter().map(|argument| argument.name.as_str().to_owned()).collect()
}

#[test]
fn hand_written_op_layout_matches_native() {
    let _serial = serial();
    let names: Array<FfiString> =
        global("ir.ListOpNames").call_packed(&[]).unwrap().try_into().unwrap();
    assert!(!names.is_empty());

    let (mut with_attrs, mut with_validator) = (0, 0);
    for name in names.iter() {
        let op = Op::get(name.as_str()).unwrap();

        // Reflected members: the direct read agrees with the native getter.
        assert_eq!(op.name.as_str(), name.as_str());
        assert_eq!(i64::from(op.index), reflected::<i64>(&op, "index"), "{name}: index");
        assert_eq!(op.name.as_str(), reflected::<FfiString>(&op, "name").as_str());
        assert_eq!(op.doc.as_str(), reflected::<FfiString>(&op, "doc").as_str(), "{name}: doc");
        assert_eq!(
            op.attrs_type_key.as_str(),
            reflected::<FfiString>(&op, "attrs_type_key").as_str(),
            "{name}: attrs_type_key"
        );
        for (field, direct) in [("args_info", &op.args_info), ("ty_args_info", &op.ty_args_info)] {
            let native = reflected::<Array<ArgumentInfo>>(&op, field);
            assert_eq!(argument_names(direct), argument_names(&native), "{name}: {field}");
        }
        for (field, direct) in
            [("var_args_info", &op.var_args_info), ("var_ty_args_info", &op.var_ty_args_info)]
        {
            let native = reflected::<Option<ArgumentInfo>>(&op, field);
            assert_eq!(
                direct.as_ref().map(|argument| argument.name.as_str().to_owned()),
                native.as_ref().map(|argument| argument.name.as_str().to_owned()),
                "{name}: {field}"
            );
        }

        // Unreflected members: nothing native reads them back, so check their invariants.
        if op.attrs_type_key.as_str().is_empty() {
            assert_eq!(op.attrs_type_index(), 0, "{name}: attrs_type_index");
        } else {
            let key = unsafe { TVMFFIByteArray::from_str(op.attrs_type_key.as_str()) };
            let mut expected = 0;
            assert_eq!(unsafe { TVMFFITypeKeyToIndex(&key, &mut expected) }, 0);
            // Native registration leaves the index at 0 for an operator defined before its
            // attrs type's index is initialized, so only a non-zero index is comparable.
            if op.attrs_type_index() != 0 {
                with_attrs += 1;
                assert_eq!(op.attrs_type_index() as i32, expected, "{name}: attrs_type_index");
            }
        }
        let validator = op.validator().type_index();
        if validator == TypeIndex::kTVMFFINone as i32 {
            assert!(!op.validator_is_custom(), "{name}: custom validator without a validator");
        } else {
            // A native adapter is stored as its function address, a packed one as a Function.
            with_validator += 1;
            assert!(
                validator == TypeIndex::kTVMFFIOpaquePtr as i32
                    || validator == TypeIndex::kTVMFFIFunction as i32,
                "{name}: validator is neither a native nor a packed function \
                 (type index {validator})"
            );
        }
    }
    // Both unreflected members were exercised with a non-default value.
    assert!(with_attrs > 0, "no operator has an attrs type index");
    assert!(with_validator > 0, "no operator has a validator");

    // The registry hands out one instance per name.
    let first = names.get(0).unwrap();
    let (lhs, rhs) = (Op::get(first.as_str()).unwrap(), Op::get(first.as_str()).unwrap());
    assert!(std::ptr::eq::<OpObj>(&*lhs, &*rhs));
    assert!(Op::get("ir.no_such_operator").is_err());
}

#[test]
fn op_attr_map_reads_the_live_registry_column() {
    let _serial = serial();
    assert!(Op::has_attr_map("TCallEffectKind").unwrap());
    assert!(!Op::has_attr_map("test.rust.MissingColumn").unwrap());
    assert!(Op::attr_map::<i64>("test.rust.MissingColumn").unwrap().is_none());

    // Every lookup agrees with the reflected per-Op `get_attr`.
    let effect = Op::attr_map::<i64>("TCallEffectKind").unwrap().unwrap();
    let get_attr =
        tvm_tirx::tvm_ffi::Function::from_type_method(OpObj::type_index(), "get_attr").unwrap();
    let names: Array<FfiString> =
        global("ir.ListOpNames").call_packed(&[]).unwrap().try_into().unwrap();
    let mut with_effect = 0;
    for name in names.iter() {
        let op = Op::get(name.as_str()).unwrap();
        let native: Any =
            get_attr.call_tuple((op.clone(), FfiString::from("TCallEffectKind"))).unwrap();
        let native = (native.type_index() != TypeIndex::kTVMFFINone as i32)
            .then(|| i64::try_from(native).unwrap());
        assert_eq!(effect.get(&op).unwrap(), native, "{name}");
        with_effect += usize::from(native.is_some());
    }
    assert!(with_effect > 0, "no operator has a TCallEffectKind");

    // A map taken earlier sees later registrations, including the column's growth.
    let key = FfiString::from("test.rust.LiveColumn");
    let register = |name: &str, value: i64| {
        global("ir.RegisterOp").call_tuple((FfiString::from(name), FfiString::from(""))).unwrap();
        let op = Op::get(name).unwrap();
        tvm_tirx::tvm_ffi::Function::from_type_key_method("ir.Op", "_set_attr")
            .unwrap()
            .call_tuple((op.clone(), key.clone(), value, false))
            .unwrap();
        op
    };
    let first = register("test.rust.live_column.first", 7);
    let column = Op::attr_map::<i64>(key.as_str()).unwrap().unwrap();
    assert_eq!(column.get(&first).unwrap(), Some(7));
    let second = register("test.rust.live_column.second", 8);
    assert_eq!(column.get(&second).unwrap(), Some(8));
    assert_eq!(column.get(&Op::get("prim.exp").unwrap()).unwrap(), None);

    // A value of another type is an error, not a silent miss.
    let as_string = Op::attr_map::<FfiString>(key.as_str()).unwrap().unwrap();
    assert!(as_string.get(&first).is_err());
}
