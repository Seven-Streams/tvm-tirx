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

//! Tests of `ir::op_def`: operators registered from Rust, by an `OpDef` or on the first use of an
//! `OpHandle`, are used by the native registry, `Call` constructor and validation, and their type
//! inference, validators and region contracts are native hooks.
//!
//! The registry does not synchronize registration, so the tests take `serial()`; the two tests
//! register operators of different names.

use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

use tvm_tirx::ir::sig::{IntExpr, PrimExpr, PrimVar};
use tvm_tirx::ir::{
    register_ops, Attrs, Call, CallObj, DictAttrs, Evaluate, Expr, FuncType, IntImm, Op, OpDef,
    OpHandle, PrimType, RegionStmt, RegionStmtObj, Signature, StringImm, Type, Var, VarObj,
    UNKNOWN_LOC,
};
use tvm_tirx::tirx::{CallFFIKernelAttr, TensorVar};
use tvm_tirx::tvm_ffi::tvm_ffi_sys::TVMFFITypeIndex;
use tvm_tirx::tvm_ffi::{
    Any, Array, Error, ErrorKind, Map, ObjectRefCast, ObjectRefCore, Result, String as FfiString,
    ATTRIBUTE_ERROR, INTERNAL_ERROR, TYPE_ERROR, VALUE_ERROR,
};

use super::{int32, parse_dtype, tensor_var};
use crate::common::{global, serial, structural_equal};

fn var(name: &str, dtype: &str) -> Var {
    Var::new(name, PrimType::from_dtype(parse_dtype(dtype)).unwrap(), UNKNOWN_LOC)
}

/// `Call(op, args)` without a result type, which the constructor infers.
fn call(op: &Op, args: Vec<Expr>) -> Result<Call> {
    Call::new(None, op.clone(), args, None, Vec::<Type>::new(), UNKNOWN_LOC)
}

fn validate(call: &Call) -> Result<()> {
    global("ir.CallValidate").call_tuple((call.clone(),)).map(|_| ())
}

/// `Call(op, args)` with `ty_args`, `attrs` and a result type, so the constructor infers none.
fn call_with(op: &Op, args: Vec<Expr>, ty_args: Vec<Type>, attrs: Option<Attrs>) -> Call {
    let ty: Type = PrimType::int(32).into();
    Call::new(Some(&ty), op.clone(), args, attrs.as_ref(), ty_args, UNKNOWN_LOC).unwrap()
}

/// The message `Call::Validate` raises for `op(args)` with `ty_args` and `attrs`, with the name
/// of `op` replaced by `<op>`.
fn rejection(op: &Op, args: &[Expr], ty_args: &[Type], attrs: Option<Attrs>) -> String {
    let call = call_with(op, args.to_vec(), ty_args.to_vec(), attrs);
    let error = validate(&call).unwrap_err();
    assert_eq!(error.kind(), TYPE_ERROR, "{}", error.message());
    error.message().replace(&format!("`{}`", op.name.as_str()), "`<op>`")
}

fn assert_error(result: Result<impl Sized>, kind: ErrorKind, message: &str) {
    let Err(error) = result else { panic!("expected {kind:?}: {message}") };
    assert_eq!(error.kind(), kind, "{}", error.message());
    assert!(error.message().contains(message), "{}", error.message());
}

/// A Call's `args` and `ty_args`.
type Arguments = (Vec<Expr>, Vec<Type>);

/// A Call's `args` and `attrs`.
type AttrsCall = (Vec<Expr>, Option<Attrs>);

static INFERRED: AtomicUsize = AtomicUsize::new(0);

fn first_arg_type(call: &CallObj) -> Result<Type> {
    INFERRED.fetch_add(1, Ordering::Relaxed);
    Ok(call.args.get(0)?.ty.clone())
}

fn only_variables(call: &CallObj) -> Result<()> {
    match call.args.iter().all(|arg| arg.as_node::<VarObj>().is_some()) {
        true => Ok(()),
        false => Err(Error::new(VALUE_ERROR, "arguments must be variables", "")),
    }
}

fn faulty(_call: &CallObj) -> Result<()> {
    panic!("validator bug")
}

#[test]
fn ops_registered_from_rust_are_used_natively() {
    let _serial = serial();
    let (x, i, j) = (var("x", "float32"), var("i", "int32"), var("j", "int32"));
    let lanes = var("v", "int32x4");
    let text: Expr = StringImm::new("shared", UNKNOWN_LOC).into();
    let one: Expr = int32(1);
    let tensor = tensor_var("A");
    let int32_ty: Type = PrimType::int(32).into();

    // A new operator: its description and attributes are in the registry, and both of its hooks
    // are native function pointers.
    let op = OpDef::new("rust_test.identity", "Return the value.")
        .signature(Signature::new().arg::<Expr>("value", "The value."))
        .set_attr("TCallEffectKind", 0i64)
        .set_infer_type(first_arg_type)
        .op()
        .unwrap();
    assert!(op.same_as(&Op::get("rust_test.identity").unwrap()));
    assert_eq!(op.doc.as_str(), "Return the value.");
    assert_eq!(op.args_info.get(0).unwrap().name.as_str(), "value");
    let effect = Op::attr_map::<i64>("TCallEffectKind").unwrap().unwrap();
    assert_eq!(effect.get(&op).unwrap(), Some(0));
    let opaque = TVMFFITypeIndex::kTVMFFIOpaquePtr as i32;
    assert_eq!(op.validator().type_index(), opaque);
    assert!(!op.validator_is_custom());
    let infer = Op::attr_map::<*mut c_void>("FInferType").unwrap().unwrap();
    assert!(infer.get(&op).unwrap().is_some());

    // The `Call` constructor, in Rust and natively, and `ReinferType` infer the result type with
    // the Rust hook.
    let identity = call(&op, vec![x.clone().into()]).unwrap();
    assert!(structural_equal(&identity.ty, &x.ty));
    let reinferred: Type =
        global("ir.CallReinferType").call_tuple((identity.clone(),)).unwrap().try_into().unwrap();
    assert!(reinferred.same_as(&x.ty));
    let native: Call = global("ir.Call")
        .call_tuple((
            Option::<Type>::None,
            Expr::from(&op),
            Array::<Expr>::new(vec![x.clone().into()]),
            Option::<Attrs>::None,
            Array::<Type>::new(vec![]),
            tvm_tirx::ir::Location::unknown(),
        ))
        .unwrap()
        .try_into()
        .unwrap();
    assert!(native.ty.same_as(&x.ty));
    assert_eq!(INFERRED.load(Ordering::Relaxed), 3);
    validate(&identity).unwrap();

    // An error a Rust hook returns reaches the native caller with its kind and message, and a
    // panic becomes an error instead of unwinding into C++.
    let untyped = OpDef::new("rust_test.untyped", "")
        .set_infer_type(|_: &CallObj| Err(Error::new(TYPE_ERROR, "no result type", "")))
        .op()
        .unwrap();
    assert_error(call(&untyped, vec![]), TYPE_ERROR, "no result type");
    let faulty = OpDef::new("rust_test.faulty", "").set_validator(faulty).op().unwrap();
    assert_error(validate(&call(&faulty, vec![]).unwrap()), INTERNAL_ERROR, "validator bug");

    // A generated validator rejects what the C++ one generated from the same signature does,
    // with the same message.
    let cases: Vec<(&str, OpDef, Vec<Arguments>)> = vec![
        (
            "tirx.get_active_lane_mask",
            OpDef::new("rust_test.lane_mask", "")
                .signature(Signature::new().arg::<IntExpr>("base", "").arg::<IntExpr>("limit", "")),
            vec![
                (vec![], vec![]),
                (vec![x.clone().into(), i.clone().into()], vec![]),
                (vec![i.clone().into(), lanes.clone().into()], vec![]),
                (vec![i.clone().into(), i.clone().into(), i.clone().into()], vec![]),
            ],
        ),
        (
            "tirx.vector_high",
            OpDef::new("rust_test.vector_high", "")
                .signature(Signature::new().arg::<PrimExpr>("vec", "")),
            vec![(vec![text.clone()], vec![])],
        ),
        (
            "tirx.gpu_thread_filter",
            OpDef::new("rust_test.thread_filter", "")
                .signature(Signature::new().arg::<PrimVar>("var", "").arg::<PrimExpr>("pred", "")),
            vec![
                (vec![one.clone(), i.clone().into()], vec![]),
                (vec![j.clone().into(), text.clone()], vec![]),
            ],
        ),
        (
            "tirx.gpu_storage_sync",
            OpDef::new("rust_test.storage_sync", "")
                .signature(Signature::new().arg::<StringImm>("storage_scope", "")),
            vec![(vec![i.clone().into()], vec![]), (vec![text.clone()], vec![int32_ty.clone()])],
        ),
        (
            "tirx.masked_load",
            OpDef::new("rust_test.masked_load", "").signature(
                Signature::new()
                    .arg::<TensorVar>("tensor", "")
                    .arg::<PrimExpr>("index", "")
                    .var_args::<PrimExpr>("args", ""),
            ),
            vec![
                (vec![], vec![]),
                (vec![i.clone().into(), i.clone().into()], vec![]),
                (vec![one.clone(), i.clone().into()], vec![]),
                (vec![tensor.clone().into(), text.clone()], vec![]),
                (
                    vec![tensor.clone().into(), i.clone().into(), j.clone().into(), text.clone()],
                    vec![],
                ),
            ],
        ),
    ];
    for (native, rust, calls) in cases {
        let (native, rust) = (Op::get(native).unwrap(), rust.op().unwrap());
        for (args, ty_args) in calls {
            let (rust, native) = (
                rejection(&rust, &args, &ty_args, None),
                rejection(&native, &args, &ty_args, None),
            );
            assert_eq!(rust, native);
        }
    }

    // A tensor variable with any number of trailing values passes both.
    for name in ["tirx.masked_load", "rust_test.masked_load"] {
        let args = vec![tensor.clone().into(), i.clone().into(), j.clone().into()];
        validate(&call_with(&Op::get(name).unwrap(), args, vec![], None)).unwrap();
    }

    // Variadic tails and type arguments.
    let tails = OpDef::new("rust_test.tails", "")
        .signature(
            Signature::new()
                .arg::<PrimExpr>("index", "")
                .var_args::<PrimExpr>("rest", "")
                .ty_arg::<PrimType>("T", "")
                .var_ty_args::<Type>("more", ""),
        )
        .op()
        .unwrap();
    let func_ty: Type = FuncType::new(Vec::<Type>::new(), PrimType::int(32)).into();
    validate(&call_with(
        &tails,
        vec![i.clone().into(), one.clone()],
        vec![int32_ty.clone(), func_ty.clone()],
        None,
    ))
    .unwrap();
    assert_eq!(
        rejection(&tails, &[], std::slice::from_ref(&int32_ty), None),
        "Op `<op>`: Call.args expected at least 1 argument, got 0"
    );
    assert_eq!(
        rejection(&tails, &[i.clone().into()], &[], None),
        "Op `<op>`: Call.ty_args expected at least 1 type argument, got 0"
    );
    assert_eq!(
        rejection(
            &tails,
            &[i.clone().into(), one.clone(), text.clone()],
            std::slice::from_ref(&int32_ty),
            None,
        ),
        "Op `<op>`: `Call.args[2]` (`rest`) expected `ir.PrimExpr`, got `ir.StringImm[ty=ir.StringType]`."
    );
    assert_eq!(
        rejection(&tails, &[i.clone().into()], std::slice::from_ref(&func_ty), None),
        "Op `<op>`: `Call.ty_args[0]` (`T`) expected `ir.PrimType`, got `ir.FuncType`."
    );

    // `call_attrs`: the operator's attrs type, and the check that `Call.attrs` holds one.
    let dict: Attrs = DictAttrs::new(Map::new()).into();
    let attrs_cases: Vec<(&str, OpDef, Vec<AttrsCall>)> = vec![
        (
            "tirx.device_entry",
            OpDef::new("rust_test.device_entry", "").signature(
                Signature::new().var_args::<Expr>("launch_values", "").call_attrs::<DictAttrs>(),
            ),
            vec![(vec![], None), (vec![i.clone().into()], None)],
        ),
        (
            "tirx.call_ffi_kernel",
            OpDef::new("rust_test.call_ffi_kernel", "").signature(
                Signature::new()
                    .arg::<Expr>("kernel", "")
                    .var_args::<Expr>("args", "")
                    .call_attrs::<CallFFIKernelAttr>(),
            ),
            vec![
                (vec![], Some(dict.clone())),
                (vec![x.clone().into()], Some(dict.clone())),
                (vec![x.clone().into(), i.clone().into()], None),
            ],
        ),
    ];
    for (native, rust, calls) in attrs_cases {
        let (native, rust) = (Op::get(native).unwrap(), rust.op().unwrap());
        assert_eq!(rust.attrs_type_key.as_str(), native.attrs_type_key.as_str());
        assert_eq!(rust.attrs_type_index(), native.attrs_type_index());
        for (args, attrs) in calls {
            let (rust, native) = (
                rejection(&rust, &args, &[], attrs.clone()),
                rejection(&native, &args, &[], attrs),
            );
            assert_eq!(rust, native);
        }
    }
    for name in ["tirx.device_entry", "rust_test.device_entry"] {
        let entry =
            call_with(&Op::get(name).unwrap(), vec![i.clone().into()], vec![], Some(dict.clone()));
        validate(&entry).unwrap();
    }
    // A signature without `call_attrs` leaves the operator without an attrs type.
    let plain = OpDef::extend("rust_test.device_entry")
        .signature(Signature::new().var_args::<Expr>("launch_values", ""))
        .op()
        .unwrap();
    assert_eq!(plain.attrs_type_key.as_str(), "");

    // A custom validator replaces the generated one, and a later signature keeps it.
    let checked = OpDef::new("rust_test.checked", "Take variables only.")
        .signature(Signature::new().arg::<Expr>("value", ""))
        .set_validator(only_variables)
        .signature(Signature::new().arg::<Expr>("value", "").var_args::<Expr>("rest", ""))
        .op()
        .unwrap();
    assert!(checked.validator_is_custom());
    assert_eq!(checked.validator().type_index(), opaque);
    assert_eq!(checked.var_args_info.as_ref().unwrap().name.as_str(), "rest");
    validate(&call(&checked, vec![x.clone().into(), x.clone().into()]).unwrap()).unwrap();
    assert_error(
        validate(&call(&checked, vec![one.clone()]).unwrap()),
        VALUE_ERROR,
        "arguments must be variables",
    );
    // The Rust `Call::validate` runs the same hook.
    assert_error(
        call(&checked, vec![one.clone()]).unwrap().validate(),
        VALUE_ERROR,
        "arguments must be variables",
    );

    // Registering a name twice, an attribute twice, or a second custom validator is an error.
    assert_error(
        OpDef::new("rust_test.identity", "").op(),
        ATTRIBUTE_ERROR,
        "is registered before",
    );
    assert_error(
        OpDef::extend("rust_test.identity").set_attr("TCallEffectKind", 1i64).op(),
        VALUE_ERROR,
        "Attribute TCallEffectKind of rust_test.identity is already registered",
    );
    assert_error(
        OpDef::extend("rust_test.checked").set_validator(only_variables).op(),
        VALUE_ERROR,
        "Validator of rust_test.checked is already registered",
    );
    assert_error(
        OpDef::extend("rust_test.missing").op(),
        ATTRIBUTE_ERROR,
        "Operator rust_test.missing is not registered",
    );

    // Replacing and removing an attribute show in a column read before.
    OpDef::extend("rust_test.identity").replace_attr("TCallEffectKind", 1i64).op().unwrap();
    assert_eq!(effect.get(&op).unwrap(), Some(1));
    OpDef::extend("rust_test.identity").reset_attr("TCallEffectKind").op().unwrap();
    assert_eq!(effect.get(&op).unwrap(), None);

    // Argument descriptions and the attrs type.
    let described = OpDef::new("rust_test.described", "")
        .add_arg("lhs", "Left.")
        .add_ty_arg("T", "Result type.")
        .attrs_type_key("ir.DictAttrs")
        .op()
        .unwrap();
    assert_eq!(described.args_info.len(), 1);
    assert_eq!(described.ty_args_info.get(0).unwrap().name.as_str(), "T");
    assert_eq!(described.attrs_type_key.as_str(), "ir.DictAttrs");

    // A native operator takes a new attribute column.
    let likely = OpDef::extend("prim.likely")
        .set_attr("rust_test.Note", FfiString::from("hint"))
        .op()
        .unwrap();
    let note = Op::attr_map::<FfiString>("rust_test.Note").unwrap().unwrap();
    assert_eq!(note.get(&likely).unwrap().unwrap().as_str(), "hint");
}

/// The type of the first argument, as `first_arg_type` but without counting the inference.
fn identity_type(call: &CallObj) -> Result<Type> {
    Ok(call.args.get(0)?.ty.clone())
}

fn accept(_call: &CallObj) -> Result<()> {
    Ok(())
}

static IDENTITY: OpHandle =
    OpHandle::define("rust_test.handle_identity", "Return the value.", |def| {
        def.signature(Signature::new().arg::<Expr>("value", "The value."))
            .set_attr("TCallEffectKind", 0i64)
            .set_infer_type(identity_type)
    });
static LAUNCH_THREAD: OpHandle = OpHandle::registered("tirx.launch_thread");
static UNREGISTERED: OpHandle = OpHandle::registered("rust_test.unregistered");
/// A name C++ registers: Rust does not take it over.
static TAKEN: OpHandle = OpHandle::define("tirx.launch_thread", "", |def| def);
/// A definition that fails part way: a second custom validator.
static BROKEN: OpHandle =
    OpHandle::define("rust_test.broken", "", |def| def.set_validator(accept).set_validator(accept));
/// `tirx.launch_thread`'s region contract (`src/tirx/op/region.cc`), written in Rust: one thread
/// index of the extent's type.
static LAUNCH: OpHandle =
    OpHandle::define("rust_test.launch", "Bind a thread index below an extent.", |def| {
        def.signature(Signature::new().arg::<StringImm>("tag", "").arg::<IntExpr>("extent", ""))
            .set_region_body_params(launch_body_params)
            .set_region_validate(no_results_or_attributes)
    });
static PANICKING: OpHandle = OpHandle::define("rust_test.panicking_region", "", |def| {
    def.set_region_body_params(|_: &CallObj| -> Result<Array<Var>> { panic!("hook bug") })
});

fn launch_body_params(call: &CallObj) -> Result<Array<Var>> {
    let tag = call.args.get(0)?.try_cast::<StringImm>()?;
    let extent_ty: PrimType = call.args.get(1)?.ty.clone().try_cast()?;
    let dtype = extent_ty.dtype;
    if tag.value.as_str().is_empty() {
        let message = "Check failed: (!tag->value.empty()) is false: launch_thread expects a \
                       nonempty StringImm thread tag";
        return Err(Error::new(VALUE_ERROR, message, ""));
    }
    if dtype.bits <= 1 {
        let message = format!(
            "Check failed: dtype.bits() > 1 ({} vs. 1) : launch_thread extent must have a scalar \
             integer type wider than one bit",
            dtype.bits
        );
        return Err(Error::new(VALUE_ERROR, &message, ""));
    }
    Ok(Array::new(vec![Var::new("", extent_ty, UNKNOWN_LOC)]))
}

fn no_results_or_attributes(region: &RegionStmtObj) -> Result<()> {
    if !region.result_vars.is_empty() || !region.attrs.dict.is_empty() {
        let message = format!(
            "Check failed: (region->result_vars.empty() && region->attrs->dict.empty()) is false: \
             {} expects no results or attributes",
            region.op.name.as_str()
        );
        return Err(Error::new(VALUE_ERROR, &message, ""));
    }
    Ok(())
}

/// `RegionStmt(op, {tag, extent}, body_params, attrs, Evaluate(0), result_vars)`, its error text
/// with the operator's name taken out.
fn region(
    op: &Op,
    tag: &str,
    extent: Expr,
    body_params: Vec<Var>,
    attrs: Vec<(&str, i64)>,
    result_vars: Vec<Var>,
) -> std::result::Result<(), (String, String)> {
    let args: Vec<Expr> = vec![StringImm::new(tag, UNKNOWN_LOC).into(), extent];
    let attrs: Map<FfiString, Any> =
        attrs.into_iter().map(|(key, value)| (FfiString::from(key), Any::from(value))).collect();
    let body = Evaluate::new(0, UNKNOWN_LOC);
    let built = RegionStmt::new(
        op.clone(),
        args,
        body_params,
        DictAttrs::new(attrs),
        body,
        result_vars,
        UNKNOWN_LOC,
    );
    built.map(|_| ()).map_err(|error| {
        let message = error.message().replace(op.name.as_str(), "<op>");
        (error.kind().to_string(), message)
    })
}

#[test]
fn rust_defined_operators() {
    let _serial = serial();

    // A handle registers its operator on first use, once; registering it again does nothing.
    assert!(Op::get("rust_test.handle_identity").is_err(), "not registered before first use");
    let identity = IDENTITY.get().unwrap();
    assert!(identity.same_as(&Op::get("rust_test.handle_identity").unwrap()));
    assert_eq!(identity.doc.as_str(), "Return the value.");
    assert!(IDENTITY.get().unwrap().same_as(&identity));
    register_ops(&[&IDENTITY, &LAUNCH_THREAD]).unwrap();
    register_ops(&[&IDENTITY, &LAUNCH_THREAD]).unwrap();
    let x = var("x", "int32");
    let call = Call::new(
        None,
        IDENTITY.get().unwrap(),
        vec![Expr::from(&x)],
        None,
        Vec::<Type>::new(),
        UNKNOWN_LOC,
    )
    .unwrap();
    assert!(call.ty.same_as(&x.ty), "the Rust FInferType types the Call");
    assert!(IDENTITY.is(&call.op));
    assert!(!LAUNCH_THREAD.is(&call.op));

    // Another thread sees the same operator.
    let name = std::thread::spawn(|| IDENTITY.get().unwrap().name.as_str().to_string());
    assert_eq!(name.join().unwrap(), "rust_test.handle_identity");

    // A handle to an operator registered elsewhere looks it up.
    assert!(LAUNCH_THREAD.get().unwrap().same_as(&Op::get("tirx.launch_thread").unwrap()));
    assert!(UNREGISTERED.get().is_err());
    assert!(!UNREGISTERED.is(&call.op));

    // A name registered elsewhere is not taken over; the first caller gets the registry's error,
    // later ones an error that names the operator.
    assert_error(TAKEN.register(), ATTRIBUTE_ERROR, "is registered before");
    assert_error(TAKEN.get(), INTERNAL_ERROR, "Operator tirx.launch_thread failed to register");
    assert_error(BROKEN.get(), VALUE_ERROR, "Validator of rust_test.broken is already registered");
    assert_error(
        register_ops(&[&IDENTITY, &BROKEN]),
        INTERNAL_ERROR,
        "Operator rust_test.broken failed to register: ValueError",
    );

    // The region hooks are native function pointers.
    let launch = LAUNCH.get().unwrap();
    for key in ["FRegionGetBodyParams", "FRegionValidate"] {
        let hooks = Op::attr_map::<*mut c_void>(key).unwrap().unwrap();
        assert!(hooks.get(&launch).unwrap().is_some(), "{key}");
    }

    // The `RegionStmt` constructor checks a region of the Rust operator as it checks one of
    // `tirx.launch_thread`: the same outcome, and the same error but for the operator's name.
    let native = LAUNCH_THREAD.get().unwrap();
    let extent: Expr = int32(8);
    let wide: Expr = IntImm::from_dtype(parse_dtype("int64"), 8, UNKNOWN_LOC).unwrap().into();
    let flag: Expr = IntImm::bool(true).into();
    let i = var("i", "int32");
    let cases = vec![
        ("a well-formed region", "threadIdx.x", extent.clone(), vec![i.clone()], vec![], vec![]),
        ("no body parameter", "threadIdx.x", extent.clone(), vec![], vec![], vec![]),
        ("a body parameter of another type", "threadIdx.x", wide, vec![i.clone()], vec![], vec![]),
        ("an empty tag", "", extent.clone(), vec![i.clone()], vec![], vec![]),
        ("a one-bit extent", "threadIdx.x", flag, vec![var("b", "bool")], vec![], vec![]),
        ("an attribute", "threadIdx.x", extent.clone(), vec![i.clone()], vec![("k", 1)], vec![]),
        (
            "a result",
            "threadIdx.x",
            extent.clone(),
            vec![i.clone()],
            vec![],
            vec![var("r", "int32")],
        ),
        ("a repeated variable", "threadIdx.x", extent, vec![i.clone()], vec![], vec![i]),
    ];
    for (what, tag, extent, params, attrs, results) in cases {
        let rust =
            region(&launch, tag, extent.clone(), params.clone(), attrs.clone(), results.clone());
        let cpp = region(&native, tag, extent, params, attrs, results);
        assert_eq!(rust, cpp, "{what}");
    }

    // A panic in a hook becomes an error instead of unwinding into C++.
    let panicking = PANICKING.get().unwrap();
    let body = Evaluate::new(0, UNKNOWN_LOC);
    let empty = DictAttrs::new(Map::new());
    let built = RegionStmt::new(
        panicking,
        Vec::<Expr>::new(),
        Vec::<Var>::new(),
        empty,
        body,
        Vec::<Var>::new(),
        UNKNOWN_LOC,
    );
    assert!(built.is_err(), "a panicking hook fails the constructor");
}
