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

//! `tirx.MakePackedAPI` (`src/tirx/transform/make_packed_api.cc`): give each function with a
//! global symbol the packed calling convention `(self, args, num_args, result) -> int32`. The
//! arguments are decoded and checked by [`super::tvm_ffi_binder`], a `return` writes the packed
//! result, and calls between such functions become `call_cpacked`.

use crate::fast_hash::HashMap;

use tvm_ffi::tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;
use tvm_ffi::{
    dispatch, structural_mutate, Any, Array, Map, Mutator, ObjectIdentity, ObjectRefCast,
    ObjectRefCore, Result, String, Unchanged, INTERNAL_ERROR,
};

use super::analyzer::is_const_int;
use super::tvm_ffi_binder::{
    i32_imm, scalar, AbiBinder, FFI_ANY_TYPE_INDEX, FFI_ANY_UNION_VALUE, FFI_ANY_ZERO_PADDING,
};
use super::util::{MutatorExt, dict_with, get_attr, module_pass, module_with_function};
use crate::ir::{
    Bind, Call, DictAttrs, Evaluate, Expr, For, GlobalVarObj, If, IntImm, PtrType, PtrTypeObj,
    RegionStmt, Return, SeqStmt, Stmt, StringImm, Type, Var,
};
use crate::ops::{int_imm, repr};
use crate::ir::prim::Cast;
use crate::target::Target;
use crate::tirx::op::{
    ABI_FIELD_SET, CALL_CPACKED, CALL_EXTERN, COMPUTE_SCOPE, DEVICE_CONTEXT, PARALLEL_LAUNCH,
    THROW_LAST_ERROR,
};
use crate::tirx::Function;
use crate::ir::transform::Pass;
use crate::ir::prim::op::PrimOps;
use crate::ir::Location;
use crate::tirx::op::reinterpret;

const FOR_PARALLEL: i32 = 1;
/// `CallingConv::kCPackedFunc`.
const CALLING_CONV_PACKED: i64 = 1;
const DL_CPU: i64 = 1;
/// `ffi::symbol::tvm_ffi_symbol_prefix`.
const SYMBOL_PREFIX: &str = "__tvm_ffi_";

const INT: u8 = 0;
const UINT: u8 = 1;
const FLOAT: u8 = 2;
const BOOL: u8 = 6;

/// An integer attribute: an `int`, or an `IntImm` as C++ `TypeTraits` also accepts.
fn int_attr(value: Any) -> Result<i64> {
    if let Some(value) = value.try_as::<i64>() {
        return Ok(value);
    }
    let imm: IntImm = value.try_into()?;
    Ok(imm.value.to_i64().unwrap_or(0))
}

/// `ReturnRewriter`: `return v` writes `v` to the packed result and returns 0.
struct ReturnRewriter {
    result: Var,
    in_parallel: i32,
}

#[dispatch(mutate)]
impl ReturnRewriter {
    /// Expressions are not entered.
    fn mutate_expr(&mut self, _expr: Expr) -> Unchanged {
        Unchanged
    }

    /// A parallel loop, other than an unannotated single iteration, runs on a thread pool.
    fn mutate_for(&mut self, op: For, m: &mut Mutator) -> Result<Any> {
        let trivial_step = is_const_int(&op.step, 1);
        let single = is_const_int(&op.extent()?, 1) && trivial_step && op.annotations.is_empty();
        let parallel = (op.kind == FOR_PARALLEL && !single) as i32;
        self.in_parallel += parallel;
        let result = m.default_mutate(self, &op);
        self.in_parallel -= parallel;
        result
    }

    fn mutate_region(&mut self, op: RegionStmt, m: &mut Mutator) -> Result<Any> {
        let parallel = PARALLEL_LAUNCH.is(&op.op) as i32;
        self.in_parallel += parallel;
        let result = m.default_mutate(self, &op);
        self.in_parallel -= parallel;
        result
    }

    fn mutate_return(&mut self, op: Return) -> Result<Any> {
        check_cmp!(
            self.in_parallel,
            ==,
            0,
            INTERNAL_ERROR,
            "in_parallel_ == 0",
            "Return cannot be used in parallel scope."
        );
        Ok(self.write_to_out(op.value.clone())?.into())
    }
}

impl ReturnRewriter {
    /// `WriteToOut`: store the value's type index, zero padding and value, then return 0.
    fn write_to_out(&self, value: Expr) -> Result<SeqStmt> {
        let (type_index, value) = convert_for_ffi(value)?;
        let store = |field: i32, value: Expr| -> Result<Stmt> {
            let args = vec![Expr::from(&self.result), i32_imm(0)?, i32_imm(field as i64)?, value];
            let call = Call::new(
                Some(&Type::from(scalar(INT, 32)?)),
                ABI_FIELD_SET.get()?,
                &args,
                None,
                Vec::<Type>::new(),
                Location::unknown(),
            )?;
            Ok(Evaluate::new(call, Location::unknown()).into())
        };
        Ok(SeqStmt::new(
            vec![
                store(FFI_ANY_TYPE_INDEX, i32_imm(type_index)?)?,
                store(FFI_ANY_ZERO_PADDING, i32_imm(0)?)?,
                store(FFI_ANY_UNION_VALUE, value)?,
                Stmt::from(Return::new(0, Location::unknown())),
            ],
            Location::unknown(),
        ))
    }
}

/// `ConvertForFFI`: the FFI type index of a returned value, and the value as the FFI stores it.
fn convert_for_ffi(value: Expr) -> Result<(i64, Expr)> {
    if value.ty.as_node::<PtrTypeObj>().is_some() {
        return Ok((TypeIndex::kTVMFFIOpaquePtr as i64, value));
    }
    let dtype = value.prim_dtype()?;
    let widened = |code: u8, bits: u8| -> Result<Expr> {
        Ok(Cast::new(scalar(code, bits)?, &value, Location::unknown())?.into())
    };
    Ok(match dtype.code {
        BOOL => (TypeIndex::kTVMFFIBool as i64, widened(INT, 64)?),
        INT | UINT => (TypeIndex::kTVMFFIInt as i64, widened(INT, 64)?),
        FLOAT => (TypeIndex::kTVMFFIFloat as i64, widened(FLOAT, 64)?),
        3 if dtype.bits == 0 && dtype.lanes == 0 => (TypeIndex::kTVMFFINone as i64, value),
        _ => {
            let dtype = tvm_ffi::DLDataTypeExt::to_string(&dtype);
            tvm_ffi::bail!(INTERNAL_ERROR, "data type {} not supported yet", dtype.as_str());
        }
    })
}

/// `SubroutineCallRewriter`: a call to a function with a packed API calls it by its symbol.
struct SubroutineCallRewriter<'a> {
    methods: &'a HashMap<ObjectIdentity, String>,
    made_change: bool,
}

#[dispatch(mutate)]
impl SubroutineCallRewriter<'_> {
    fn mutate_call(&mut self, op: Call, m: &mut Mutator) -> Result<Any> {
        let call: Call = m.default_mutate_as(self, &op)?;
        if call.op.as_node::<GlobalVarObj>().is_some() {
            if let Some(symbol) = self.methods.get(&ObjectIdentity::of(&call.op)) {
                let mut args: Vec<Expr> =
                    vec![StringImm::new(symbol.as_str(), Location::unknown()).into()];
                args.extend(call.args.iter());
                // An empty handle, for the current `call_cpacked` convention.
                let zero: Expr = int_imm(scalar(UINT, 64)?.dtype, 0)?.into();
                let null_handle = reinterpret(PtrType::void()?, zero)?;
                args.push(null_handle);
                self.made_change = true;
                return Ok(Call::new(
                    Some(&Type::from(&call.ty)),
                    CALL_CPACKED.get()?,
                    &args,
                    None,
                    Vec::<Type>::new(),
                    Location::unknown(),
                )?
                .into());
            }
        }
        Ok(call.into())
    }
}

impl SubroutineCallRewriter<'_> {
    /// `Apply`: the statement with its calls rewritten, if any was.
    fn apply(methods: &HashMap<ObjectIdentity, String>, stmt: SeqStmt) -> Result<Option<Stmt>> {
        let mut rewriter = SubroutineCallRewriter { methods, made_change: false };
        let stmt: Stmt = structural_mutate(stmt, &mut rewriter)?.try_into()?;
        Ok(rewriter.made_change.then_some(stmt))
    }
}

/// `RequiresPackedAPI`: the global symbol of a function that should get the packed API.
fn requires_packed_api(func: &Function) -> Result<Option<String>> {
    // A function with an explicit calling convention has already been lowered.
    if let Some(convention) = get_attr(&func.attrs, "calling_conv")? {
        if int_attr(convention)? != 0 {
            return Ok(None);
        }
    }
    match get_attr(&func.attrs, "global_symbol")? {
        Some(symbol) => Ok(Some(String::try_from(symbol)?)),
        None => Ok(None),
    }
}

fn no_attrs() -> DictAttrs {
    DictAttrs::new(Map::new())
}

/// `MakePackedAPI(func)`.
fn make_packed_function(func: Function) -> Result<Function> {
    let Some(body) = func.body.clone() else { return Ok(func) };
    let Some(symbol) = requires_packed_api(&func)? else { return Ok(func) };
    let name_hint = symbol.as_str().to_string();
    let Some(target) = get_attr(&func.attrs, "target")? else {
        tvm_ffi::bail!(
            INTERNAL_ERROR,
            "Check failed: (opt) is false: MakePackedAPI required the function to be annotated \
             with tvm::attr::kTarget (target), but the function only has attributes {}",
            repr(&func.attrs)
        );
    };
    let target: Target = target.try_into()?;
    let target_device_type = target.get_target_device_type()? as i64;
    // A function without a host target has already been lowered.
    let Some(target_host) = target.host()? else { return Ok(func) };
    let target_host: Target = target_host.try_cast()?;

    let int32: Type = scalar(INT, 32)?.into();
    let self_handle = Var::new("self_handle", Type::from(PtrType::void()?), Location::unknown());
    let packed_args = Var::new("args", Type::from(PtrType::void()?), Location::unknown());
    let num_packed_args = Var::new("num_args", Type::from(&int32), Location::unknown());
    let result = Var::new("result", Type::from(PtrType::void()?), Location::unknown());
    let device_id = Var::new("dev_id", Type::from(&int32), Location::unknown());
    let device_type: Expr = int_imm(scalar(INT, 32)?.dtype, target_device_type as i128)?.into();

    let mut binder = AbiBinder::new(
        &name_hint,
        &func.params,
        &packed_args,
        &num_packed_args,
        &device_type,
        &device_id,
    )?;
    binder.decode_all_params()?;
    let mut binding = binder.finalize()?;
    let need_set_device = binding.var_defs.contains_key(&ObjectIdentity::of(&device_id));

    let attrs = dict_with(&func.attrs, "calling_conv", Any::from(CALLING_CONV_PACKED));
    let attrs = dict_with(&attrs, "target", Any::from(target_host));
    let attrs = dict_with(
        &attrs,
        "global_symbol",
        Any::from(String::from(format!("{SYMBOL_PREFIX}{name_hint}"))),
    );

    let mut rewriter = ReturnRewriter { result: result.clone(), in_parallel: 0 };
    let body: Stmt = structural_mutate(body, &mut rewriter)?.try_into()?;
    let scope: Expr = StringImm::new(&format!("{name_hint}_compute_"), Location::unknown()).into();
    let mut body: Stmt = RegionStmt::new(
        COMPUTE_SCOPE.get()?,
        &Array::new(vec![scope]),
        Vec::<Var>::new(),
        no_attrs(),
        body,
        Vec::<Var>::new(),
        Location::unknown(),
    )?
    .into();
    if need_set_device && target_device_type != DL_CPU {
        let status = Var::new("set_device_status", Type::from(&int32), Location::unknown());
        let set_device_args = vec![
            StringImm::new("TVMBackendSetDevice", Location::unknown()).into(),
            device_type.clone(),
            Expr::from(&device_id),
        ];
        let set_device = Call::new(
            Some(&Type::from(&int32)),
            CALL_EXTERN.get()?,
            &set_device_args,
            None,
            Vec::<Type>::new(),
            Location::unknown(),
        )?;
        let set_device: Stmt = Bind::new(status.clone(), set_device, Location::unknown())?.into();
        let raise = Call::new(
            Some(&Type::from(&int32)),
            THROW_LAST_ERROR.get()?,
            Vec::<Expr>::new(),
            None,
            Vec::<Type>::new(),
            Location::unknown(),
        )?;
        let raise = SeqStmt::new(
            vec![Stmt::from(Evaluate::new(raise, Location::unknown()))],
            Location::unknown(),
        );
        let check_status: Stmt =
            If::new(status.not_equal(i32_imm(0)?).expr()?, raise, None, Location::unknown())?
                .into();
        body = SeqStmt::new(vec![set_device, check_status, body], Location::unknown()).into();
    }
    // An error code of zero on success.
    body = SeqStmt::new(
        vec![body, Stmt::from(Return::new(0, Location::unknown()))],
        Location::unknown(),
    )
    .into();
    binding.decl_tensors.push(body);
    body = SeqStmt::new(binding.decl_tensors, Location::unknown()).into();
    if !binding.asserts.is_empty() {
        binding.asserts.push(body);
        body = SeqStmt::new(binding.asserts, Location::unknown()).into();
    }
    if need_set_device {
        let args = Array::new(vec![device_type, device_id.into()]);
        let context = DEVICE_CONTEXT.get()?;
        body = RegionStmt::new(
            context,
            &args,
            Vec::<Var>::new(),
            no_attrs(),
            body,
            Vec::<Var>::new(),
            Location::unknown(),
        )?
        .into();
    }
    if !binding.init_nest.is_empty() {
        binding.init_nest.push(body);
        body = SeqStmt::new(binding.init_nest, Location::unknown()).into();
    }
    let body = SeqStmt::from_stmt(body)?;
    let params = Array::new(vec![self_handle, packed_args, num_packed_args, result]);

    let undefined = crate::tirx::analysis::undefined_vars(Stmt::from(&body), params.iter())?;
    check_cmp!(
        undefined.len(),
        ==,
        0,
        INTERNAL_ERROR,
        "undefined.size() == 0",
        "In Function {} variables {} are used, but are not passed in as API arguments",
        name_hint,
        repr(&undefined)
    );
    Ok(func.with_attrs(attrs).with(|node| {
        node.body = Some(body);
        node.params = params;
        node.ret_type = int32;
    }))
}

pub fn make_packed_api() -> Result<Pass> {
    module_pass("tirx.MakePackedAPI", 0, |module, _context| {
        let mut methods = HashMap::default();
        for (gvar, base_func) in module.functions.iter() {
            if let Ok(func) = base_func.try_cast::<Function>() {
                if let Some(symbol) = requires_packed_api(&func)? {
                    methods.insert(ObjectIdentity::of(&gvar), symbol);
                }
            }
        }
        let mut updates = Vec::new();
        for (gvar, base_func) in module.functions.iter() {
            let Ok(func) = base_func.try_cast::<Function>() else { continue };
            let Some(body) = func.body.clone() else { continue };
            let original = func.clone();
            let mut func = func;
            if let Some(body) = SubroutineCallRewriter::apply(&methods, body)? {
                func = func.with_body(Some(SeqStmt::from_stmt(body)?));
            }
            let func = make_packed_function(func)?;
            if !func.same_as(&original) {
                updates.push((gvar, func));
            }
        }
        let mut module = module;
        for (gvar, func) in updates {
            module = module_with_function(&module, &gvar, func);
        }
        Ok(module)
    })
}
