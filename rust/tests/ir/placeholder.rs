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

//! Tests of the [`Placeholder`]s of the IR, which `ir::type_def` implements: each is the node its
//! C++ constructor builds from the same arguments, and a type of a test that extends an IR object
//! and holds IR fields is made from them.

use tvm_tirx::ir::{
    dtype, prim, register_type, AnyType, BaseFunc, Call, Constant, DataTypeImm, DictAttrs, Expr,
    ExprObj, FloatImm, FuncType, GenericConst, GlobalVar, IntImm, LambdaExpr, Location,
    MissingType, MissingTypeObj, NoneType, OpaqueExpr, OpaqueExprObj, OpaqueType, PrimType,
    PtrType, StagingExpr, StringImm, StringType, TensorLoad, TensorRegion, TensorRegionType, Tuple,
    TupleGetItem, TupleType, Type, Var, VarObj,
};
use tvm_tirx::runtime::{MemoryScope, MemoryScopeObj};
use tvm_tirx::tirx::{self, TensorMapType, TensorType};
use tvm_tirx::tvm_ffi::{
    self, def_object_ref, rs_class, Any, AnyView, Array, FieldGetter, Function, Map, ObjectCore,
    ObjectRefCore, Placeholder, String as FfiString,
};

use crate::common::{agree, agree_fresh, global, serial, structural_equal};

fn loc() -> Location {
    Location::unknown()
}

fn zero() -> Expr {
    IntImm::int32(0).into()
}

/// The name a memory scope is registered under.
fn scope_name(scope: &MemoryScope) -> String {
    let name = FieldGetter::new(MemoryScopeObj::type_index(), "_str_index").unwrap();
    let name: FfiString = name.get_any(&**scope).unwrap().try_into().unwrap();
    name.as_str().to_owned()
}

#[test]
fn types_agree_with_cpp() {
    let _serial = serial();
    let none = Any::default();
    let no_types = Array::<Type>::default();
    let call = |name: &str, args: &[AnyView]| global(name).call_packed(args);

    let missing = call("ir.MissingType", &[]);
    agree("Type", Ok(Type::placeholder()), missing.clone());
    agree("MissingType", Ok(MissingType::placeholder()), missing);
    agree("NoneType", Ok(NoneType::placeholder()), call("ir.NoneType", &[]));
    agree("AnyType", Ok(AnyType::placeholder()), call("ir.AnyType", &[(&loc()).into()]));
    agree("OpaqueType", Ok(OpaqueType::placeholder()), call("ir.OpaqueType", &[]));
    agree("StringType", Ok(StringType::placeholder()), call("ir.StringType", &[]));
    let region = call("ir.TensorRegionType", &[]);
    agree("TensorRegionType", Ok(TensorRegionType::placeholder()), region);
    let void = dtype::void();
    agree("PrimType", Ok(PrimType::placeholder()), call("ir.PrimType", &[(&void).into()]));

    // The global memory, the default C++ gives a pointer and a tensor.
    let scope = MemoryScope::placeholder();
    assert_eq!(scope_name(&scope), "global");
    assert!(PtrType::placeholder().scope.same_as(&scope));
    let void_ty = PrimType::void();
    let native_ptr = call("ir.PtrType", &[(&void_ty).into(), (&scope).into()]);
    agree("PtrType", Ok(PtrType::placeholder()), native_ptr);
    let empty = call("ir.TupleType", &[(&no_types).into(), (&loc()).into()]);
    agree("TupleType", Ok(TupleType::placeholder()), empty.clone());
    let unit = Type::try_from(empty.unwrap()).unwrap();
    let native_func = call("ir.FuncType", &[(&no_types).into(), (&unit).into()]);
    agree("FuncType", Ok(FuncType::placeholder()), native_func);
    let native_map = call("tirx.TensorMapType", &[(&loc()).into()]);
    agree("TensorMapType", Ok(TensorMapType::placeholder()), native_map);
    let (shape, alignment) = (Array::<Expr>::default(), 0i32);
    let native_tensor = call(
        "tirx.TensorType",
        &[
            (&shape).into(),
            (&void_ty).into(),
            (&scope).into(),
            (&alignment).into(),
            (&none).into(),
            (&loc()).into(),
        ],
    );
    agree("TensorType", Ok(TensorType::placeholder()), native_tensor);
}

#[test]
fn expressions_agree_with_cpp() {
    let _serial = serial();
    let none = Any::default();
    let no_exprs = Array::<Expr>::default();
    let call = |name: &str, args: &[AnyView]| global(name).call_packed(args);
    let empty = FfiString::from("");

    let (int32, zero_value) = (dtype::int(32), 0i64);
    let native_zero = call("ir.IntImm", &[(&int32).into(), (&zero_value).into(), (&loc()).into()]);
    agree("Expr", Ok(Expr::placeholder()), native_zero.clone());
    agree("Constant", Ok(Constant::placeholder()), native_zero.clone());
    agree("IntImm", Ok(IntImm::placeholder()), native_zero);
    let (float32, zero_float) = (dtype::float(32), 0.0f64);
    let native_float =
        call("ir.FloatImm", &[(&float32).into(), (&zero_float).into(), (&loc()).into()]);
    agree("FloatImm", Ok(FloatImm::placeholder()), native_float);
    let native_string = call("ir.StringImm", &[(&empty).into(), (&loc()).into()]);
    agree("StringImm", Ok(StringImm::placeholder()), native_string);
    let void = dtype::void();
    let native_dtype = call("ir.DataTypeImm", &[(&void).into(), (&loc()).into()]);
    agree("DataTypeImm", Ok(DataTypeImm::placeholder()), native_dtype);
    let none_type = NoneType::new();
    let native_none =
        call("ir.GenericConst", &[(&none_type).into(), (&none).into(), (&loc()).into()]);
    agree("GenericConst", Ok(GenericConst::placeholder()), native_none);

    let missing = MissingType::new();
    let native_var = call("ir.Var", &[(&empty).into(), (&missing).into(), (&loc()).into()]);
    agree_fresh("Var", Ok(Var::placeholder()), native_var);
    let native_global = call("ir.GlobalVar", &[(&empty).into(), (&loc()).into()]);
    agree_fresh("GlobalVar", Ok(GlobalVar::placeholder()), native_global.clone());
    let native_tuple = call("ir.Tuple", &[(&no_exprs).into(), (&loc()).into()]);
    agree("Tuple", Ok(Tuple::placeholder()), native_tuple);
    let one: Array<Expr> = [zero()].into_iter().collect();
    let pair = call("ir.Tuple", &[(&one).into(), (&loc()).into()]).unwrap();
    let index = 0i32;
    let native_item = call("ir.TupleGetItem", &[(&pair).into(), (&index).into(), (&loc()).into()]);
    agree("TupleGetItem", Ok(TupleGetItem::placeholder()), native_item);
    let (callee, no_types) = (native_global.unwrap(), Array::<Type>::default());
    let native_call = call(
        "ir.Call",
        &[
            (&none).into(),
            (&callee).into(),
            (&no_exprs).into(),
            (&none).into(),
            (&no_types).into(),
            (&loc()).into(),
        ],
    );
    agree_fresh("Call", Ok(Call::placeholder()), native_call);

    let tensor = Var::new("", TensorType::placeholder(), loc());
    let native_load =
        call("ir.TensorLoad", &[(&tensor).into(), (&no_exprs).into(), (&loc()).into()]);
    agree_fresh("TensorLoad", Ok(TensorLoad::placeholder()), native_load);
    let (region, region_type) = (Array::<tvm_tirx::ir::Range>::default(), TensorRegionType::new());
    let native_region = call(
        "ir.TensorRegion",
        &[(&tensor).into(), (&region).into(), (&region_type).into(), (&loc()).into()],
    );
    agree_fresh("TensorRegion", Ok(TensorRegion::placeholder()), native_region);

    let no_vars = Array::<Var>::default();
    let native_lambda =
        call("ir.LambdaExpr", &[(&no_vars).into(), (&zero()).into(), (&loc()).into()]);
    agree("LambdaExpr", Ok(LambdaExpr::placeholder()), native_lambda.clone());
    agree("StagingExpr", Ok(StagingExpr::placeholder()), native_lambda);
    let attrs = DictAttrs::new(Map::default());
    let native_function = call(
        "tirx.Function",
        &[(&no_vars).into(), (&none).into(), (&missing).into(), (&attrs).into(), (&loc()).into()],
    );
    agree("Function", Ok(tirx::Function::placeholder()), native_function.clone());
    agree("BaseFunc", Ok(BaseFunc::placeholder()), native_function);

    // C++ constructs no `OpaqueExpr` of its own: the placeholder is the node of a missing type.
    let opaque = OpaqueExpr::placeholder();
    assert_eq!(AnyView::from(&opaque).type_index(), OpaqueExprObj::type_index());
    assert!(opaque.ty.same_as(&missing));
}

#[test]
fn primitive_expressions_agree_with_cpp() {
    let _serial = serial();
    let (zero, fals) = (zero(), Expr::from(IntImm::bool(false)));
    let with_loc = |name: &str, args: &[&Expr]| {
        let loc = loc();
        let mut views: Vec<AnyView> = args.iter().map(|arg| AnyView::from(*arg)).collect();
        views.push(AnyView::from(&loc));
        global(name).call_packed(&views)
    };
    macro_rules! binary {
        ($operand:expr => $($node:ident),+) => {$(
            let native = with_loc(concat!("prim.", stringify!($node)), &[$operand, $operand]);
            agree(stringify!($node), Ok(prim::$node::placeholder()), native);
        )+};
    }
    binary!(&zero => Add, Sub, Mul, Div, Mod, FloorDiv, FloorMod, Min, Max);
    binary!(&zero => LShift, RShift, BitwiseAnd, BitwiseOr, BitwiseXor);
    binary!(&zero => EQ, NE, LT, LE, GT, GE);
    binary!(&fals => And, Or);
    agree("Not", Ok(prim::Not::placeholder()), with_loc("prim.Not", &[&fals]));
    agree("BitwiseNot", Ok(prim::BitwiseNot::placeholder()), with_loc("prim.BitwiseNot", &[&zero]));
    let select = with_loc("prim.Select", &[&fals, &zero, &zero]);
    agree("Select", Ok(prim::Select::placeholder()), select);
    let int32 = PrimType::int(32);
    let native_cast = global("prim.Cast").call_tuple((&int32, &zero, loc()));
    agree("Cast", Ok(prim::Cast::placeholder()), native_cast);
    let var = Expr::from(Var::new("", PrimType::int(32), loc()));
    agree_fresh("Let", Ok(prim::Let::placeholder()), with_loc("prim.Let", &[&var, &zero, &zero]));
    let (one, two) = (Expr::from(IntImm::int32(1)), Expr::from(IntImm::int32(2)));
    agree("Ramp", Ok(prim::Ramp::placeholder()), with_loc("prim.Ramp", &[&zero, &one, &two]));
    let broadcast = with_loc("prim.Broadcast", &[&zero, &two]);
    agree("Broadcast", Ok(prim::Broadcast::placeholder()), broadcast);
    let lanes: Array<Expr> = [zero.clone()].into_iter().collect();
    let native_shuffle = global("prim.Shuffle").call_tuple((&lanes, &lanes, loc()));
    agree("Shuffle", Ok(prim::Shuffle::placeholder()), native_shuffle);
}

/// An expression annotated with a function and a memory scope, which `__ffi_init__` does not
/// take: a type of a test that extends `ExprObj` and holds fields of the IR.
#[rs_class("rust_test.Annotated", final, placeholder)]
pub struct AnnotatedObj {
    base: ExprObj,
    pub value: Expr,
    pub sum: prim::Add,
    #[reflect(init_off)]
    pub func: BaseFunc,
    #[reflect(init_off)]
    pub scope: MemoryScope,
}
def_object_ref!(pub Annotated, Expr, AnnotatedObj);

/// A variable of a test that extends `VarObj`, without fields of its own.
#[rs_class("rust_test.PlainVar", final, placeholder)]
pub struct PlainVarObj {
    base: VarObj,
}
def_object_ref!(pub PlainVar, Var, PlainVarObj);

tvm_ffi::static_init_block! {
    register_type::<Annotated>()?;
    register_type::<PlainVar>()?;
}

#[test]
fn types_defined_in_rust_are_made_from_placeholders() {
    let _serial = serial();

    // The object a type extends holds the placeholders of its fields.
    let base = ExprObj::placeholder();
    assert!(base.loc.same_as(&loc()) && base.ty.as_node::<MissingTypeObj>().is_some());
    let var = PlainVar::from(PlainVarObj::placeholder());
    assert_eq!(var.name.as_str(), "");

    // `__ffi_init__` takes the fields of the type and the defaulted ones of `ExprObj`; the fields
    // it does not take keep their placeholders.
    let init = Function::from_type_attr(AnnotatedObj::type_index(), "__ffi_init__").unwrap();
    let kwargs = global("ffi.GetKwargsObject").call_tuple(()).unwrap();
    let sum = prim::Add::new(1, 2, loc()).unwrap();
    let (value, int32) = (Expr::from(IntImm::int32(3)), Type::from(PrimType::int(32)));
    let (ty_key, value_key, sum_key) =
        (FfiString::from("ty"), FfiString::from("value"), FfiString::from("sum"));
    let built = init
        .call_packed(&[
            AnyView::from(&kwargs),
            AnyView::from(&ty_key),
            AnyView::from(&int32),
            AnyView::from(&value_key),
            AnyView::from(&value),
            AnyView::from(&sum_key),
            AnyView::from(&sum),
        ])
        .unwrap();
    let annotated = Annotated::try_from(built).unwrap();
    assert!(annotated.ty.same_as(&int32) && annotated.loc.same_as(&loc()));
    assert!(annotated.value.same_as(&value) && annotated.sum.same_as(&sum));
    assert!(structural_equal(&annotated.func, &BaseFunc::placeholder()));
    assert!(annotated.scope.same_as(&MemoryScope::placeholder()));

    // A JSON graph reads it back through `__ffi_new__`, which makes it from placeholders.
    let json = global("ffi.ToJSONGraph").call_packed(&[AnyView::from(&annotated), AnyView::new()]);
    let read = global("ffi.FromJSONGraph").call_packed(&[AnyView::from(&json.unwrap())]);
    let read = Annotated::try_from(read.unwrap()).unwrap();
    assert!(!read.same_as(&annotated) && structural_equal(&read, &annotated));
}
