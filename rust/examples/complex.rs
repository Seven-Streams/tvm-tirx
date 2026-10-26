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

//! Complex numbers, in a library of their own: an IR type and operators defined in Rust, and a
//! pass, also in Rust, that lowers them to scalar arithmetic before code generation.
//!
//! `cargo build --example complex` builds this as `target/debug/examples/libcomplex.so`.
//! `docs/advanced/examples/complex.py` loads it, which registers the type `my_ext.Complex(elem)`
//! and the operators `my_ext.complex`, `my_ext.cmul`, `my_ext.real` and `my_ext.imag`, writes a
//! TVMScript function with them, and compiles it through a pipeline that runs `lower_complex()`
//! first. The `// begin` and `// end` comments mark what the documentation quotes.

use std::cell::RefCell;

use tvm_tirx::ir::op_attr::CALL_EFFECT_KIND;
use tvm_tirx::ir::prim::ex;
use tvm_tirx::ir::sig::{self, PrimExpr};
use tvm_tirx::ir::transform::{Pass, PassInfo};
use tvm_tirx::ir::{
    Bind, Call, CallEffectKind, CallObj, Expr, Location, OpDef, OpHandle, OpObj, PrimType, SeqStmt,
    Stmt, Type, TypeObj, Var,
};
use tvm_tirx::tirx::transform::FunctionPass;
use tvm_tirx::tirx::Function;
use tvm_tirx::tvm_ffi::{self, rs_class, Error, ObjectRefCast, ObjectRefCore, Result, WalkOrder};

// begin type
// The type: a complex number, printed as `my_ext.Complex(T.float32)`.
tvm_tirx::register_types! {
    /// A complex number whose real and imaginary parts are of the scalar type `elem`.
    #[rs_class("my_ext.ComplexType", final)]
    pub struct ComplexTypeObj {
        base: TypeObj,
        pub elem: PrimType,
    }
    def_object_ref!(pub ComplexType, Type, ComplexTypeObj);
}

impl ComplexType {
    pub fn new(elem: PrimType) -> Self {
        ComplexTypeObj::new(TypeObj::new(Location::unknown()), elem).into()
    }
}
// end type

// begin type inference
// The operators, whose result types are inferred here.

fn type_error(call: &CallObj, what: &str) -> Error {
    let op = call.op.as_node::<OpObj>().map_or("the operator", |op| op.name.as_str());
    Error::new(tvm_ffi::TYPE_ERROR, &format!("{op} expects {what}"), "")
}

/// The complex type of argument `index` of `call`.
fn complex_arg(call: &CallObj, index: usize) -> Result<ComplexType> {
    let ty = call.args.get(index)?.ty.clone();
    ty.try_cast().map_err(|_| type_error(call, "a complex value"))
}

/// `my_ext.complex(re, im)`: a complex number of the type of its parts.
fn infer_complex(call: &CallObj) -> Result<Type> {
    let re: PrimType = call.args.get(0)?.ty.clone().try_cast()?;
    let im: PrimType = call.args.get(1)?.ty.clone().try_cast()?;
    if re.dtype != im.dtype {
        return Err(type_error(call, "parts of the same type"));
    }
    Ok(ComplexType::new(re).into())
}

/// `my_ext.cmul(a, b)`: a complex number of the type of its operands.
fn infer_cmul(call: &CallObj) -> Result<Type> {
    let (a, b) = (complex_arg(call, 0)?, complex_arg(call, 1)?);
    if a.elem.dtype != b.elem.dtype {
        return Err(type_error(call, "complex values of the same type"));
    }
    Ok(a.into())
}

/// `my_ext.real(z)` and `my_ext.imag(z)`: a scalar of the type of the parts of `z`.
fn infer_part(call: &CallObj) -> Result<Type> {
    Ok(complex_arg(call, 0)?.elem.clone().into())
}
// end type inference

// begin operators
tvm_tirx::register_ops! {
    OpDef::new("my_ext.complex", "The complex number `re + im * i`.")
        .signature((
            sig::arg::<PrimExpr>("re", "The real part."),
            sig::arg::<PrimExpr>("im", "The imaginary part."),
        ))
        .set_infer_type(infer_complex)
        .set_attr(CALL_EFFECT_KIND, CallEffectKind::Pure);
    OpDef::new("my_ext.cmul", "The product of two complex numbers.")
        .signature((
            sig::arg::<Expr>("a", "A complex number."),
            sig::arg::<Expr>("b", "A complex number."),
        ))
        .set_infer_type(infer_cmul)
        .set_attr(CALL_EFFECT_KIND, CallEffectKind::Pure);
    OpDef::new("my_ext.real", "The real part of a complex number.")
        .signature(sig::arg::<Expr>("z", "A complex number."))
        .set_infer_type(infer_part)
        .set_attr(CALL_EFFECT_KIND, CallEffectKind::Pure);
    OpDef::new("my_ext.imag", "The imaginary part of a complex number.")
        .signature(sig::arg::<Expr>("z", "A complex number."))
        .set_infer_type(infer_part)
        .set_attr(CALL_EFFECT_KIND, CallEffectKind::Pure);
}

static COMPLEX: OpHandle = OpHandle::new("my_ext.complex");
static CMUL: OpHandle = OpHandle::new("my_ext.cmul");
static REAL: OpHandle = OpHandle::new("my_ext.real");
static IMAG: OpHandle = OpHandle::new("my_ext.imag");
// end operators

// The pass: `LowerComplex` rewrites a function bottom-up, so the operands of each node are lowered
// before the node itself. A complex value it can compute becomes `my_ext.complex(re, im)` of scalar
// expressions, a complex variable bound to one becomes two scalar variables, and the parts of
// either are read directly.

// begin lowering state
/// What the pass knows while it walks a function: the scalar variables that replace each complex
/// variable it has seen bound.
#[derive(Default)]
struct Lowering {
    split: Vec<(Var, Var, Var)>,
}

impl Lowering {
    /// The real and imaginary parts of `expr`, when it is `my_ext.complex(re, im)` or a complex
    /// variable already split.
    fn parts(&self, expr: &Expr) -> Option<(Expr, Expr)> {
        if let Some(call) = expr.as_node::<CallObj>() {
            if COMPLEX.is(&call.op) {
                return Some((call.args.get(0).ok()?, call.args.get(1).ok()?));
            }
        }
        let (_, re, im) = self.split.iter().find(|(var, _, _)| var.same_as(expr))?;
        Some((re.clone().into(), im.clone().into()))
    }
    // end lowering state

    // begin lowering calls
    /// Rewrite a call whose operands are already lowered.
    fn call(&self, call: Call) -> Result<Expr> {
        if CMUL.is(&call.op) {
            let (lhs, rhs) = (self.parts(&call.args.get(0)?), self.parts(&call.args.get(1)?));
            if let (Some((a, b)), Some((c, d))) = (lhs, rhs) {
                // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
                let re = ex(&a) * ex(&c) - ex(&b) * ex(&d);
                let im = ex(&a) * ex(&d) + ex(&b) * ex(&c);
                let args = [re, im];
                let call = Call::new(None, COMPLEX.get()?, args, None, [] as [Type; 0], &call.loc)?;
                return Ok(call.into());
            }
        } else if REAL.is(&call.op) || IMAG.is(&call.op) {
            if let Some((re, im)) = self.parts(&call.args.get(0)?) {
                return Ok(if REAL.is(&call.op) { re } else { im });
            }
        }
        Ok(call.into())
    }
    // end lowering calls

    // begin lowering binds
    /// Split `z = my_ext.complex(re, im)` into `z_re = re` and `z_im = im`.
    fn bind(&mut self, bind: Bind) -> Result<Stmt> {
        let (Some(ty), Some((re, im))) =
            (bind.var.ty.as_node::<ComplexTypeObj>(), self.parts(&bind.value))
        else {
            return Ok(bind.into());
        };
        let (name, loc) = (bind.var.name.as_str(), &bind.var.loc);
        let re_var = Var::new(&format!("{name}_re"), ty.elem.clone(), loc);
        let im_var = Var::new(&format!("{name}_im"), ty.elem.clone(), loc);
        self.split.push((bind.var.clone(), re_var.clone(), im_var.clone()));
        let binds = [Bind::new(re_var, re, &bind.loc)?, Bind::new(im_var, im, &bind.loc)?];
        Ok(SeqStmt::new(binds, &bind.loc).into())
    }
    // end lowering binds
}

// begin pass
/// Lower every complex value of `func` to scalar arithmetic.
fn lower(func: Function) -> Result<Function> {
    let state = RefCell::new(Lowering::default());
    let lowered = tvm_ffi::structural_map(
        func,
        (
            |bind: Bind| state.borrow_mut().bind(bind),
            |call: Call| state.borrow().call(call),
            // A split binding leaves a sequence inside a sequence: flatten it.
            |seq: SeqStmt| SeqStmt::new(seq.seq.iter(), &seq.loc),
        ),
        WalkOrder::PostOrder,
    )?;
    // Code generation knows no complex type, so none may remain: a complex parameter, say.
    let mut left = 0;
    tvm_ffi::structural_map(
        lowered.clone(),
        |expr: Expr| {
            left += usize::from(expr.ty.as_node::<ComplexTypeObj>().is_some());
            expr
        },
        WalkOrder::PostOrder,
    )?;
    if left > 0 {
        let message = format!("LowerComplex cannot lower {left} uses of complex values");
        return Err(Error::new(tvm_ffi::VALUE_ERROR, &message, ""));
    }
    lowered.try_into()
}
// end pass

// begin export
/// The pass, which Python gets from the library as `lower_complex()`.
fn lower_complex() -> Result<Pass> {
    let pass = FunctionPass::new(
        |func, _module, _ctx| lower(func).map(Some),
        PassInfo::new(0, "my_ext.LowerComplex"),
    );
    Ok(Pass::from(&pass))
}

tvm_ffi::tvm_ffi_dll_export_typed_func!(lower_complex, lower_complex);
// end export
