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

//! Tests of `tvm_tirx::ty`, which is the single file `src/ty.rs`, and of `#[tirx_type]`: an
//! `ir.Type` defined in Rust, seen through the FFI type table, the C++ engines (structural
//! equality and hashing, deep copy, `repr`, TVMScript), the Rust engines and the IR that uses it.

#[path = "../common/mod.rs"]
mod common;

use std::mem::offset_of;

use tvm_tirx::ir::{Evaluate, Location, PrimType, Type, TypeObj, Var, VarObj, UNKNOWN_LOC};
use tvm_tirx::tirx_type;
use tvm_tirx::tvm_ffi::reflection::ShallowCopy;
use tvm_tirx::tvm_ffi::{
    structural_map, Any, AnyView, DefRegionKind, FieldGetter, MapDispatch, ObjectCore,
    ObjectRefCore, Result, String as FfiString, StructuralView, WalkOrder,
};

use common::{assert_type_contract, global, runtime_type_info, structural_equal};

/// How C++ prints a node (`ffi.ReprPrint`).
fn repr<T>(value: &T) -> String
where
    for<'a> AnyView<'a>: From<&'a T>,
{
    let text = global("ffi.ReprPrint").call_packed(&[AnyView::from(value)]).unwrap();
    FfiString::try_from(text).unwrap().as_str().to_string()
}

/// C++ `ffi::StructuralHash()(value)`.
fn structural_hash<T>(value: &T) -> Result<i64>
where
    for<'a> AnyView<'a>: From<&'a T>,
{
    let flag = <AnyView as From<&bool>>::from;
    global("ffi.StructuralHash")
        .call_packed(&[AnyView::from(value), flag(&false), flag(&false)])?
        .try_into()
}

/// A pair of types with a tag, for the tests.
#[tirx_type("testing.rust.PairType")]
pub struct PairType {
    /// The first element.
    pub lhs: Type,
    /// The second element.
    pub rhs: Type,
    pub tag: i64,
    /// Not part of structural equality or hashing.
    #[seq_hash_ignore]
    pub note: FfiString,
}

impl PairType {
    /// A copy with the fields `edit` assigns replaced (C++ `CopyOnWrite()`).
    fn with(&self, edit: impl FnOnce(&mut PairTypeObj)) -> Self {
        let mut node = self.shallow_copy();
        edit(&mut node);
        node.into()
    }
}

fn setup() {
    // The first use registers the type and its reflection.
    PairTypeObj::type_index();
}

fn pair(lhs: Type, rhs: Type, tag: i64, note: &str) -> PairType {
    PairType::from_complete_fields(UNKNOWN_LOC, lhs, rhs, tag, note)
}

fn int_float_pair(tag: i64, note: &str) -> PairType {
    pair(PrimType::int(32).into(), PrimType::float(32).into(), tag, note)
}

#[test]
fn registers_under_ir_type_with_its_fields_reflected() {
    setup();
    assert_type_contract::<PairTypeObj, TypeObj>(true);
    assert_eq!(PairTypeObj::TYPE_DEPTH, TypeObj::TYPE_DEPTH + 1);

    let info = runtime_type_info::<PairTypeObj>();
    let metadata = unsafe { &*info.metadata };
    assert_eq!(metadata.structural_eq_hash_kind, 1, "TreeNode, as every ir.Type");
    let fields: Vec<(String, i64, i64)> = (0..info.num_fields as usize)
        .map(|index| {
            let field = unsafe { &*info.fields.add(index) };
            (field.name.as_str().to_string(), field.offset, field.flags)
        })
        .collect();
    assert_eq!(
        fields,
        vec![
            ("lhs".to_string(), offset_of!(PairTypeObj, lhs) as i64, 0),
            ("rhs".to_string(), offset_of!(PairTypeObj, rhs) as i64, 0),
            ("tag".to_string(), offset_of!(PairTypeObj, tag) as i64, 0),
            ("note".to_string(), offset_of!(PairTypeObj, note) as i64, 1 << 3),
        ]
    );
}

#[test]
fn reflection_getters_read_own_and_inherited_fields() {
    setup();
    let value = int_float_pair(7, "seven");
    let get = |field: &str| -> Any {
        FieldGetter::new(PairTypeObj::type_index(), field).unwrap().get_any(&*value).unwrap()
    };
    assert!(Type::try_from(get("lhs")).unwrap().same_as(&value.lhs));
    assert!(Type::try_from(get("rhs")).unwrap().same_as(&value.rhs));
    assert_eq!(i64::try_from(get("tag")).unwrap(), 7);
    assert_eq!(FfiString::try_from(get("note")).unwrap().as_str(), "seven");
    // `loc` belongs to the parent `ir.Type`.
    assert!(Location::try_from(get("loc")).unwrap().same_as(&value.loc));
}

#[test]
fn cpp_structural_equal_and_hash_walk_the_reflected_fields() {
    setup();
    let a = int_float_pair(7, "a");
    let b = int_float_pair(7, "b");
    let c = int_float_pair(8, "a");
    assert!(structural_equal(&a, &b), "the note is ignored");
    assert!(!structural_equal(&a, &c));
    assert_eq!(structural_hash(&a).unwrap(), structural_hash(&b).unwrap());
    assert_ne!(structural_hash(&a).unwrap(), structural_hash(&c).unwrap());

    // Nested in itself, as any type.
    let int32: Type = PrimType::int(32).into();
    let outer_a = pair(a.clone().into(), int32.clone(), 0, "");
    let outer_b = pair(b.into(), int32.clone(), 0, "");
    let outer_c = pair(c.into(), int32, 0, "");
    assert!(structural_equal(&outer_a, &outer_b));
    assert!(!structural_equal(&outer_a, &outer_c));
    assert_eq!(structural_hash(&outer_a).unwrap(), structural_hash(&outer_b).unwrap());
}

#[test]
fn cpp_repr_and_tvmscript_print_the_node() {
    setup();
    let value = int_float_pair(7, "seven");
    let text = repr(&value);
    assert!(text.contains("testing.rust.PairType"), "{text}");
    assert!(text.contains("tag=7"), "{text}");
    assert!(text.contains("seven"), "{text}");

    let script = global("script.printer.Script")
        .call_packed(&[AnyView::from(&value), AnyView::new()])
        .unwrap();
    let script = FfiString::try_from(script).unwrap();
    assert!(script.as_str().contains("testing.rust.Pair("), "{}", script.as_str());
    assert!(script.as_str().contains("\"seven\""), "{}", script.as_str());
}

#[test]
fn cpp_deep_copy_goes_through_the_shallow_copy_and_the_setters() {
    setup();
    let value = int_float_pair(7, "seven");
    let copy = global("ffi.DeepCopy").call_packed(&[AnyView::from(&value)]).unwrap();
    let copy = PairType::try_from(copy).unwrap();
    assert!(!copy.same_as(&value));
    assert!(structural_equal(&copy, &value));
    assert_eq!(copy.tag, 7);
    assert_eq!(copy.note.as_str(), "seven");
}

/// Adds one to every integer leaf.
struct IncrementIntegers;

impl MapDispatch for IncrementIntegers {
    fn dispatch_map(&mut self, value: &StructuralView, _: DefRegionKind) -> Option<Result<Any>> {
        value.cast::<i64>().map(|integer| Ok(Any::from(integer + 1)))
    }
}

#[test]
fn rust_engines_descend_through_the_reflected_fields() {
    setup();
    let value = int_float_pair(7, "seven");
    // Shared with `value`, so the engine copies the node through `__ffi_shallow_copy__`.
    let mapped =
        structural_map(value.clone(), &mut IncrementIntegers, WalkOrder::PostOrder).unwrap();
    let mapped = PairType::try_from(mapped).unwrap();
    assert!(!mapped.same_as(&value));
    assert_eq!(mapped.tag, 8);
    assert_eq!(mapped.note.as_str(), "seven");
    assert!(mapped.lhs.same_as(&value.lhs));
    assert_eq!(value.tag, 7);
}

#[test]
fn with_is_copy_on_write() {
    setup();
    let value = int_float_pair(7, "seven");
    let edited = value.with(|node| node.tag = 9);
    assert!(!edited.same_as(&value));
    assert_eq!(edited.tag, 9);
    assert_eq!(value.tag, 7);
    assert!(edited.lhs.same_as(&value.lhs));
    assert!(edited.rhs.same_as(&value.rhs));
    assert!(edited.loc.same_as(&value.loc));
    assert_eq!(edited.note.as_str(), "seven");
}

#[test]
fn converts_through_any_and_the_type_hierarchy() {
    setup();
    let value = int_float_pair(1, "");
    let as_type: Type = value.clone().into();
    assert!(as_type.as_node::<PairTypeObj>().is_some());
    let any = Any::from(as_type);
    assert_eq!(any.type_index(), PairTypeObj::type_index());
    let back = PairType::try_from(any).unwrap();
    assert!(back.same_as(&value));
}

/// A type with a private field, which the module that defines it reads.
#[tirx_type("testing.rust.TagType")]
struct TagType {
    tag: i64,
}

#[test]
fn a_variable_of_the_type_takes_part_in_ir() {
    setup();
    // The type of a variable, as `PrimType` or `TupleType` is one.
    let x = Var::new("x", int_float_pair(7, "seven"), UNKNOWN_LOC);
    assert_eq!(x.ty.as_node::<PairTypeObj>().unwrap().tag, 7);
    assert!(structural_equal(&x.ty, &int_float_pair(7, "other")));
    assert!(!structural_equal(&x.ty, &int_float_pair(8, "seven")));

    // Inside IR, which C++ deep-copies through the shallow copy and the setters.
    let stmt = Evaluate::new(x.clone(), UNKNOWN_LOC);
    let copy = global("ffi.DeepCopy").call_packed(&[AnyView::from(&stmt)]).unwrap();
    let copy = Evaluate::try_from(copy).unwrap();
    let copied = copy.value.as_node::<VarObj>().unwrap();
    let ty = copied.ty.as_node::<PairTypeObj>().unwrap();
    assert_eq!((ty.tag, ty.note.as_str()), (7, "seven"));
    assert!(structural_equal(&copied.ty, &x.ty));

    // A private field reads in the module that defines the type.
    let tag = TagType::from_complete_fields(UNKNOWN_LOC, 3);
    assert_eq!(tag.tag, 3);
    let tag: Type = tag.into();
    assert!(repr(&tag).contains("testing.rust.TagType"), "{}", repr(&tag));
}
