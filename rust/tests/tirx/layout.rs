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

//! Tests of `tirx::layout`: layout axes, tile layouts and swizzled layouts.

use tvm_tirx::ir::Expr;
use tvm_tirx::tirx::{self, Axis, ComposeLayout, Iter, Layout, TileLayout};
use tvm_tirx::tvm_ffi::{Array, Map, Object, ObjectCore, ObjectRefCast, String as FfiString};

use super::{as_i64, int32, int_imm, ir, s, tile_8x8};
use crate::common::{
    assert_fields, assert_type_contracts, global, runtime_type_info, serial, structural_equal,
};

#[test]
fn generated_types_match_the_registry() {
    let _serial = serial();
    assert_type_contracts!(true; tirx::IterObj: Object);
    // Opaque bindings embed only their parent; the native object is larger.
    assert_type_contracts!(false;
        tirx::LayoutObj: Object, tirx::ComposeLayoutObj: tirx::LayoutObj,
        tirx::TileLayoutObj: tirx::LayoutObj,
    );
    // `tirx.Axis` derives from the builtin `ffi.Enum`, whose stand-in is private to the module.
    assert_eq!(tirx::AxisObj::TYPE_KEY, "tirx.Axis");
    assert_eq!(runtime_type_info::<tirx::AxisObj>().type_depth, tirx::AxisObj::TYPE_DEPTH);
}

#[test]
fn memory_axes() {
    let _serial = serial();
    let m = Axis::get("m").unwrap();
    assert!(m.is_memory().unwrap() && !m.is_thread().unwrap());
    assert!(m.backend().unwrap().is_none());
    assert!(m.get_scope().unwrap().is_none());
}

#[test]
fn layouts() {
    let _serial = serial();
    let tile = tile_8x8();
    let layout = Layout::from(&tile);
    layout.check_well_formed().unwrap();
    assert_eq!(as_i64(&layout.size(None).unwrap()), 64);
    assert_eq!(as_i64(&layout.span(Some("m")).unwrap()), 64);
    let key = FfiString::from("m");
    let at = |coord: Map<FfiString, Expr>| as_i64(&coord.get(&key).unwrap().unwrap());
    assert_eq!(at(layout.apply([int32(1), int32(2)]).unwrap()), 10);
    assert_eq!(at(layout.apply_linear(int32(10)).unwrap()), 10);
    assert_eq!(at(layout.apply_with_shape([int32(10)], [int32(64)]).unwrap()), 10);
    let canonical = layout.canonicalize().unwrap();
    assert_eq!(as_i64(&canonical.size(None).unwrap()), 64);

    assert!(tile.is_trivial().unwrap());
    assert!(tile.is_contiguous(None).unwrap());
    let (grouped, separators) = tile.group([int32(8), int32(8)]).unwrap();
    assert_eq!(separators.len(), 3);
    assert!(grouped.is_trivial().unwrap());
    assert!(tile.get_scope().unwrap().is_none());

    let tiled = layout.tile(&tile, [int32(8), int32(8)], [int32(8), int32(8)]).unwrap();
    assert_eq!(as_i64(&tiled.size(None).unwrap()), 64 * 64);

    // Memory offset 10 becomes 10 ^ ((10 >> 3) & 3) = 11.
    let swizzled = ComposeLayout::new(0, 2, 3, &tile, true).unwrap();
    assert_eq!(swizzled.atom_len().unwrap(), 3);
    assert_eq!(at(Layout::from(&swizzled).apply([int32(1), int32(2)]).unwrap()), 11);
}

#[test]
fn thread_layout_scope() {
    let _serial = serial();
    global("tirx.cuda.RegisterLayoutAxes").call_tuple(()).unwrap();
    let laneid = Axis::get("laneid").unwrap();
    let lanes =
        TileLayout::new([Iter::new(int32(32), int32(1), laneid)], Vec::<Iter>::new(), &Map::new());
    let (inner, outer) = lanes.get_scope().unwrap().unwrap();
    assert_eq!(inner.local_name().unwrap().as_str(), "thread");
    assert_eq!(outer.local_name().unwrap().as_str(), "warp");
}

#[test]
fn native_ir_objects_agree_with_getters_and_rust_allocation() {
    let _serial = serial();
    let ir = ir();

    // tirx.Axis is a registry enum: one instance per name.
    let axis = |name: &str| -> tirx::Axis {
        global("tirx.AxisGet").call_tuple((s(name),)).unwrap().try_into().unwrap()
    };
    let memory = axis("m");
    assert!(std::ptr::eq::<tirx::AxisObj>(&*memory, &*axis("m")));

    // tirx.Iter
    let (extent, stride): (Expr, Expr) = (int_imm(&ir, 4).into(), int_imm(&ir, 1).into());
    let iter: tirx::Iter = global("tirx.Iter")
        .call_tuple((extent.clone(), stride.clone(), memory.clone()))
        .unwrap()
        .try_into()
        .unwrap();
    assert_fields!(iter; extent = extent, stride = stride, axis = memory);
    let rust = tirx::Iter::new(extent.clone(), stride.clone(), memory.clone());
    assert!(structural_equal(&rust, &iter));

    // tirx.TileLayout is opaque: its fields come through the getters.
    let layout: tirx::TileLayout = global("tirx.TileLayout")
        .call_tuple((
            Array::new(vec![iter.clone()]),
            Array::<tirx::Iter>::new(vec![]),
            Map::<tirx::Axis, Expr>::new(),
        ))
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(layout.shard().unwrap().len(), 1);
    assert!(structural_equal(&layout.shard().unwrap().get(0).unwrap(), &iter));
    assert_eq!(layout.replica().unwrap().len(), 0);
    assert_eq!(layout.offset().unwrap().len(), 0);
    let as_layout: tirx::Layout = layout.clone().into();
    assert!(as_layout.clone().try_cast::<tirx::TileLayout>().is_ok());
    assert!(as_layout.try_cast::<tirx::ComposeLayout>().is_err());
}
