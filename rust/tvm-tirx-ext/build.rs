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

//! The loader path of the tests. `tvm-tirx` links `libtvm_tirx_compiler` and passes its search
//! path on to the crates that link it, but the loader-path environment its build script emits only
//! applies to its own package: this script emits it again, from what `tvm-tirx` hands its
//! dependents (`links = "tvm_tirx_compiler"`), so that a plain `cargo test` finds the libraries at
//! startup.

use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // The values below come from `tvm-tirx`'s build script, which runs again when these change.
    println!("cargo:rerun-if-env-changed=PATH");
    println!("cargo:rerun-if-env-changed=TVM_TIRX_LIBRARY_PATH");
    let var =
        env::var("DEP_TVM_TIRX_COMPILER_LOADER_VAR").expect("tvm-tirx's build script sets it");
    let path =
        env::var("DEP_TVM_TIRX_COMPILER_LOADER_PATH").expect("tvm-tirx's build script sets it");
    if var != "PATH" {
        println!("cargo:rerun-if-env-changed={var}");
    }
    println!("cargo:rustc-env={var}={path}");
}
