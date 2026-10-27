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

//! The registry of the [`register_ops!`](crate::register_ops) and
//! [`register_types!`](crate::register_types) blocks: gathered at link time, wherever in the
//! program or library they are, and run when it loads, as a C++ static init block does. Once
//! Python loads a library built on this crate, by any means, it sees its operators and types as
//! it sees those of a C++ library. A block that several libraries of a process have, as each
//! links its own copy of a crate they share, runs once.

use std::ffi::c_void;
use std::panic::catch_unwind;
use std::sync::{Once, OnceLock};

use tvm_ffi::{Any, Error, Function, Result, INTERNAL_ERROR};

/// A [`register_ops!`](crate::register_ops) or [`register_types!`](crate::register_types) block,
/// gathered at link time.
#[doc(hidden)]
pub struct Registration {
    /// The macro of the block, and its module and line.
    kind: &'static str,
    module: &'static str,
    line: u32,
    /// The version of the block's crate and a hash of the block's text, which with the module and
    /// line identify the block in every library that links the crate.
    version: &'static str,
    text_hash: u64,
    register: fn() -> Result<()>,
    /// The outcome of the block, which runs once: the error's text if it failed.
    outcome: OnceLock<std::result::Result<(), std::string::String>>,
}

/// The blocks of [`register_ops!`](crate::register_ops) and
/// [`register_types!`](crate::register_types), gathered at link time from the whole program or
/// library.
#[doc(hidden)]
#[linkme::distributed_slice]
pub static REGISTRATIONS: [Registration];

impl Registration {
    #[doc(hidden)]
    pub const fn __new(
        kind: &'static str,
        module: &'static str,
        line: u32,
        version: &'static str,
        text: &str,
        register: fn() -> Result<()>,
    ) -> Self {
        let text_hash = text_hash(text);
        Self { kind, module, line, version, text_hash, register, outcome: OnceLock::new() }
    }

    /// The identity of the block in the process, `module:line@version#hash`: the same in every
    /// library that links the block's crate, and another one if the crate version or the block's
    /// text differs.
    #[doc(hidden)]
    pub fn identity(&self) -> std::string::String {
        format!("{}:{}@{}#{:016x}", self.module, self.line, self.version, self.text_hash)
    }

    /// The name of the global function that marks the block registered in the process,
    /// `ir.registered_block.<identity>`. The libraries of a process share the global function
    /// table of TVM FFI, and their statics are their own.
    #[doc(hidden)]
    pub fn marker(&self) -> std::string::String {
        format!("ir.registered_block.{}", self.identity())
    }

    /// Run the block if it has not run. The caller that runs a failing block gets its error;
    /// later ones get an error that names the block.
    #[doc(hidden)]
    pub fn register(&self) -> Result<()> {
        let mut failure = None;
        let outcome = self.outcome.get_or_init(|| {
            self.run().map_err(|error| {
                let text = format!("{}: {}", error.kind(), error.message());
                failure = Some(error);
                text
            })
        });
        match (failure, outcome) {
            (Some(error), _) => Err(error),
            (None, Ok(())) => Ok(()),
            (None, Err(text)) => {
                let message = format!("The {} block at {} failed: {text}", self.kind, self.place());
                Err(Error::new(INTERNAL_ERROR, &message, ""))
            }
        }
    }

    /// Run the block unless another library of the process has run it. Every library links its
    /// own copy of the crates it is built from, so the same block can be in several, and what it
    /// defines is registered once, with the hooks of the first library that runs it. The block
    /// is marked once it succeeds, so that one that fails fails in every library.
    fn run(&self) -> Result<()> {
        let marker = self.marker();
        if Function::get_global(&marker).is_err() {
            (self.register)()?;
            Function::register_global(&marker, Function::from_packed(|_| Ok(Any::default())))?;
        }
        Ok(())
    }

    /// Where the block is, for an error.
    fn place(&self) -> std::string::String {
        format!("{}:{}", self.module, self.line)
    }
}

/// The FNV-1a hash of `text` without its whitespace, which does not depend on how the compiler
/// spaces the tokens of a block it stringifies.
const fn text_hash(text: &str) -> u64 {
    let bytes = text.as_bytes();
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut i = 0;
    while i < bytes.len() {
        if !bytes[i].is_ascii_whitespace() {
            hash = (hash ^ bytes[i] as u64).wrapping_mul(0x0100_0000_01b3);
        }
        i += 1;
    }
    hash
}

/// Run every [`register_ops!`](crate::register_ops) and [`register_types!`](crate::register_types)
/// block of this program or library that has not run, in the order of their modules and lines;
/// running again does nothing. Every block is tried, and the error lists those that failed. A
/// block that another library of the process has registered, from the same version of its crate
/// and the same text, is skipped: libraries that link the same crate register its operators and
/// types once.
///
/// The blocks run when their program or library loads, so this reports the errors of those that
/// failed. Every library built on this crate exports it as the TVM FFI function
/// `tirx_register_ir`, which `import tvm_tirx` calls on `libtvm_tirx_rust` to raise them. Where
/// the loader does not run the blocks, the first lookup through an [`OpHandle`](crate::OpHandle) of an operator
/// that is not registered calls it.
pub fn register_defined_ir() -> Result<()> {
    let mut blocks: Vec<&Registration> = REGISTRATIONS.iter().collect();
    blocks.sort_by_key(|block| (block.module, block.line));
    let mut failures: Vec<(&Registration, Error)> =
        blocks.into_iter().filter_map(|block| block.register().err().map(|e| (block, e))).collect();
    if failures.len() <= 1 {
        return failures.pop().map_or(Ok(()), |(_, error)| Err(error));
    }
    let mut kinds: Vec<&str> = failures.iter().map(|(block, _)| block.kind).collect();
    kinds.sort();
    kinds.dedup();
    let mut message = format!("{} {} blocks failed:", failures.len(), kinds.join(" and "));
    for (block, error) in &failures {
        message.push_str(&format!("\n  {}: {}: {}", block.place(), error.kind(), error.message()));
    }
    Err(Error::new(failures[0].1.kind(), &message, ""))
}

/// The constructor of every [`register_ops!`](crate::register_ops) and
/// [`register_types!`](crate::register_types) block, which the loader runs when the program or
/// library loads: the first one keeps the library loaded and runs every block of it. An error
/// cannot reach a caller here, so it is printed; [`register_defined_ir`] reports it again.
#[doc(hidden)]
pub extern "C" fn __register_at_load() {
    static LOADED: Once = Once::new();
    LOADED.call_once(|| {
        let library = keep_loaded().unwrap_or_else(|| "register_ops!".to_string());
        match catch_unwind(register_defined_ir) {
            Ok(Ok(())) => {}
            Ok(Err(error)) => eprintln!(
                "{library}: registering operators and types failed: {}: {}",
                error.kind(),
                error.message()
            ),
            Err(_) => eprintln!("{library}: registering operators and types panicked"),
        }
    });
}

/// Keep the library that holds this code loaded for the rest of the process, as the operators and
/// types it registers keep its hooks: unloading it, as `tvm_ffi.load_module` does when Python drops the
/// module, would leave them dangling. Returns the path of the library.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn keep_loaded() -> Option<std::string::String> {
    use std::ffi::{c_char, c_int, CStr};

    #[repr(C)]
    struct DlInfo {
        fname: *const c_char,
        fbase: *mut c_void,
        sname: *const c_char,
        saddr: *mut c_void,
    }
    extern "C" {
        fn dladdr(addr: *const c_void, info: *mut DlInfo) -> c_int;
        fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    }
    // RTLD_LAZY | RTLD_NOLOAD | RTLD_NODELETE.
    #[cfg(target_os = "linux")]
    const FLAGS: c_int = 0x1 | 0x4 | 0x1000;
    #[cfg(target_os = "macos")]
    const FLAGS: c_int = 0x1 | 0x10 | 0x80;

    let null = std::ptr::null_mut();
    let mut info = DlInfo { fname: null as _, fbase: null, sname: null as _, saddr: null };
    // SAFETY: `dladdr` fills `info` for an address of this library, whose path stays valid while
    // the library is loaded; `dlopen` of a loaded library only marks it, and its handle is kept.
    unsafe {
        if dladdr(keep_loaded as *const c_void, &mut info) == 0 || info.fname.is_null() {
            return None;
        }
        dlopen(info.fname, FLAGS);
        Some(CStr::from_ptr(info.fname).to_string_lossy().into_owned())
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn keep_loaded() -> Option<std::string::String> {
    None
}

/// The registration of a [`register_ops!`](crate::register_ops) or
/// [`register_types!`](crate::register_types) block: gathered at link time, and run by a
/// constructor when its program or library loads.
#[doc(hidden)]
#[macro_export]
macro_rules! __register_block {
    ($kind:literal, $text:expr, $register:expr) => {
        const _: () = {
            #[$crate::__private::linkme::distributed_slice($crate::REGISTRATIONS)]
            #[linkme(crate = $crate::__private::linkme)]
            static REGISTRATION: $crate::Registration = $crate::Registration::__new(
                $kind,
                module_path!(),
                line!(),
                match option_env!("CARGO_PKG_VERSION") {
                    Some(version) => version,
                    None => "",
                },
                $text,
                $register,
            );
            #[used]
            #[cfg_attr(target_os = "linux", link_section = ".init_array")]
            #[cfg_attr(target_os = "macos", link_section = "__DATA,__mod_init_func")]
            #[cfg_attr(windows, link_section = ".CRT$XCU")]
            static REGISTER_AT_LOAD: extern "C" fn() = $crate::__register_at_load;
        };
    };
}
