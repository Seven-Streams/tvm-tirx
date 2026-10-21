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

/*!
 * \file src/ir/link_anchor.cc
 * \brief The link anchor of libtvm_tirx_compiler.
 *
 * Each TIRx shared library exports a `<library>_link_anchor` function that does nothing, defined
 * in the `link_anchor.cc` of its sources. A language binding that uses the library only through
 * the TVM FFI registry references no other symbol of it, so a linker that drops unreferenced
 * libraries (`--as-needed`) would drop the dependency and the library's registrations would
 * never run; the binding references the anchor (its address is enough) to keep the library
 * linked. There is no header: the binding declares the function itself.
 */
#include <tvm/runtime/base.h>

extern "C" TVM_TIRX_DLL int tvm_tirx_compiler_link_anchor() { return 0; }
