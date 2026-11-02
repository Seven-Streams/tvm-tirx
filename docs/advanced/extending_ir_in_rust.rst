..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

Getting Started with Extensions
===============================

.. note::

   All the code in this tutorial is in the repository, in
   `examples/extension <https://github.com/apache/tvm-tirx/tree/main/examples/extension>`_:
   a Python prototype of the extension and the Rust package it becomes.

This tutorial shows how to extend TIRx: define new IR types and operators,
write a pass that lowers them, and use them from Python like built-in features.
Each step is shown in Rust and in Python.

In this chapter, we will first look at the two ways to write an extension, then
get ready for the rest of the tutorial: the extension in each language and the
example program it will compile.

From a Python Prototype to Rust
-------------------------------

TIRx is built on a stable C++ library for its IR, analysis, transformations,
and code generation. In most cases, you work with it from Python: tensor
programs are written as Python functions in TVMScript, then compiled and run
directly.

With agents, compiler development iterates quickly. Along the way, a project may
need new types and operators to describe its own concepts, such as a new number
format or a hardware feature, and passes that lower them to what the backend
supports. An extension defines them on the same IR objects as C++, outside the
C++ library, so the C++ core stays stable while the extension iterates quickly.
Once loaded, its types and operators can be written in TVMScript and its passes
can run in Python compilation pipelines, just like built-in ones.

An extension can be written in Rust or in Python, with the same design in both:

- **Rust**, through the ``tvm-tirx`` crate, builds the extension into a native
  library that builds and loads on its own. Its hooks, such as type inference,
  are native functions that C++ calls directly, and its operators can have
  validators that check the types of their arguments.
- **Python** defines the extension in a module when it is imported, with no
  build step, which makes it the quickest way to try out a design.

A typical extension starts as a Python prototype and moves to Rust once its
design settles. The tabs in this tutorial show both versions; selecting a
language in one tab selects it in the others. :doc:`using_ir_extensions` lists
how each part carries over.

Setting Up the Project
----------------------

First build and install TIRx from this checkout. The Rust version is a Python
package that contains its Rust library, so it is installed and imported like
any other package; building it needs Rust 1.84 or newer. The Python version is
one module that needs no build: with its directory on the module path,
``import my_ext`` defines the extension. From the TIRx repository root, in the
same Python environment:

.. tabs::

   .. group-tab:: Rust

      .. code-block:: bash

         uv pip install scikit-build-core
         uv pip install --no-build-isolation --no-deps ./examples/extension

   .. group-tab:: Python

      .. code-block:: bash

         export PYTHONPATH=$PWD/examples/extension/my_ext_py

The Rust package depends on ``apache-tvm-tirx`` at run time. Disabling build
isolation builds it against this checkout's Rust crate, pinned TVM FFI sources,
and installed TIRx, instead of a released package installed in a temporary
build environment. The two versions are laid out as follows:

.. tabs::

   .. group-tab:: Rust

      .. code-block:: text

         examples/extension/
         ├── Cargo.toml        # the Rust library, which depends on tvm-tirx
         ├── src/lib.rs        # the type, operators, and pass
         ├── CMakeLists.txt    # builds the library with cargo and installs it into my_ext
         ├── pyproject.toml    # the Python package, built with scikit-build-core
         ├── python/my_ext/    # loads the library and registers a compilation pipeline
         └── run_example.py    # compiles and runs a function that uses the extension

   .. group-tab:: Python

      .. code-block:: text

         examples/extension/
         ├── my_ext_py/my_ext.py  # the type, operators, and pass
         └── run_example.py       # compiles and runs a function that uses the extension

The Rust library is an ordinary crate that builds a shared library. Its only
dependency is the `tvm-tirx <https://crates.io/crates/tvm-tirx>`_ crate, which
re-exports TVM FFI as ``tvm_tirx::tvm_ffi``:

.. literalinclude:: ../../examples/extension/Cargo.toml
   :language: toml
   :start-at: [lib]

Case Study: Complex Numbers
---------------------------

To show how to define new types, operators, and passes, we will use complex
numbers as a running example throughout this tutorial: a complex type, operators
that build and multiply complex values, and a pass that expands complex
multiplication into real arithmetic:

.. math::

   (a + bi)(c + di) = (ac - bd) + (ad + bc)i.

The following function multiplies two arrays of complex numbers element by
element. ``A`` and ``B`` hold the real and imaginary parts of the first operand,
``C`` and ``D`` those of the second:

.. literalinclude:: ../../examples/extension/run_example.py
   :language: python
   :start-after: # begin kernel
   :end-before: # end kernel

``my_ext`` is the extension's Python module, the Rust package or the prototype,
which holds the builders of its type and operators; :doc:`using_ir_extensions`
shows how it provides them. ``z``, ``w``, and ``product`` are complex values
that exist only in the IR. The function's inputs and outputs are ordinary
``float32`` arrays, so no runtime storage format for complex numbers is needed.

Next, :doc:`defining_ir_extensions` defines the type, operators, and pass this
function uses, and :doc:`using_ir_extensions` compiles and runs it.
