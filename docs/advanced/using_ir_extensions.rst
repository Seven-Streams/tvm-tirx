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

Using and Compiling the Extension
=================================

In this chapter, we look at how the Python module ``my_ext`` provides the
extension defined in :doc:`defining_ir_extensions`, then use it to compile the
complex multiplication function and check the result against NumPy. Last, we
compare the two versions, to move a prototype to Rust.

The Python Module
-----------------

To write the type and operators in TVMScript, Python needs a module that holds
their builders. Definitions under ``tirx.`` go to ``T``, but an extension
namespace such as ``my_ext`` is published only into a module that the
extension declares. ``my_ext`` therefore declares itself as that module first:

.. tabs::

   .. group-tab:: Rust

      .. literalinclude:: ../../examples/extension/python/my_ext/__init__.py
         :language: python
         :start-after: # begin load
         :end-before: # end load

   .. group-tab:: Python

      .. literalinclude:: ../../examples/extension/my_ext_py/my_ext.py
         :language: python
         :start-after: # begin load
         :end-before: # end load

The Rust package's
:download:`__init__.py <../../examples/extension/python/my_ext/__init__.py>`
declares the module before it loads the library, whose static init blocks
register the type and operators; ``load_lib_module`` finds the library that the
package installs next to ``__init__.py``. The Python version then defines the
type and operators, as in :doc:`defining_ir_extensions`, and publishes each into
the module as it completes.

The two ``_init_exposed_*_api`` calls, internal to ``tvm_tirx.ir`` for now,
publish the operators and the type into ``my_ext``; ``auto_emit_void_call``
gives its builders the TVMScript behavior of the ``T`` builders. To parse
printed IR that uses the extension, pass the module as well, as in
``extra_vars={"T": T, "my_ext": my_ext}``.

The registered type and operators let the IR express complex values;
``LowerComplex`` turns them into a form the backend can compile. The default
pipeline does not know about the pass, so ``my_ext`` registers a pipeline, also
named ``my_ext``, that runs it first. Both versions register it with the same
code:

.. literalinclude:: ../../examples/extension/python/my_ext/__init__.py
   :language: python
   :start-after: # begin pipeline
   :end-before: # end pipeline

``lower_complex`` returns the pass: the Rust package returns the one its library
exports, and the prototype builds it with ``function_pass``.

Compile the Function
--------------------

:download:`run_example.py <../../examples/extension/run_example.py>` imports
``my_ext``, which provides the extension and registers the pipeline, and
compiles the function from :doc:`extending_ir_in_rust` with it:

.. literalinclude:: ../../examples/extension/run_example.py
   :language: python
   :start-after: # begin compile
   :end-before: # end compile

Compiling with the default pipeline instead fails in LLVM code generation,
which has no complex type.

Check the Result
----------------

A successful compilation shows that ``LowerComplex`` left no complex values for
the backend. To check that its rewrite also computes the right numbers, the
script runs the compiled function on random ``float32`` inputs and compares
both outputs with NumPy's complex multiplication:

.. literalinclude:: ../../examples/extension/run_example.py
   :language: python
   :start-after: # begin comparison
   :end-before: # end comparison

The script is the same for both versions. With the module path set, or the
package installed, as in :doc:`extending_ir_in_rust`, run it from the TIRx
repository root:

.. tabs::

   .. group-tab:: Rust

      .. code-block:: bash

         python examples/extension/run_example.py

   .. group-tab:: Python

      .. code-block:: bash

         PYTHONPATH=examples/extension/my_ext_py python examples/extension/run_example.py

A successful run prints:

.. code-block:: text

   Complex multiplication matches NumPy.

``tests/python/ir/test_op_rust.py`` runs the script with both versions, and also
checks printing and parsing, invalid calls, chained products, and that lowering
leaves no complex types or operators.

Moving the Prototype to Rust
----------------------------

The two versions register the same type key and operator names, so TVMScript,
printed IR, and ``run_example.py`` work with either: replacing the prototype
with the Rust package leaves the code that uses the extension unchanged. Since
the names are the same, a process loads one of the two. Each part carries over
as follows:

.. list-table::
   :header-rows: 1
   :widths: 18 41 41

   * - Part
     - Rust library
     - Python prototype
   * - Type
     - ``#[rs_class]`` object and ``def_object_ref!`` reference, registered
       with ``register_type``
     - ``@py_class`` subclass of ``ir.Type``, published with ``ir.expose_type``
   * - Operator
     - An ``OpDef`` chain in a ``static_init_block!``
     - ``ir.Op.register``, then ``set_signature``, ``set_attr``, and
       ``expose``
   * - Type inference
     - A Rust function that C++ calls natively, set with ``set_infer_type``
     - A Python function in ``FInferType``
   * - Validation
     - A Rust function, set with ``set_validator``, that checks the arguments
       and the result type
     - The argument count, from the signature; the pass checks the types
   * - Operator lookup
     - ``Op::get``, kept by the pass
     - The ``Op`` that ``Op.register`` returns
   * - Pass
     - ``#[dispatch(map)]`` methods, wrapped by ``FunctionPass::new`` and
       exported
     - ``(type, callback)`` entries for ``tvm_ffi.structural_map``, wrapped
       by ``function_pass``
   * - Loading
     - The package loads the library, whose static init blocks run the
       definitions
     - ``import my_ext`` runs the definitions

Next Steps
----------

Complex addition or conjugation needs one more operator and one more rewrite
rule in ``LowerComplex``, registered and written the same way, in either
version. The example keeps complex values inside a function: parameters and
outputs are ordinary arrays, and complex numbers have no runtime storage format
or calling convention. See the :doc:`Rust API reference <../reference/rust/index>`
for the full Rust API.
