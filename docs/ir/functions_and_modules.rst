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

Functions and Modules
=====================

In this chapter, we will introduce functions and modules as the structures
that organize IR programs. A function combines parameters, a result type,
and a statement body; a module collects named function definitions. Compiler
passes use these structures to inspect a function's interface and
implementation, locate definitions through global references, and replace
an implementation while preserving references to it.

Functions
---------

``ir.BaseFunc`` is the base class for all function definitions and is also
an expression. Its ``attrs`` field holds named function properties:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Field
     - Value type
     - Meaning
   * - ``attrs``
     - ``ir.DictAttrs``
     - Named function properties.

:class:`tirx.Function <tvm_tirx.tirx.Function>` extends ``ir.BaseFunc`` with
parameters, a result type, and a statement body:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Meaning
   * - ``params``
     - Parameter variables in calling order.
   * - ``ret_type``
     - The declared result type.
   * - ``body``
     - A ``SeqStmt``, or None for a declaration without an implementation.

The parameter and result types form the function's signature, available as
an ``ir.FuncType`` through ``ty``. The ``global_symbol`` attribute gives the
function an exported name for code generation.

The program defines a function with an input matrix, a bias vector, and an
output matrix. Its IR definition combines these three parameters with
the batch and channel loops:

.. literalinclude:: examples/sigmoid.py
   :language: python
   :start-after: # begin function construction
   :end-before: # end function construction
   :dedent: 4

``batch_loop`` contains the four-channel path and scalar tail. The function
writes activated values into ``Y``; ``ir.TupleType([])`` declares that it
returns no value.

The input and output types have shape ``(2*n, channels)``. The symbol ``n``
is used in these types and the loop bound; it is not a separate parameter.
The compiled function derives its value from the tensor shapes at runtime.

Modules
-------

:class:`ir.IRModule <tvm_tirx.ir.IRModule>` maps ``ir.GlobalVar`` references to
``ir.BaseFunc`` definitions. An ``ir.GlobalVar`` identifies a function within
the module. Its definition can be replaced while that reference stays valid,
so callers do not need to change when a pass updates the implementation.

In the bias + sigmoid example, one module keeps two model specializations.
The same factory builds kernels for 128-channel and 130-channel outputs,
each with its own exported name:

.. literalinclude:: examples/sigmoid.py
   :language: python
   :start-after: # begin module construction
   :end-before: # end module construction

Both accept tensors with a runtime row count of ``2*n``. The 128-channel
specialization processes complete four-element groups; the 130-channel
specialization also handles two residual channels. Their tensor types record
the different widths.

The module provides a global reference for each specialization:

.. literalinclude:: examples/sigmoid.py
   :language: python
   :start-after: # begin module lookup
   :end-before: # end module lookup

``entry`` identifies the function; ``module[entry]`` and
``module["bias_sigmoid_130"]`` retrieve its current definition.

For example, a compiler can keep the original expanded sigmoid and build
another module that uses the named ``prim.sigmoid`` operator. The factory's
``activation`` argument selects this representation:

.. literalinclude:: examples/sigmoid.py
   :language: python
   :start-after: # begin module update
   :end-before: # end module update

``variant`` keeps the 128-channel definition and replaces the 130-channel
definition with one that calls ``prim.sigmoid``. The replacement keeps the
same tensor interface and exported name, so ``variant[entry]`` now retrieves
the new implementation through the existing global reference. The original
``module`` still contains the expanded sigmoid expression.
