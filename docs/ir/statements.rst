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

.. _statements-and-control-flow:

Statements
==========

In this chapter, we will introduce the statement nodes that organize a
function's execution. Statements describe value bindings, tensor writes,
and control flow. Statement sequences record execution order, while branches
and loops combine statements into program bodies. Compiler passes can use
these structures to construct and transform programs, including bodies
defined by custom operators.

.. _statements-and-sequences:

Statement Nodes
---------------

``ir.Stmt`` is the base class for all statements, from bindings and tensor
writes to sequences, branches, and loops:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Class
     - Represents
     - Fields
   * - ``ir.Bind``
     - A variable definition
     - ``var``, ``value``
   * - ``ir.TensorStore``
     - A tensor element write; the destination type must support stores
     - ``dest``, ``indices``, ``value``
   * - ``ir.Evaluate``
     - Expression evaluation with its result discarded
     - ``value``
   * - ``ir.AssertStmt``
     - A program assertion
     - ``condition``, ``error_kind``, ``message_parts``
   * - ``ir.Return``
     - A function return
     - ``value``
   * - ``ir.Break``
     - Exit the current loop
     - None
   * - ``ir.Continue``
     - Start the next loop iteration
     - None
   * - ``ir.SeqStmt``
     - A sequence of statements in execution order
     - ``seq``
   * - ``ir.If``
     - Select a branch
     - ``condition``, ``then_case``, ``else_case``
   * - ``ir.For``
     - Repeat a body over an integer range
     - ``var``, ``begin``, ``end``, ``step``, ``kind``, ``body``, ``annotations``
   * - ``ir.While``
     - Repeat while a condition is true
     - ``condition``, ``body``
   * - ``ir.RegionStmt``
     - An operator-defined statement with a nested body
     - ``op``, ``args``, ``attrs``, ``body_params``, ``body``, ``result_vars``

Statement Sequences
-------------------

``ir.SeqStmt`` groups statements in execution order in its ``seq`` field.
A sequence can serve as a function body or as the body of a branch or loop.
It introduces no scope of its own.

In the program's four-channel path, a binding makes the biased values
available to the following output store:

.. literalinclude:: examples/sigmoid.py
   :language: python
   :start-after: # begin statement sequence
   :end-before: # end statement sequence
   :dedent: 4

``biased_values`` is the expression ``X[i, indices] + bias[indices]``.
``ir.Bind`` defines the four-lane variable ``packed_value`` with that value.
The following ``TensorStore`` applies the activation and writes four elements
to ``Y``. Both the bound variable and its value have type ``float32x4``.

The sequence belongs to the vector branch, which determines the binding's
scope.

Control Flow and Scope
----------------------

Control-flow statement nodes include ``ir.If`` for branching, ``ir.For`` and
``ir.While`` for loops, ``ir.Break`` and ``ir.Continue`` for loop control, and
``ir.Return`` for returning from a function.

The statements inside a branch or loop form its body. Branch bodies are
stored in ``If.then_case`` and ``If.else_case``, while loop bodies are stored
in ``For.body`` and ``While.body``. Each body is a ``SeqStmt`` and can contain
further branches and loops.

An ``ir.For`` specifies a loop variable, begin and end bounds, a step, and an
execution kind. The end bound is exclusive. ``ForKind.DEFAULT`` executes
iterations in order; the :class:`~tvm_tirx.ir.ForKind` API reference describes
the other execution kinds.

Each vector access in this example requires four valid channels.
In the IR API version, the condition
``full_vector = j + four <= end`` checks this before selecting the vector or
scalar path. Here, ``j`` is the start of the current group, ``four`` is an
int64 constant, and ``end`` is the channel count:

.. literalinclude:: examples/sigmoid.py
   :language: python
   :start-after: # begin branch
   :end-before: # end branch
   :dedent: 4

For 128 channels, every group is complete. For 130 channels, the group starting
at ``j=128`` has only two elements. The else branch handles those elements
with a scalar loop, keeping the vector access within the tensor's bounds:

.. literalinclude:: examples/sigmoid.py
   :language: python
   :start-after: # begin scalar tail
   :end-before: # end scalar tail
   :dedent: 4

``k`` starts at ``j`` and stops before ``end``. In the 130-channel case, it
processes channels 128 and 129. The binding defines one activated value, then
the store writes it to ``Y[i, k]``.

The surrounding loops cover samples and channel groups:

.. literalinclude:: examples/sigmoid.py
   :language: python
   :start-after: # begin loop
   :end-before: # end loop
   :dedent: 4

``batch_loop`` advances ``i`` by one sample. ``channel_loop`` advances ``j``
by four channels, selecting the vector or scalar body at each step. Both
loops use int64 variables, bounds, and steps. The vector operations inside
an iteration are represented by their four-lane expressions.

Scope determines where a definition can be used. A variable defined inside
a branch or loop is available only within that body. In the example,
``packed_value`` is defined inside the vector branch, while ``scalar_value``
is defined inside the tail loop. Neither is available after its body. The
loop variables ``i``, ``j``, and ``k`` are likewise visible only within their
loop bodies.

Some operators introduce their own statement bodies.
:class:`ir.RegionStmt <tvm_tirx.ir.RegionStmt>` represents these using a
registered operator, whose definition specifies the body's parameters.

These scope rules also apply to programs built through the IR API. After
construction, :func:`tirx.analysis.check_well_formed <tvm_tirx.tirx.analysis.check_well_formed>`
can check a function or module for invalid variable and tensor uses and
misplaced loop-control statements.
