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

.. _expressions-and-types:

Types and Expressions
=====================

In this chapter, we will introduce types and expressions and explain how
they work together in the IR. Types describe properties such as dtypes,
tensor shapes, and function signatures, while expressions represent values
and computations. Compiler passes can work with these objects directly to
inspect or rewrite a program without executing it.

.. _user-ir-type:

Types
-----

``ir.Type`` is the base class for all IR types, including primitive types
for scalar and vector values, tensor types, and function types:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Class
     - Describes
     - Fields
   * - ``ir.PrimType``
     - A scalar or vector dtype
     - ``dtype``
   * - ``ir.PtrType``
     - A pointer's element type and memory scope
     - ``element_type``, ``scope``
   * - ``ir.TupleType``
     - Ordered tuple field types
     - ``fields``
   * - ``ir.FuncType``
     - Function argument and return types
     - ``arg_types``, ``ret_type``
   * - ``tirx.TensorType``
     - A tensor's shape, element type, and storage properties
     - ``shape``, ``dtype``, ``scope``, ``data_alignment``, ``layout``

Next, we will look at two common types in tensor programs in more detail:
``ir.PrimType`` for scalar and vector values, and ``tirx.TensorType`` for tensors.

.. _user-ir-prim-type:

PrimType
~~~~~~~~

``ir.PrimType`` describes integer, floating-point, and boolean values, either
as scalars or vectors. Its ``dtype`` records the element kind, bit width,
and number of lanes.

**Scalar types.** A scalar holds one value. The program declares ``n`` as:

.. literalinclude:: ../get_started/examples/program.py
   :language: python
   :start-at:     n = T.dynamic
   :end-at:     n = T.dynamic
   :dedent: 4

where ``n`` is a 64-bit integer symbol with type ``ir.PrimType("int64")``.

The sigmoid calculation uses ``T.float32(1)`` to create a float32 constant.
Its type, ``T.float32``, corresponds to ``ir.PrimType("float32")``.

**Vector types.** A vector holds several values, called lanes. In the
four-channel path, ``biased`` is declared with a vector type:

.. literalinclude:: ../get_started/examples/program.py
   :language: python
   :start-at:                     biased: T.let
   :end-before:                     Y[i, j : j + T.int64(4)]
   :dedent: 20
   :emphasize-lines: 1

``T.float32x4`` corresponds to ``ir.PrimType("float32x4")``.
``T.let[...]`` declares the intermediate value's type.
The ``x4`` suffix specifies four lanes within a single vector value, so
``biased`` holds four channel values at a time.

.. _user-tensor-type:

TensorType
~~~~~~~~~~

A tensor needs an element type and a shape. ``tirx.TensorType`` stores these
in its ``dtype`` and ``shape`` fields. The element type is a ``PrimType``,
and each dimension is an integer expression.

The input ``X`` is declared as:

.. literalinclude:: ../get_started/examples/program.py
   :language: python
   :start-at:         X: T.Tensor
   :end-at:         X: T.Tensor
   :dedent: 8

where ``X`` is the parameter name and the part after ``:`` is its *type
annotation*, which tells TVMScript what kind of value the parameter accepts.
This annotation becomes a ``tirx.TensorType`` in the IR: ``shape`` describes
``2*n`` rows and ``channels`` columns, and ``dtype`` is
``ir.PrimType("float32")``.

**Symbolic shapes.** A dimension such as ``2*n`` can vary between calls.
In this example, ``channels`` is fixed at 130 when the program is constructed,
while ``n`` is determined from the input's row count at runtime. An input
with shape ``(16, 130)`` gives ``n=8``. A later call can use a different
even row count without recompiling the function. Both fixed shapes such as
``(16, 130)`` and symbolic shapes such as ``(2*n, 130)`` use the same
``TensorType`` representation.

As a type object, ``TensorType`` records the tensor's properties without
allocating storage for its elements. The
:class:`~tvm_tirx.tirx.TensorType` API reference covers the available storage options.

Expressions
-----------

Expressions describe program values and the operations that produce them.
``ir.Expr`` is the base class for all expressions. Its ``ty`` field records
the result type:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Field
     - Value type
     - Meaning
   * - ``ty``
     - ``ir.Type``
     - The type of value represented by the expression.

Expressions include variables, constants, arithmetic, tensor reads, and calls.
These classes store their values or operands in the following fields:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Classes (examples)
     - Represents
     - Additional fields
   * - ``ir.Var``
     - A variable
     - ``name``
   * - ``ir.IntImm``, ``ir.FloatImm``, ``ir.StringImm``
     - Literal constants
     - ``value``
   * - ``ir.prim.Add``, ``ir.prim.Sub``, ``ir.prim.Mul``
     - Binary arithmetic
     - ``a``, ``b``
   * - ``ir.prim.EQ``, ``ir.prim.LT``
     - Comparisons
     - ``a``, ``b``
   * - ``ir.Op``
     - A registered operator
     - ``name``, ``doc``, ``args_info``
   * - ``ir.Call``
     - A call to an operator or function
     - ``op``, ``args``, ``attrs``, ``ty_args``
   * - ``ir.TensorLoad``
     - A tensor element read
     - ``source``, ``indices``

The input parameter ``X`` is represented by an ``ir.Var`` expression.
We can inspect its type through ``program``:

.. literalinclude:: ../get_started/examples/program.py
   :language: python
   :start-at:     function = program["bias_sigmoid"]
   :end-at:     row_extent = tensor_type.shape[0]
   :dedent: 4

``input_var.ty`` is the ``tirx.TensorType`` from the parameter annotation.
``tensor_type.shape[0]`` is an expression inside that type: its ``a`` field
holds the integer constant 2, and its ``b`` field refers to ``n``. The type
of this dimension expression is ``ir.PrimType("int64")``. Its ``loc`` records
where ``2*n`` occurs in the TVMScript annotation.

.. _user-ir-prim-operations:

Primitive Operations
~~~~~~~~~~~~~~~~~~~~

The ``ir.prim`` namespace provides expression nodes for arithmetic,
comparisons, and other primitive computations. Arithmetic nodes include
``Add``, ``Sub``, ``Mul``, and ``Div``, while comparison nodes include ``EQ``,
``LT``, and ``LE``. The same namespace includes boolean logic (``And``, ``Or``, ``Not``),
bitwise operations (``BitwiseAnd``, ``LShift``), dtype conversion (``Cast``),
and value selection (``Select``). The
:doc:`ir.prim </reference/python/ir.prim>` API reference lists the available
nodes and construction helpers.

The source code uses operators such as ``+``, ``<=``, and ``/`` to express
these computations. The four-channel path combines a bounds check with
bias addition and sigmoid:

.. literalinclude:: ../get_started/examples/program.py
   :language: python
   :start-at:                 if j + T.int64(4)
   :end-before:                 else:
   :dedent: 16

In this code, the ``+``, ``<=``, and ``/`` operations become
``ir.prim.Add``, ``ir.prim.LE``, and ``ir.prim.Div`` nodes, respectively.
Each binary node has two operand fields: ``a`` for the expression on its
left and ``b`` for the expression on its right.

**Bounds check.** The condition between ``if`` and ``:`` uses ``<=``, so
its outermost node is ``ir.prim.LE``. Its ``a`` field points to
``j + T.int64(4)``, and its ``b`` field points to ``T.int64(channels)``.
The left expression contains a ``+``, so it is itself an ``ir.prim.Add``.
With ``channels=130``, the condition has this structure:

.. code-block:: text

   ir.prim.LE                  j + 4 <= channels
   ├── a: ir.prim.Add           j + 4
   │   ├── a: ir.Var            j
   │   └── b: ir.IntImm         4
   └── b: ir.IntImm             130

The outer ``a`` points to the whole addition node, which has its own ``a``
and ``b`` fields. The four accessed channels are ``j`` through ``j+3``, so
the group is valid when ``j+4 <= channels``.

**Bias addition.** The ``+`` in the definition of ``biased`` becomes another
``ir.prim.Add``. Its ``a`` field points to the ``X[...]`` tensor read, and
its ``b`` field points to the ``bias[...]`` tensor read. The addition
combines their four channel values element by element.

**Sigmoid.** In the final line, the ``+`` in
``T.float32(1) + T.exp(-biased)`` becomes an ``ir.prim.Add``. The ``/`` becomes
an ``ir.prim.Div`` whose ``a`` operand supplies the numerator, ``1``, and
whose ``b`` operand is the addition that computes the denominator.
The exponential ``T.exp(-biased)`` is represented by an ``ir.Call``, which
we will discuss in the :ref:`user-ir-call` section.

.. _user-ir-const:

Primitive Expressions
~~~~~~~~~~~~~~~~~~~~~

A primitive expression is an expression whose ``ty`` is an ``ir.PrimType``.
The expression may be a variable, a tensor element read, an arithmetic
calculation, or a call. Its result type tells us whether it produces an
integer, a floating-point value, a boolean, or a vector of these values.

**Index expressions.** The loop ``for i in range(2 * n)`` uses ``2*n`` to
determine how many rows to process. The multiplication's ``ty`` is
``ir.PrimType("int64")``. The bounds check
``j + T.int64(4) <= T.int64(channels)`` produces a boolean condition for the
vector or scalar path, so its ``ty`` is ``ir.PrimType("bool")``.

**Floating-point expressions.** The scalar tail computes sigmoid for one
remaining channel ``k`` in row ``i``:

.. literalinclude:: ../get_started/examples/program.py
   :language: python
   :start-at:                         activated_tail: T.let
   :end-before:                         Y[i, k] = activated_tail
   :dedent: 24

The element reads ``X[i, k]`` and ``bias[k]`` each produce one float32 value,
so their ``ty`` is ``ir.PrimType("float32")``. The sum, exponential call, and
division also have this result type. They are all primitive expressions.
The ``T.let[T.float32]`` declaration gives ``activated_tail`` the same
scalar type.

**Vector expressions.** The four-channel path performs the same calculation
on four values at once:

.. literalinclude:: ../get_started/examples/program.py
   :language: python
   :start-at:                     biased: T.let
   :end-at:                     Y[i, j : j + T.int64(4)]
   :dedent: 20

``T.Ramp(j, T.int64(1), 4)`` constructs the indices ``[j, j+1, j+2, j+3]``.
Its ``ty`` is ``ir.PrimType("int64x4")``. Each tensor read returns
four float32 values, so the reads and their sum have type
``ir.PrimType("float32x4")``. This is the type declared for ``biased``.

The vector sigmoid result also has type ``ir.PrimType("float32x4")``.
Its IR uses ``ir.prim.Broadcast`` to repeat the scalar constant
``T.float32(1)`` across four lanes, so it can participate in the vector
addition and division.

.. _user-ir-var:

Var
~~~

``ir.Var`` represents a named value with a type. Variables can be function
parameters, loop indices, or intermediate values defined by bindings.

The program uses variables for its tensor parameters, loop indices, and
intermediate results. Their IR API declarations are:

.. literalinclude:: examples/sigmoid.py
   :language: python
   :start-after: # begin variables
   :end-before: # end variables
   :dedent: 4

``X``, ``bias``, and ``Y`` have tensor types. ``i`` selects a sample, ``j``
selects the start of a four-channel group, and ``k`` indexes any remaining
channels. These indices have int64 types. ``packed_value`` holds four biased
values; ``scalar_value`` holds one activated value in the tail loop.

The same ``n`` symbol appears in the tensor shapes and outer loop bound,
connecting the number of rows in the tensors to the number of loop iterations.

.. _user-ir-call:

Call
~~~~

:class:`ir.Call <tvm_tirx.ir.Call>` represents a call to an operator or function.
Its ``op`` field identifies the callee, and ``args`` holds the argument
expressions. A registered operator is represented by ``ir.Op``.

In sigmoid, ``1 / (1 + exp(-x))``, the exponential is a call inside a larger
arithmetic expression:

.. code-block:: python

   from tvm_tirx import ir, tirx
   from tvm_tirx.script import from_source
   from tvm_tirx.script import tirx as T

   exp_op = ir.Op.get("prim.exp")

   def sigmoid(x):
       one = ir.const(1, x.ty)
       exp_neg_x = ir.Call(exp_op, [-x])
       return one / (one + exp_neg_x)

``exp_op`` is the registered ``ir.Op`` for the exponential operation.
``ir.Call(exp_op, [-x])`` constructs a call with the negated input as its
argument. The sum ``one + exp_neg_x`` forms the denominator; the division
produces sigmoid's result.
The helper works on scalar or vector inputs, preserving their primitive type.

In the scalar tail, the input to the helper is itself an expression:

.. literalinclude:: examples/sigmoid.py
   :language: python
   :start-after: # begin biased activation
   :end-before: # end biased activation
   :dedent: 4

``activation`` defaults to the ``sigmoid`` helper. ``biased_value`` is an
``ir.prim.Add`` containing two tensor reads. ``result`` is an ``ir.prim.Div``
whose denominator contains another addition and the exponential call. The
code in the scalar tail binds this result and stores it in ``Y``. The
four-lane path uses the same helper with ``packed_value`` instead.

The helper constructs the IR for this computation. When the compiled program
runs, each loop iteration reads the input and bias values, computes the
activation, and writes it to ``Y``. The biased values stay local to that
iteration, so bias addition and sigmoid share one traversal without an
intermediate tensor.
