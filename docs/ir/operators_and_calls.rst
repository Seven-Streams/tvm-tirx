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

Operators and Calls
===================

In this chapter, we will introduce operators and calls, from an operator's
shared definition to the arguments and result type of an individual call.
We will examine argument requirements, type inference, validation, and
effects. Built-in and custom operators share the same IR representation,
so compiler passes can inspect and transform both through the existing
interfaces.

Operators
---------

``ir.Op`` is an expression that identifies a registered computation. We can
retrieve it with ``ir.Op.get(name)`` and inspect ``args_info`` for the expected
inputs.

Operator attributes describe properties shared by all calls to that operator.
The ``TCallEffectKind`` attribute records whether a call only computes a value
or can read or change program state.

The program uses the same exponential operator for scalar tail elements
and four-lane channel groups. Its shared argument and effect information
can be inspected through the operator:

.. literalinclude:: examples/sigmoid.py
   :language: python
   :start-after: # begin operator lookup
   :end-before: # end operator lookup

``argument_names`` is ``["x"]``, and ``effect`` is
``ir.CallEffectKind.Pure``. Each call computes an exponential from its input;
it does not read or change external state. The tensor loads supplying that
input are separate expressions. The :class:`~tvm_tirx.ir.Op` API reference
describes its argument information and attributes.

Calls
-----

:class:`ir.Call <tvm_tirx.ir.Call>` represents an invocation of an operator or
function. The expression being called is its *callee*. The call records that
callee, its arguments, and any attributes for the invocation:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Meaning
   * - ``op``
     - The callee: a registered operator or another function-valued expression.
   * - ``args``
     - Value arguments in calling order.
   * - ``ty_args``
     - Explicit type arguments, empty by default.
   * - ``attrs``
     - Attributes for this invocation, or None. A dictionary becomes
       ``ir.DictAttrs``.

In the bias + sigmoid example, the activation can be an expanded expression,
``1 / (1 + exp(-x))``, or a single call to ``prim.sigmoid``. A named
activation operator lets a later compiler pass choose its implementation:

.. literalinclude:: examples/sigmoid.py
   :language: python
   :pyobject: intrinsic_sigmoid

``intrinsic_sigmoid(x)`` constructs an ``ir.Call`` to the registered
``prim.sigmoid`` operator. The call's result has the same type as ``x``:
float32 for a scalar input or float32x4 for a four-lane input.

The constructor records the call as IR. The activation is evaluated when the
compiled program runs. Calls can also target function-valued expressions,
including global references to definitions in a module.

.. _types-and-validation:

Type Inference and Validation
-----------------------------

``ir.Call`` can infer its result type from the operator's registered rule or
the callee's function signature, so each call does not need a separate type
annotation. The constructor infers the type when ``ty`` is omitted and
stores the supplied type when ``ty`` is explicit. ``MissingType`` marks a
result whose type has not yet been determined.

Validation checks whether the arguments meet the operator's requirements.
It is separate from construction, so tools can assemble a call before
checking its arguments with :meth:`call.validate() <tvm_tirx.ir.Call.validate>`.
Type inference can also reject inputs while computing the result type.

A rewrite can accidentally drop an input while retaining the old result
type. In the bias + sigmoid example, the constructor can record a vector
exponential with no input and an explicit float32x4 type:

.. literalinclude:: examples/sigmoid.py
   :language: python
   :start-after: # begin validation failure
   :end-before: # end validation failure
   :append: invalid.validate()

``invalid.validate()`` raises ``TypeError`` because ``prim.exp`` requires
one input argument.

When a pass builds vector operations, it must also update result types to
match the vector inputs. In the four-channel path, ``exp_call`` represents
``T.exp(-biased)``, whose input is a float32x4 vector. The following code
reconstructs the call with a missing type, then uses
:func:`ir.reinfer_type <tvm_tirx.ir.reinfer_type>` to infer its result type:

.. literalinclude:: examples/sigmoid.py
   :language: python
   :start-after: # begin provisional call
   :end-before: # end provisional call

``inferred`` is ``ir.PrimType("float32x4")``, matching the input vector's type.
The inference function returns this type without changing ``provisional``;
``rebuilt_call`` stores it in its ``ty`` field. The rebuilt call can then be
checked with ``rebuilt_call.validate()`` before a pass uses it in place of
the original.
