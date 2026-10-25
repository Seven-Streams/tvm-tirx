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

Your First TIRx Program
=======================

In this chapter, we'll write and run a small TIRx program, then look at the
IR it builds. Our example adds a bias to each channel and applies sigmoid,
a common activation in neural networks. For each input element, we'll compute
``Y[i, j] = 1 / (1 + exp(-(X[i, j] + bias[j])))``.

Each row of ``X`` is a sample, and each column is a channel. The same bias
vector is shared across all rows. We can compute the addition and sigmoid
in one kernel, so there's no need to store the biased values in a separate
tensor.

We'll use TVMScript, TIRx's Python-based notation for IR. On the first read,
you can follow the data from ``X`` and ``bias`` through ``biased`` to ``Y``.

With an LLVM-enabled TIRx build, you can download
:download:`program.py <examples/program.py>` and run ``python program.py``.
Here's the complete program:

.. literalinclude:: examples/program.py
   :language: python
   :start-after: # begin program
   :end-before: # end program

Here, ``@T.function`` turns the bias + sigmoid definition into IR. The code
in ``bias_sigmoid`` processes four channels at a time and uses a scalar loop
for the remaining channels. ``run()`` compiles it for the CPU and checks its
output against NumPy. We'll revisit parts of this program in the
:doc:`IR Walkthrough </ir/overview>` to explain the IR objects behind them.
