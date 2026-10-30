<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# TIRx Extension Example

A Python package, `my-ext`, that extends TIRx from Rust: a complex-number type,
operators that build and multiply complex values, and a pass that lowers them to
scalar arithmetic. Importing `my_ext` loads the Rust library, provides the
`my_ext.*` builders for TVMScript, and registers the `my_ext` compilation
pipeline. The tutorial in `docs/advanced/` walks through the code.

- `src/lib.rs`: the type, operators, and pass, built by cargo as a shared library
- `CMakeLists.txt`, `pyproject.toml`: build the library and install it into the package
- `python/my_ext/`: loads the library and registers the pipeline
- `run_example.py`: compiles a function that uses the extension and checks it against NumPy

## Install and Run

With Rust 1.84 or newer, from the repository root:

```bash
pip install ./examples/extension
python examples/extension/run_example.py
```

The build compiles the library against the published `tvm-tirx` crate and the
`apache-tvm-tirx` package, which the extension also depends on at run time.
