# Symbo — A Hybrid Generative Symbolic Reasoning Engine
### © 2025 Damien Davison & Michael Maillet & Sacha Davison
### Recursive AI Devs  
Licensed under the Apache License, Version 2.0
psychcoherence@gmail.com OR therealmichaelmaillet@gmail.com


Symbo is a hybrid symbolic–numeric reasoning system built from the ground up by
deconstructing **318 classical algorithms** across algebra, optimization, dynamics,
and computational geometry.

Instead of adopting any algorithm wholesale, each was **reduced to its atomic
computational components**, de-parented from its original context, and evaluated
for suitability within a **generative symbolic algebra architecture**. These
primitives were then recombined into a unified framework capable of:

- generating Taylor-manifold policy functions,
- evaluating symbolic tensors with exact semantics,
- performing Gröbner-based constraint solving with streaming output,
- conducting second-order perturbation analysis,
- producing neural-assisted approximate solutions,
- running pathfinding over symbolic energy landscapes to perform reasoning,
- and enabling browser-side inference through WASM.

The result is not a clone of existing tools, nor a traditional CAS, nor a neural network.
It is a **new model class**: a nano-scale symbolic generative engine that can be trained,
fitted, perturbed, serialized, visualized, and reasoned over — all while maintaining
mathematical interpretability.

Symbo provides:

- A true n-dimensional symbolic tensor type
- A generative Taylor-expansion core 
- A multi-method coefficient solver (Gröbner bases, perturbation theory, least squares)  
- Arrow/MessagePack serialization for high-speed I/O  
- WASM-friendly execution for browser runtimes  
- Manifold-based reasoning via A* pathfinding
- Explainability through derivative trees and variable influence mapping

This repository contains the full implementation, usage examples, demos, and
a high-level description of the architecture derived from the algorithmic decomposition
process.

---

## Install

Symbo is a normal Python package (`pyproject.toml`); Python 3.10+ and SymPy are all
the core needs.

```bash
pip install -e ".[dev]"                # engine + pytest + ruff
pip install -e ".[viz,io,opt,kb,dev]"  # add plotting, Arrow/msgpack, GP search, kanren
pip install -e ".[all]"                # everything, including torch
```

Only `sympy`, `numpy` and `networkx` are required. Every other package is an extra;
using a feature whose extra is missing raises `MissingOptionalDependency` (an
`ImportError` subclass) that names the extra to install, and `symbo --check` prints
which features your interpreter can currently use.

## Quickstart

```python
import sympy as sp
from symbo import NanoTensor, SymbolicTensor, TaylorExpansion

x, y = sp.symbols("x y")

# 1. Generative Taylor core: build the symbolic ansatz, then bind coefficients.
taylor = TaylorExpansion([x, y], {x: 0.0, y: 0.0}, max_order=2)
taylor.generate("g")                        # g_0 + g_x*x + g_y*y + g_x_x*x**2/2 + ...
print(taylor.coefficient_names)
policy = taylor.to_policy_function(
    {name: float(i) for i, name in enumerate(taylor.coefficient_names)})
print(policy(x=1.0, y=0.5))

# 2. Exact n-dimensional tensor algebra.
A = SymbolicTensor.from_nested([[x, 1], [y, x * y]], name="A")
print((A @ A).to_nested())

# 3. NanoTensor: a scalar symbolic tensor with caching, differentiation and fitting.
nt = NanoTensor((1,), max_order=2, base_vars=["x", "y"])
nt.data[0] = sp.sin(x) * sp.cos(y)
print(nt.diff("x").data[0])                  # differentiation never mutates nt
```

```text
['g_0', 'g_x', 'g_y', 'g_x_x', 'g_y_y', 'g_x_y']
6.5
[[x**2 + y, x*(y + 1)], [x*y*(y + 1), y*(x**2*y + 1)]]
cos(x)*cos(y)
```

For a worked end-to-end example — a real RBC model solved to second order with
residual verification — see *§5 Second-Order Perturbation* in
[`MODULES.md`](MODULES.md), or run the bundled demo:

```bash
python -m symbo --demo rbc       # fit the RBC policy and (with the viz extra) plot it
python -m symbo --demo pipeline  # RBC + Kamke ADE + benchmark + knowledge base
python -m symbo --check          # environment / feature report
```

## Development

```bash
python -m pytest                       # unit tests; docs examples are tests too
python -m pytest -q -m "not slow"       # skip the multi-second symbolic solves
python -m pytest --doctest-modules symbo -q
python -m ruff check symbo tests demo_military_grade.py stress_test.py
```

Continuous integration runs lint, a core-only matrix (Python 3.10 and 3.13, no
extras installed), an extras job, a torch job and a wheel build. The workflow is
staged at [`.ci/ci.yml`](.ci/ci.yml) — move it to `.github/workflows/ci.yml` to
enable it (the bot that prepared the change may not write to that directory).

## Where to read next

| document | contents |
|---|---|
| [`MODULES.md`](MODULES.md) | every module: API, runnable examples, gotchas, extras |
| [`HANDOFF.md`](HANDOFF.md) | current state, decisions, known limitations, how to verify |
| [`README_MILITARY_GRADE.md`](README_MILITARY_GRADE.md) | the hardened `NanoTensor` (validation, health, recovery) |
| [`IMPLEMENTATION_SUMMARY.md`](IMPLEMENTATION_SUMMARY.md) | module-by-module implementation notes |
| [`QUALITY-AUDIT.md`](QUALITY-AUDIT.md) | an earlier audit of the pre-0.2 layout (some findings no longer apply — `HANDOFF.md` says which) |
| [`docs/`](docs) | whitepaper, architecture essay, owner's manual |

---

Project Authors

Damien Davison — Architect, Symbolic Systems

Michael Maillet — Architect, Computational Structures

Sacha Davison — Architect, Project Manager

Recursive AI Devs — Foundational Research Team

Symbo is part of the Recursive AI Devs model ecosystem:
- Symbo (symbolic engine)
- FortArch (encrypted container)
- Topo (topological reasoning)
- Chrono (temporal propagation engine)
- Morpho (transformational generative engine)

---

Support & Contribution

Issues and feature requests are welcome.
Pull requests are reviewed for mathematical integrity, symbolic correctness, and
architectural compatibility.

If you use Symbo in research or production, please cite and attribute the authors.

---

## License — Apache 2.0

This project is licensed under the **Apache License, Version 2.0**, which permits:

- Commercial use  
- Private modifications  
- Distribution  
- Patent protection  
- Use in closed-source or open-source projects  

Copyright 2025
Damien Davison & Michael Maillet & Sacha Davison
Recursive AI Devs

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---

## NOTICE

This product includes original work by:  
**Damien Davison & Michael Maillet & Sacha Davison (Recursive AI Devs)**  
Additional details can be found in the project's LICENSE and source headers.
