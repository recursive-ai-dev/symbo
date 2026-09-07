# Symbo Module Architecture

This document describes the modular structure of Symbo and provides usage examples for each component.

## Overview

Symbo is organized into specialized modules, each addressing a specific aspect of symbolic-numeric computation:

```
symbo/
├── primitives.py         # Atomic computational operations
├── tensor.py            # N-dimensional symbolic tensors
├── wasm_bindings.py     # WASM-compatible interfaces
├── ecosystem.py         # Integration with other AI models
├── generative/
│   └── taylor.py        # Taylor expansion and policy functions
├── solver/
│   └── groebner.py      # Gröbner basis solver with streaming
├── analytics/
│   ├── perturbation.py  # Second-order perturbation analysis
│   └── explain.py       # Derivative trees for explainability
├── reasoning/
│   └── a_star.py        # A* pathfinding on symbolic landscapes
└── io/
    └── serialization.py # Arrow/MessagePack serialization
```

## Core Modules

### 1. Atomic Primitives (`primitives.py`)

Atomic operations used as building blocks of the engine. There is no published
mapping from a "318 classical algorithms" figure onto these primitives; see
[`docs/algorithm-corpus.md`](docs/algorithm-corpus.md).

**Usage:**
```python
from symbo.primitives import AtomicPrimitives, add, mul, diff
import sympy as sp

prims = AtomicPrimitives()
x, y = sp.symbols('x y')

# Algebraic operations
result = prims.symbolic_add(x, y)
product = prims.symbolic_mul(x, y)

# Differential operations
expr = x**2 + y**2
gradient = prims.gradient(expr, [x, y])
hessian = prims.hessian(expr, [x, y])

# Matrix operations
M = sp.Matrix([[x, y], [y, x]])
det = prims.matrix_det(M)
eigenvals = prims.matrix_eigenvals(M)
```

**Key Features:**
- 35+ atomic operations
- Algebraic: add, mul, pow, div
- Differential: diff, gradient, hessian, jacobian
- Tensor: contraction, outer product, trace
- Polynomial: expand, factor, coefficients
- Matrix: determinant, inverse, eigenvalues

### 2. Symbolic Tensors (`tensor.py`)

True n-dimensional symbolic tensors with complete tensor algebra.

**Usage:**
```python
from symbo.tensor import SymbolicTensor
import sympy as sp

# Create 3x3 matrix
T = SymbolicTensor((3, 3), name="A")
T.fill_with_symbols("a")

# Tensor operations
T2 = SymbolicTensor((3, 3), name="B")
T2.fill_with_symbols("b")

# Outer product
outer = T.outer_product(T2)  # Shape: (3, 3, 3, 3)

# Trace
trace = T.trace(0, 1)

# Contraction (generalized)
C = T.contract(T2, (1,), (0,))  # Matrix multiplication

# Arithmetic
result = T + T2
scaled = T * 2

# Differentiation
x, y = sp.symbols('x y')
dT = T.diff(x)

# Construct from expressions, and contract with @
A = SymbolicTensor.from_nested([[x, 1], [y, x * y]], name="A")
B = SymbolicTensor.from_matrix(sp.Matrix([[1, 2], [3, 4]]), name="B")
AB = A @ B                       # same as A.contract(B, (1,), (0,))

# Numeric evaluation over a whole tensor at once
values = B.eval_numeric({x: 1.0, y: 2.0})
```

**Key Features:**
- Arbitrary rank (0D to ND)
- Complete tensor algebra
- Symbolic arithmetic
- Differentiation
- Numeric evaluation

**Notes:**
- `add`, `sub`, `mul`, `div`, `outer_product`, `diff`, `subs`, `simplify` and the
  operators return **new** tensors; nothing is mutated in place (this differs from
  `NanoTensor.diff`/`simplify`, which update the tensor you called them on).
- Shapes are validated on construction and on every binary operation, so a
  mismatched `T + T2` raises `ValueError` instead of broadcasting.
- `eval_numeric` is strict: every free symbol of the tensor must appear in the
  substitution mapping, otherwise it raises `KeyError` rather than evaluating
  partially. The result is an `object` ndarray shaped like the tensor.
- `SymbolicTensor` has no `__eq__`; compare with
  `sp.simplify(A.to_matrix() - B.to_matrix()) == 0` or `SymboSerializer.verify_equivalence`.

### 3. Taylor Expansion Core (`generative/taylor.py`)

Arbitrary-order multivariate Taylor expansions for policy functions.

**Usage:**
```python
from symbo.generative.taylor import TaylorExpansion, PolicyFunction
import sympy as sp

x, y = sp.symbols('x y')

# Create expansion around (0, 0)
taylor = TaylorExpansion(
    variables=[x, y],
    center={x: 0, y: 0},
    max_order=2
)

# Generate the symbolic ansatz
poly = taylor.generate("g")
# Result: g_0 + g_x*x + g_y*y + g_x_x*x**2/2 + g_x_y*x*y + g_y_y*y**2/2
taylor.coefficient_names        # ('g_0', 'g_x', 'g_y', 'g_x_x', 'g_x_y', 'g_y_y')

# Supply every coefficient the ansatz declares
coeffs = {name: float(i) for i, name in enumerate(taylor.coefficient_names)}
policy = taylor.to_policy_function(coeffs)
value = policy(x=1.0, y=0.5)             # keyword arguments, one per variable

# WASM export: coefficient_map gives the multi-index per name, coefficient_values
# the numbers, so a browser never has to parse a SymPy expression
payload = taylor.to_wasm_json()
```

**Key Features:**
- Arbitrary order (1st, 2nd, 3rd, ...)
- Multivariate support
- Symbolic policy functions
- WASM serialization
- Fast compiled evaluation

**Notes:**
- Coefficient names join the differentiation variables with a single underscore:
  the second derivative w.r.t. `x` is `g_x_x`, *not* `g_xx`. `generate_taylor` on
  `NanoTensor` uses the same convention, and the `symbol` prefix you pass
  (`"g"` above) only names the family.
- `generate()` must run before `coefficient_names`, `to_policy_function()` or
  `to_wasm_json()`; each raises a `RuntimeError` that says so.
- The factorial is already divided out, so `coefficient(g_x_x)` is `x**2/2` and a
  coefficient value is the *derivative value* itself.
- `substitute_coefficients(values)` returns a new expansion; pass `apply=True` to
  update `self.expansion` in place.
- `PolicyFunction` is callable with keywords only and keeps exact SymPy numbers
  available (`policy.expression`); `policy.coefficients` is the mapping you passed.

### 4. Gröbner Basis Solver (`solver/groebner.py`)

Streaming Gröbner basis computation with edge case handling. `NanoTensor.groebner_solve`
returns all **real** solutions of a **zero-dimensional** system; inconsistent
systems and positive-dimensional ideals raise instead of returning `[]`.

**Usage:**
```python
from symbo.solver.groebner import StreamingGröbnerSolver, solve_with_groebner
import sympy as sp

x, y, z = sp.symbols('x y z')

# Define polynomial system
polys = [
    x**2 + y**2 - 1,
    x - y
]

# Non-streaming solve
solutions = solve_with_groebner(polys, [x, y])

# Streaming solve
solver = StreamingGröbnerSolver(polys, [x, y])

for chunk in solver.stream_basis():
    print(chunk['type'])
    if chunk['type'] == 'basis_chunk':
        for item in chunk['items']:
            print(f"  Poly {item['index']}: {item['poly_str']}")

# Get solutions
for solution in solver.stream_solutions():
    print(solution)
```

**Key Features:**
- Streaming output
- Real-time progress
- Infinite solution handling
- Multiple orderings (lex, grlex, grevlex)
- Edge case graceful handling

### 5. Second-Order Perturbation (`analytics/perturbation.py`)

Full second-order perturbation analysis for dynamical systems.

The solver needs a *complete* model: one equation per endogenous variable, with
the lead variables named `*_next`. Writing a single Euler equation is not enough —
the solver reports the system as under-determined rather than inventing a solution.

**Usage:**
```python
from symbo.analytics.perturbation import SecondOrderPerturbation, perturbation_solve
import sympy as sp

# state k, control c, exogenous shock a (a log-deviation, so TFP is exp(a))
alpha, beta, delta, rho = sp.symbols('alpha beta delta rho')
k, c, a = sp.symbols('k c a')
kp, cp, ap = sp.symbols('k_next c_next a_next')

equations = [
    # Euler equation
    c**(-1) - beta * cp**(-1) * (alpha * sp.exp(ap) * kp**(alpha - 1) + 1 - delta),
    # resource constraint (y = exp(a) k^alpha)
    sp.exp(a) * k**alpha + (1 - delta) * k - c - kp,
    # law of motion of the shock
    ap - rho * a,
]
params = {alpha: 0.36, beta: 0.99, delta: 0.08, rho: 0.9}

solver = SecondOrderPerturbation(equations, [k], [c], [a], params,
                                shock_persistence={a: rho})
print(solver.compute_steady_state())        # {k: 8.708..., c: 1.4829..., a: 0.0}
print(solver.determined, solver.n_effective_equations)   # True 2  (preflight check)

solution = solver.solve(order=2, variance=1.0, verify_at=(1e-3, 1e-2, 1e-1))
print(solution.diagnostics["determined"], solution.diagnostics["residual_h0.001"])
print(solution.coefficients['g_k_k'])        # ~0.91 (capital persistence)
print(solution.diagnostics['residual_h0.001'])
print(solution.policy_expression('c'))       # the c policy as a SymPy expression

# or one-shot, without diagnostics:
quick = perturbation_solve(equations, [k], [c], [a], params,
                           shock_persistence={a: rho}, order=1)
```

**Key Features:**
- First and second-order approximations
- Steady-state computation
- Risk/variance corrections
- Policy function generation
- Coefficient solving

**Notes:**
- `variance` is the shock variance used for the second-order risk correction: a
  scalar applies to every shock, a `{shock: value}` dict per shock, `None` leaves
  the correction symbolic. `solution.risk_corrections['h_c_sigma_sigma']` is half of
  `g_c_a_a`, and the corrections are linear in the variance.
- First-order conditions are quadratic in the policy coefficients (expectational
  consistency). The solver uses damped Newton plus a Blanchard–Kahn filter
  (state-transition eigenvalues inside the unit circle), not a one-shot linear
  solve. Residuals are evaluated on the *raw* user equations, independently of
  the substitution used to form the conditions.
- `verify_at` (scalar or sequence of perturbation scales) fills `solution.diagnostics`
  with `residual_h<scale>` and `scaling_exponent`; residuals are what makes a
  solution trustworthy, so prefer it over trusting the linear algebra.
- `policy_functions` and `risk_corrections` are keyed by **string** variable name,
  while `first_order`/`second_order`/`coefficients` are keyed by the SymPy symbol
  name you gave the derivative (`g_k_a`). `coefficients` is the read-only merged
  view of the two orders.
- `shock_persistence` values may be numbers or symbols that also appear in
  `parameters` (resolved automatically).
- A solve that cannot be verified logs a warning and reports the residual as
  `None`; it never silently omits the diagnostic.

### 6. A* Pathfinding (`reasoning/a_star.py`)

Pathfinding on energy landscapes. Grid search is A* with an admissible
Manhattan heuristic on non-negative costs (optimal) and SPFA/Bellman-Ford when
cells can be negative. The symbolic pathfinder uses an L1 heuristic that is a
lower bound on `edge_cost`; it returns a *low-cost* path, not a marketing
"optimal path" over an arbitrary heuristic blend.

**Usage:**
```python
from symbo.reasoning.a_star import SymbolicAStarPathfinder, SymbolicEnergyLandscape
import sympy as sp

x, y = sp.symbols('x y')

# Define energy landscape
energy = x**2 + y**2 - 2*x*y
landscape = SymbolicEnergyLandscape(energy, [x, y])

# Create pathfinder
pathfinder = SymbolicAStarPathfinder(
    landscape=landscape,
    variables=[x, y],
    bounds={x: (-5, 5), y: (-5, 5)},
    step_size=0.1,
    mode='minimize'
)

# Find path
path = pathfinder.find_path(
    start={x: -2.0, y: -2.0},
    goal={x: 2.0, y: 2.0}
)

# Analyze path
analysis = pathfinder.analyze_path(path)
print(f"Path length: {analysis['length']}")
print(f"Total cost: {analysis['cost']}")
print(f"Energy change: {analysis['energy_change']}")
```

**Key Features:**
- Symbolic state representation
- Energy-based cost functions
- Variable influence heuristics
- Manifold-aware search
- Path analysis

### 7. Derivative Trees (`analytics/explain.py`)

Explainability through derivative tree visualization.

**Usage:**
```python
from symbo.analytics.explain import DerivativeTree, derivative_tree
import sympy as sp

x, y, z = sp.symbols('x y z')

# Define complex expression
expr = x**2 * sp.sin(y) + sp.exp(x*z)

# Build derivative tree
tree = DerivativeTree(expr, [x, y, z])
tree.build(max_depth=2)

# Get influence ranking (pass an evaluation point for meaningful numbers)
tree = derivative_tree(expr, [x, y, z], evaluation_point={x: 1.0, y: 0.5, z: 0.0})
ranking = tree.get_influence_ranking()
for var, score in ranking:
    print(f"{var}: {score}")

# ... or the path-independent first-order sensitivities
print(tree.direct_sensitivities())

# Export to Graphviz
tree.export_graphviz("tree.dot")

# Export to JSON for web viz
json_str = tree.to_json()

# Text summary
print(tree.visualize_influence())
```

**Key Features:**
- Full derivative tree construction
- Variable influence scoring
- Graphviz export
- JSON export for web
- Chain rule tracking

**Notes:**
- Pass `evaluation_point=` if you want numbers: without it each edge weight falls
  back to an operation-count *complexity* proxy, which compares structure rather
  than sensitivity. Keys may be symbols or names.
- `get_influence_ranking()` sums over every path from the target to the variable,
  so a variable that appears deep in the graph is counted once per path. Use
  `direct_sensitivities()` for the plain `|df/dv|` ranking.
- `export_graphviz(path)` writes DOT directly (no `graphviz`/`pydot` install
  needed) and returns the path it wrote.
- Nodes are identified by their content (`type:label:expression`), so repeated
  derivatives of the same expression share one node instead of duplicating it.

### 8. WASM Bindings (`wasm_bindings.py`)

WASM-*friendly* interfaces (JSON/msgpack-serializable signatures). No `.wasm`
artifact or browser bundle is produced by this repository.

**Usage:**
```python
from symbo.wasm_bindings import WASMInterface, create_browser_test_payload

# Evaluate expression
result = WASMInterface.eval_expression(
    "x**2 + 2*x + 1",
    {"x": 3.0}
)

# Differentiate
deriv = WASMInterface.differentiate("x**2 + y", "x")

# Simplify
simplified = WASMInterface.simplify("(x + y)**2")

# Solve
solutions = WASMInterface.solve_equation("x**2 - 4", "x")

# Create browser test
payload = create_browser_test_payload()
```

**Key Features:**
- WASM-compatible signatures
- JSON interfaces
- MessagePack support
- Browser test payloads
- Efficient data transfer

### 9. Serialization (`io/serialization.py`)

High-speed I/O with Arrow and MessagePack.

**Usage:**
```python
from symbo.io.serialization import SymboSerializer
from symbo.tensor import SymbolicTensor
import sympy as sp

x, y = sp.symbols('x y')

# Serialize expression (JSON is built in; msgpack/Arrow need `pip install 'symbo[io]'`)
expr = x**2 + y
data = SymboSerializer.serialize_expression(expr, fmt='json')
reconstructed = SymboSerializer.deserialize_expression(data, fmt='json')

# Serialize tensor
tensor = SymbolicTensor((2, 2))
tensor.fill_with_symbols("A")
data = SymboSerializer.serialize_tensor(tensor, fmt='json')
reconstructed = SymboSerializer.deserialize_tensor(data, fmt='json')

# Round-trip test
success = SymboSerializer.round_trip_test(
    expr,
    lambda e: SymboSerializer.serialize_expression(e, fmt='json'),
    lambda d: SymboSerializer.deserialize_expression(d, fmt='json'),
)
assert success
```

**Key Features:**
- Arrow format support
- MessagePack support
- Zero data loss
- Round-trip verification
- Mathematical equivalence checking

### 10. Ecosystem Integration (`ecosystem.py`)

Abstract interfaces for integration with FortArch, Topo, Chrono, and Morpho.

**Usage:**
```python
from symbo.ecosystem import (
    EcosystemBridge, MockChrono, MockFortArch, MockMorpho, MockTopo,
)
import sympy as sp

x, y = sp.symbols('x y')

# Any subset of the four providers can be wired in; each bridge method names the
# provider it needs when one is missing.
bridge = EcosystemBridge(
    encryption=MockFortArch(),
    topology=MockTopo(),
    temporal=MockChrono(),
    transformation=MockMorpho(),
)

# Chrono: integrate dx/dt = rhs, one entry per state variable
trajectory = bridge.propagate_forward({x: 1.0, y: 0.0}, {x: y, y: -x},
                                     time_horizon=1.0, dt=0.05)
exponents = MockChrono().compute_lyapunov_exponents({x: y, y: -x}, {x: 0.0, y: 0.0})

# Topo: critical points and homology of the expression's zero set
topo = MockTopo(resolution=81)
print(topo.compute_manifold_topology(x**2 + y**2 - 1, [x, y])['betti_numbers'])
print(topo.find_critical_points(x**3 - 3*x, [x]))

# Morpho: named transformations of an expression
morpho = MockMorpho()
print(morpho.available())                       # 17 transformations
print(morpho.morpho_transform((x + 1)**2, 'expand'))
print(morpho.morpho_transform(x**3, 'diff', {'var': x, 'order': 2}))
print(morpho.generate_variants(sp.sin(x) + sp.cos(y), n_variants=4))

# FortArch: round-trip an expression through the (mock) encrypting provider
print(bridge.secure_compute(x + x, 'simplify'))
```

**Key Features:**
- Abstract interfaces for each model
- Unified bridge class
- Mock implementations for testing
- Future-ready architecture
- Clear separation of concerns

**Notes:**
- Provider arguments are positional-keyword on purpose
  (`chrono_propagate(symbolic_state, dynamics, time_horizon, dt=0.01)`): *state
  first, then the dynamics map*. Swapping them is the easiest way to get a
  nonsense trajectory, so the argument names are part of the contract.
- `EcosystemBridge.__init__` checks each provider with `isinstance` against its
  ABC and raises `TypeError` naming the methods it still owes.
- `MockTopo` is numeric: it samples the expression on a grid, so
  `compute_manifold_topology` reports `method="numeric-grid"` alongside
  `components`, `betti_numbers`, `genus`, `resolution` and `bounds`. Contour
  topology uses a marching-segments crossing graph, which is why nested circles
  correctly give `b1 == 2`.
- `MockMorpho.morpho_transform` takes its parameters as **one dict**
  (`{'var': x, 'order': 2}`); transformations with required parameters raise
  `ValueError` listing what is missing. `TRANSFORMATIONS` has 17 entries.
- `MockMorpho.learn_transformation(source, target)` returns the matching
  registered transformation, falling back to a fitted affine map.

## Package layout, extras and the CLI

`symbo` is a normal installable package (`pyproject.toml`); the engine core lives in
`symbo/nanotensor.py` and each module above is a submodule of the package:

```
symbo/
  __init__.py        re-exported public API (50 names), __version__
  __main__.py        python -m symbo / the `symbo` console script
  nanotensor.py      NanoTensor, deriv_tree, SymbolicTrainer, HybridTrainer,
                     KnowledgeBase, WASM helpers, basis serialisation
  tensor.py primitives.py security.py demos.py _optional.py ecosystem.py
  analytics/{perturbation,explain}.py generative/taylor.py solver/groebner.py
  reasoning/{a_star,hsws}.py io/serialization.py wasm_bindings.py
```

Only `sympy`, `numpy` and `networkx` are required. Everything else is an extra, and
a missing extra produces one actionable `MissingOptionalDependency` (an
`ImportError` subclass) that names the extra to install:

| extra | provides |
|---|---|
| `viz` | `plot_contour`, `plot_surface`, `plot_grid_with_path`, demo plots |
| `io` | msgpack/Arrow persistence, `SymboSerializer`, `save_brain`/`load_brain` |
| `neuro` | `HybridTrainer.make_loader`, `HybridTrainer.torch_fit` |
| `opt` | Gaussian-process order search in `HybridTrainer.symbolic_regression` |
| `kb` | `kanren` relational queries in `KnowledgeBase` (pure-python fallback exists) |
| `dashboard` | `python -m symbo --dashboard` |

```bash
pip install -e ".[dev]"                     # core + pytest/ruff
pip install -e ".[viz,io,opt,kb,dev]"       # everything except torch
pip install -e ".[all]"                     # including torch

python -m symbo --check                     # environment / feature report
python -m symbo --demo rbc|kamke|bench|pipeline
python -m symbo --repl                      # REPL over a fitted tensor
python -m symbo --dashboard                 # needs the dashboard extra
symbo --check                               # same, via the console script
```

`symbo.demos` holds the four entry points above; `--check` is the fastest way to see
which optional features your interpreter can actually use.

## Testing

```bash
python -m pytest                       # the whole suite (testpaths = tests)
python -m pytest -q -m "not slow"       # skip the multi-second symbolic solves
python -m pytest -q -m torch            # only the neuro-extra tests
python -m pytest --doctest-modules symbo -q   # every docstring example
python -m ruff check symbo tests demo_military_grade.py stress_test.py   # lint (line-length 110, E/F/W/B/C4/SIM/RUF)
```

Tests never require an optional dependency: each extra-gated test either uses
`pytest.importorskip` or asserts the *fallback* behaviour, so the suite is green on
a bare `pip install -e ".[dev]"` and greener still with the extras installed.
`tests/conftest.py` exposes the shared RBC model fixture. The CI workflow
(`.ci/ci.yml`, to be moved to `.github/workflows/`) runs lint, a core-only matrix on the supported Python
floor and ceiling, an extras job, an allowed-failure torch job and a wheel build.

## Performance Considerations

- **Symbolic operations**: Use caching for repeated evaluations
- **Tensor operations**: NumPy acceleration for numeric parts
- **WASM transfers**: MessagePack for binary efficiency
- **Serialization**: Arrow format for large datasets

## Best Practices

1. **Type hints**: All functions have complete type annotations
2. **Documentation**: Comprehensive docstrings with examples
3. **Testing**: Unit and integration tests for all modules
4. **Modularity**: Clear separation of concerns
5. **Extensibility**: Abstract interfaces for future additions

## See Also

- Main documentation: `README.md`
- Architecture essay: `docs/Symbo-Architectural-Basis-Essay.md`
- Technical whitepaper: `docs/Symbo-A-Technical-Whitepaper.pdf`
