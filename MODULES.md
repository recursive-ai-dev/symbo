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

Atomic operations derived from 318 classical algorithms.

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
x = sp.Symbol('x')
dT = T.diff(x)
```

**Key Features:**
- Arbitrary rank (0D to ND)
- Complete tensor algebra
- Symbolic arithmetic
- Differentiation
- Numeric evaluation

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

# Generate expansion
poly = taylor.generate("g")
# Result: g_0 + g_x*x + g_y*y + g_xx*x²/2 + g_xy*x*y + g_yy*y²/2

# Convert to policy function
coeffs = {'g_x': 0.5, 'g_y': 0.3, 'g_xx': -0.1}
policy = taylor.to_policy_function(coeffs)

# Evaluate
value = policy(x=1.0, y=0.5)

# WASM export
json_str = taylor.to_wasm_json()
```

**Key Features:**
- Arbitrary order (1st, 2nd, 3rd, ...)
- Multivariate support
- Symbolic policy functions
- WASM serialization
- Fast compiled evaluation

### 4. Gröbner Basis Solver (`solver/groebner.py`)

Streaming Gröbner basis computation with edge case handling.

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

**Usage:**
```python
from symbo.analytics.perturbation import SecondOrderPerturbation, perturbation_solve
import sympy as sp

# Define system
k, c, a = sp.symbols('k c a')  # state, control, shock
alpha, beta, delta = sp.symbols('alpha beta delta')

# Euler equation for RBC model
euler = c**(-1) - beta * c_next**(-1) * (alpha * a * k**(alpha-1) + 1 - delta)

# Solve
solution = perturbation_solve(
    equations=[euler],
    state_vars=[k],
    control_vars=[c],
    shock_vars=[a],
    parameters={alpha: 0.3, beta: 0.96, delta: 0.1},
    order=2
)

print(f"Steady state: {solution.steady_state}")
print(f"First-order coefficients: {solution.first_order}")
print(f"Second-order coefficients: {solution.second_order}")
```

**Key Features:**
- First and second-order approximations
- Steady-state computation
- Risk/variance corrections
- Policy function generation
- Coefficient solving

### 6. A* Pathfinding (`reasoning/a_star.py`)

Symbolic state-based pathfinding on energy landscapes.

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

# Get influence ranking
ranking = tree.get_influence_ranking()
for var, score in ranking:
    print(f"{var}: {score}")

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

### 8. WASM Bindings (`wasm_bindings.py`)

Browser-compatible interfaces for symbolic computation.

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

# Serialize expression
expr = x**2 + y
data = SymboSerializer.serialize_expression(expr, format='msgpack')
reconstructed = SymboSerializer.deserialize_expression(data, format='msgpack')

# Serialize tensor
tensor = SymbolicTensor((2, 2))
tensor.fill_with_symbols("A")
data = SymboSerializer.serialize_tensor(tensor, format='arrow')
reconstructed = SymboSerializer.deserialize_tensor(data, format='arrow')

# Round-trip test
success = SymboSerializer.round_trip_test(
    expr,
    SymboSerializer.serialize_expression,
    SymboSerializer.deserialize_expression
)
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
from symbo.ecosystem import EcosystemBridge, MockChrono
import sympy as sp

# Create bridge with components
bridge = EcosystemBridge(
    temporal=MockChrono()
)

# Use integrated functionality
x, y = sp.symbols('x y')
dynamics = {x: y, y: -x}  # Simple oscillator
state = {x: 1.0, y: 0.0}

trajectory = bridge.propagate_forward(state, dynamics, time_horizon=10.0)
```

**Key Features:**
- Abstract interfaces for each model
- Unified bridge class
- Mock implementations for testing
- Future-ready architecture
- Clear separation of concerns

## Testing

Run tests:
```bash
# Unit tests
python -m unittest tests.test_primitives -v

# Integration test
python -c "
from symbo import *
print('All imports successful')
"
```

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
