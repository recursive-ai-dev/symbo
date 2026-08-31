# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Symbo — Nano-scale Hybrid Generative Symbolic Engine
====================================================

A modular symbolic-numeric reasoning system: exact symbolic tensors, generative
Taylor policy functions, Gröbner-based solving, perturbation analysis,
explainability trees, manifold reasoning and serialization — with a
"military-grade" tensor (`NanoTensor`) that self-monitors and learns.

Quick start
-----------
>>> import symbo
>>> import sympy as sp
>>> nt = symbo.NanoTensor((1,), max_order=1, base_vars=['k', 'a'])
>>> nt.generate_taylor({'k': 0.0, 'a': 0.0}, ss_value=sp.S(0))
>>> nt.apply_linear_coeffs({'g_k': 0.9, 'g_a': 0.3, 'g_bias': 0.0})
>>> float(nt.eval_numeric({'k': 1.0, 'a': 1.0})[0])
1.2

Only :mod:`sympy`, :mod:`numpy` and :mod:`networkx` are required. Every heavier
integration (torch, matplotlib, plotly, scikit-optimize, kanren, msgpack,
pyarrow, dill, streamlit) is imported lazily and reported through
:class:`symbo._optional.MissingOptionalDependency` when a feature needs it.
"""

__version__ = "0.2.0"
__author__ = "Damien Davison & Michael Maillet & Sacha Davison"
__organization__ = "Recursive AI Devs"

from ._optional import MissingOptionalDependency, is_available
from .primitives import AtomicPrimitives, add, diff, mul
from .tensor import SymbolicTensor
from .nanotensor import (
    HybridTrainer,
    KnowledgeBase,
    NanoTensor,
    SymbolicTrainer,
    deriv_tree,
    serialize_basis_arrow,
    serialize_basis_msgpack,
    wasm_eval_expression,
    wasm_groebner_solve_json,
)
from .nano_tensor_enhanced import (
    AgencyCore,
    Experience,
    HealthStatus,
    MilitaryGradeNanoTensor,
    OperationType,
    PerformanceMetrics,
)
from .security import safe_sympify
from .generative.taylor import PolicyFunction, TaylorExpansion, generate_multivariate_taylor
from .solver.groebner import (
    StreamingGröbnerSolver,
    handle_infinite_solutions,
    solve_with_groebner,
)
from .analytics.explain import DerivativeTree, derivative_tree
from .analytics.perturbation import (
    PerturbationSolution,
    SecondOrderPerturbation,
    perturbation_solve,
)
from .io.serialization import SymboSerializer
from .reasoning.a_star import SymbolicAStarPathfinder, SymbolicEnergyLandscape
from .reasoning.hsws import (
    Betaconcept,
    Concept,
    DictionarySemanticEngine,
    HSWS,
    SemanticEngine,
    Subconcept,
)
from .ecosystem import EcosystemBridge, MockChrono, MockFortArch
from .wasm_bindings import WASMInterface

__all__ = [
    "HSWS",
    "AgencyCore",
    # primitives & tensors
    "AtomicPrimitives",
    "Betaconcept",
    "Concept",
    "DerivativeTree",
    "DictionarySemanticEngine",
    "EcosystemBridge",
    "Experience",
    "HealthStatus",
    "HybridTrainer",
    "KnowledgeBase",
    # agency / self-monitoring variant
    "MilitaryGradeNanoTensor",
    # optional-dependency helpers
    "MissingOptionalDependency",
    "MockChrono",
    "MockFortArch",
    # core engine
    "NanoTensor",
    "OperationType",
    "PerformanceMetrics",
    "PerturbationSolution",
    "PolicyFunction",
    # analytics
    "SecondOrderPerturbation",
    "SemanticEngine",
    # solving
    "StreamingGröbnerSolver",
    "Subconcept",
    # i/o & reasoning
    "SymboSerializer",
    "SymbolicAStarPathfinder",
    "SymbolicEnergyLandscape",
    "SymbolicTensor",
    "SymbolicTrainer",
    # generative
    "TaylorExpansion",
    "WASMInterface",
    "__author__",
    "__organization__",
    # versioning
    "__version__",
    "add",
    "deriv_tree",
    "derivative_tree",
    "diff",
    "generate_multivariate_taylor",
    "handle_infinite_solutions",
    "is_available",
    "mul",
    "perturbation_solve",
    "safe_sympify",
    "serialize_basis_arrow",
    "serialize_basis_msgpack",
    "solve_with_groebner",
    "wasm_eval_expression",
    "wasm_groebner_solve_json",
]
