# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Ecosystem Integration Interfaces
=================================

This module defines abstract interface classes for integrating Symbo
with other Recursive AI Devs models: FortArch, Topo, Chrono, and Morpho.

These interfaces ensure architectural readiness for future integration
while maintaining clear separation of concerns.

Models:
- FortArch: Encrypted container for secure symbolic computation
- Topo: Topological reasoning over symbolic manifolds
- Chrono: Temporal propagation of symbolic states
- Morpho: Transformational generative engine
"""

import itertools
import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import sympy as sp

from symbo.security import safe_sympify

logger = logging.getLogger(__name__)


class EncryptionProvider(ABC):
    """
    Abstract interface for encrypted symbolic computation.

    Defines the contract for FortArch integration, enabling
    secure computation on encrypted symbolic expressions.
    """

    @abstractmethod
    def encrypt_expression(self, expr: sp.Expr) -> bytes:
        """
        Encrypt a symbolic expression.

        Parameters
        ----------
        expr : sp.Expr
            Expression to encrypt

        Returns
        -------
        bytes
            Encrypted data
        """
        pass

    @abstractmethod
    def decrypt_expression(self, encrypted: bytes) -> sp.Expr:
        """
        Decrypt to recover symbolic expression.

        Parameters
        ----------
        encrypted : bytes
            Encrypted data

        Returns
        -------
        sp.Expr
            Decrypted expression
        """
        pass

    @abstractmethod
    def homomorphic_eval(self, encrypted: bytes, operation: str) -> bytes:
        """
        Perform homomorphic operation on encrypted data.

        Parameters
        ----------
        encrypted : bytes
            Encrypted expression
        operation : str
            Operation to perform ('add', 'mul', 'diff', etc.)

        Returns
        -------
        bytes
            Result in encrypted form
        """
        pass


class TopologicalReasoner(ABC):
    """
    Abstract interface for topological reasoning.

    Defines the contract for Topo integration, enabling
    topological analysis of symbolic manifolds.
    """

    @abstractmethod
    def compute_manifold_topology(self,
                                  expression: sp.Expr,
                                  variables: List[sp.Symbol]) -> Dict[str, Any]:
        """
        Compute topological properties of manifold defined by expression.

        Parameters
        ----------
        expression : sp.Expr
            Expression defining manifold (e.g., level set)
        variables : List[sp.Symbol]
            Manifold coordinates

        Returns
        -------
        Dict[str, Any]
            Topological properties (genus, connectivity, etc.)
        """
        pass

    @abstractmethod
    def find_critical_points(self,
                            energy_function: sp.Expr,
                            variables: List[sp.Symbol]) -> List[Dict[sp.Symbol, float]]:
        """
        Find critical points of energy function.

        Parameters
        ----------
        energy_function : sp.Expr
            Energy/potential function
        variables : List[sp.Symbol]
            Variables

        Returns
        -------
        List[Dict[sp.Symbol, float]]
            Critical points (minima, maxima, saddles)
        """
        pass

    @abstractmethod
    def compute_homology(self,
                        simplicial_complex: Any) -> Dict[int, int]:
        """
        Compute homology groups.

        Parameters
        ----------
        simplicial_complex : Any
            Simplicial complex representation

        Returns
        -------
        Dict[int, int]
            Betti numbers for each dimension
        """
        pass


class TemporalPropagator(ABC):
    """
    Abstract interface for temporal propagation.

    Defines the contract for Chrono integration, enabling
    time-evolution of symbolic states.
    """

    @abstractmethod
    def chrono_propagate(self,
                        symbolic_state: Dict[sp.Symbol, sp.Expr],
                        dynamics: Dict[sp.Symbol, sp.Expr],
                        time_horizon: float,
                        dt: float) -> List[Dict[sp.Symbol, float]]:
        """
        Propagate symbolic state forward in time.

        Parameters
        ----------
        symbolic_state : Dict[sp.Symbol, sp.Expr]
            Initial state as symbolic expressions
        dynamics : Dict[sp.Symbol, sp.Expr]
            Time derivatives dx/dt = f(x)
        time_horizon : float
            Total time to propagate
        dt : float
            Time step

        Returns
        -------
        List[Dict[sp.Symbol, float]]
            Trajectory as list of states
        """
        pass

    @abstractmethod
    def compute_lyapunov_exponents(self,
                                   dynamics: Dict[sp.Symbol, sp.Expr],
                                   steady_state: Dict[sp.Symbol, float]) -> List[float]:
        """
        Compute Lyapunov exponents for stability analysis.

        Parameters
        ----------
        dynamics : Dict[sp.Symbol, sp.Expr]
            Dynamical system
        steady_state : Dict[sp.Symbol, float]
            Equilibrium point

        Returns
        -------
        List[float]
            Lyapunov exponents
        """
        pass

    @abstractmethod
    def forecast_trajectory(self,
                           historical_states: List[Dict[sp.Symbol, float]],
                           steps_ahead: int) -> List[Dict[sp.Symbol, float]]:
        """
        Forecast future trajectory from historical data.

        Parameters
        ----------
        historical_states : List[Dict[sp.Symbol, float]]
            Past states
        steps_ahead : int
            Number of steps to forecast

        Returns
        -------
        List[Dict[sp.Symbol, float]]
            Forecasted states
        """
        pass


class TransformationEngine(ABC):
    """
    Abstract interface for transformational generation.

    Defines the contract for Morpho integration, enabling
    symbolic transformations and generative operations.
    """

    @abstractmethod
    def morpho_transform(self,
                        source_expr: sp.Expr,
                        transformation_type: str,
                        parameters: Optional[Dict[str, Any]] = None) -> sp.Expr:
        """
        Apply symbolic transformation.

        Parameters
        ----------
        source_expr : sp.Expr
            Source expression
        transformation_type : str
            Type of transformation ('simplify', 'expand', 'factor', etc.)
        parameters : Dict[str, Any], optional
            Transformation parameters

        Returns
        -------
        sp.Expr
            Transformed expression
        """
        pass

    @abstractmethod
    def generate_variants(self,
                         template_expr: sp.Expr,
                         n_variants: int,
                         constraints: Optional[List[sp.Expr]] = None) -> List[sp.Expr]:
        """
        Generate variants of template expression.

        Parameters
        ----------
        template_expr : sp.Expr
            Template expression
        n_variants : int
            Number of variants to generate
        constraints : List[sp.Expr], optional
            Constraints on variants

        Returns
        -------
        List[sp.Expr]
            Generated variants
        """
        pass

    @abstractmethod
    def learn_transformation(self,
                            source_exprs: List[sp.Expr],
                            target_exprs: List[sp.Expr]) -> callable:
        """
        Learn transformation from examples.

        Parameters
        ----------
        source_exprs : List[sp.Expr]
            Source expressions
        target_exprs : List[sp.Expr]
            Target expressions

        Returns
        -------
        callable
            Learned transformation function
        """
        pass


class EcosystemBridge:
    """
    Bridge class for coordinating multiple ecosystem components.

    This class provides a unified interface for using multiple
    ecosystem models together with Symbo.

    Parameters
    ----------
    encryption : EncryptionProvider, optional
        FortArch encryption provider
    topology : TopologicalReasoner, optional
        Topo topological reasoner
    temporal : TemporalPropagator, optional
        Chrono temporal propagator
    transformation : TransformationEngine, optional
        Morpho transformation engine

    Raises
    ------
    TypeError
        If a provider is given but does not implement the matching interface. The
        check happens here rather than at call time so a wiring mistake names the
        provider and the methods it still owes.

    Examples
    --------
    >>> import sympy as sp
    >>> from symbo.ecosystem import EcosystemBridge, MockChrono
    >>> bridge = EcosystemBridge(temporal=MockChrono())
    >>> x, y = sp.symbols('x y')
    >>> trajectory = bridge.propagate_forward({x: 1.0, y: 0.0}, {x: y, y: -x}, 0.02)
    >>> len(trajectory)
    3
    """

    #: provider attribute -> the interface it has to satisfy
    PROVIDER_INTERFACES: ClassVar[Dict[str, type]] = {
        "encryption": EncryptionProvider,
        "topology": TopologicalReasoner,
        "temporal": TemporalPropagator,
        "transformation": TransformationEngine,
    }

    def __init__(self,
                 encryption: Optional[EncryptionProvider] = None,
                 topology: Optional[TopologicalReasoner] = None,
                 temporal: Optional[TemporalPropagator] = None,
                 transformation: Optional[TransformationEngine] = None):
        """Initialize ecosystem bridge."""
        given = {"encryption": encryption, "topology": topology,
                  "temporal": temporal, "transformation": transformation}
        for name, interface in self.PROVIDER_INTERFACES.items():
            provider = given[name]
            if provider is not None and not isinstance(provider, interface):
                missing = sorted(interface.__abstractmethods__ - set(dir(provider)))
                raise TypeError(
                    f"the {name!r} provider must implement "
                    f"{interface.__name__}"
                    + (f"; missing {', '.join(missing)}" if missing else "")
                )
            setattr(self, name, provider)

    #: reference implementation shipped with Symbo, for the error messages
    PROVIDER_MOCKS: ClassVar[Dict[str, str]] = {
        "EncryptionProvider": "MockFortArch",
        "TopologicalReasoner": "MockTopo",
        "TemporalPropagator": "MockChrono",
        "TransformationEngine": "MockMorpho",
    }

    def _require(self, name: str, method: str) -> Any:
        """Return the provider for *name*, or explain which one *method* needs."""
        provider = getattr(self, name)
        if provider is None:
            interface = self.PROVIDER_INTERFACES[name].__name__
            raise NotImplementedError(
                f"{method}() needs a {interface} provider: construct EcosystemBridge "
                f"with {name}=... (implement {interface}, or use "
                f"{self.PROVIDER_MOCKS[interface]} from symbo.ecosystem)"
            )
        return provider

    def secure_compute(self,
                      expr: sp.Expr,
                      operation: str) -> sp.Expr:
        """
        Perform secure computation on expression.

        Uses FortArch encryption if available.
        """
        provider = self._require('encryption', 'secure_compute')

        # Encrypt, compute on the encrypted form, then decrypt
        encrypted = provider.encrypt_expression(expr)
        result_encrypted = provider.homomorphic_eval(encrypted, operation)
        return provider.decrypt_expression(result_encrypted)

    def analyze_topology(self,
                        expr: sp.Expr,
                        variables: List[sp.Symbol]) -> Dict[str, Any]:
        """
        Perform topological analysis.

        Uses Topo if available.
        """
        provider = self._require('topology', 'analyze_topology')

        return provider.compute_manifold_topology(expr, variables)

    def propagate_forward(self,
                         state: Dict[sp.Symbol, sp.Expr],
                         dynamics: Dict[sp.Symbol, sp.Expr],
                         time_horizon: float,
                         dt: float = 0.01) -> List[Dict[sp.Symbol, float]]:
        """
        Propagate state forward in time.

        Uses Chrono if available.

        Parameters
        ----------
        state : dict
            ``{symbol: initial_value}`` for the symbolic state.
        dynamics : dict
            ``{symbol: rhs_expression}``, i.e. ``dx/dt = rhs``.
        time_horizon : float
            How long to integrate.
        dt : float, optional
            Integration step (default ``0.01``).
        """
        provider = self._require('temporal', 'propagate_forward')

        return provider.chrono_propagate(state, dynamics, time_horizon, dt)

    def transform_expression(self,
                           expr: sp.Expr,
                           transformation: str) -> sp.Expr:
        """
        Apply symbolic transformation.

        Uses Morpho if available.
        """
        provider = self._require('transformation', 'transform_expression')

        return provider.morpho_transform(expr, transformation)


# Placeholder implementations for testing

class MockFortArch(EncryptionProvider):
    """Mock FortArch implementation for testing."""

    def encrypt_expression(self, expr: sp.Expr) -> bytes:
        return str(expr).encode('utf-8')

    def decrypt_expression(self, encrypted: bytes) -> sp.Expr:
        return safe_sympify(encrypted.decode('utf-8'))

    def homomorphic_eval(self, encrypted: bytes, operation: str) -> bytes:
        expr = self.decrypt_expression(encrypted)
        if operation == 'simplify':
            result = sp.simplify(expr)
        else:
            result = expr
        return self.encrypt_expression(result)


class MockChrono(TemporalPropagator):
    """
    Reference ``Chrono`` implementation used for tests and demos.

    The dynamics are interpreted as an autonomous ODE system

        d x_i / dt = f_i(x)

    and integrated with explicit Euler steps of size ``dt``. This is a mock -- it
    is deliberately simple and transparent rather than numerically sophisticated
    -- but it is a *real* implementation: every documented behaviour (trajectory
    propagation, stability exponents, forecasting) is computed from the input
    instead of being stubbed out.
    """

    @staticmethod
    def _as_number(value: Any) -> float:
        """Coerce a numeric or symbolic-but-evaluable value to float."""
        if isinstance(value, sp.Basic):
            if value.free_symbols:
                raise ValueError(
                    f"state value {value} still contains free symbols "
                    f"{sorted(str(s) for s in value.free_symbols)}"
                )
            return float(value.evalf())
        return float(value)

    def chrono_propagate(self,
                        symbolic_state: Dict[sp.Symbol, sp.Expr],
                        dynamics: Dict[sp.Symbol, sp.Expr],
                        time_horizon: float,
                        dt: float = 0.01) -> List[Dict[sp.Symbol, float]]:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if time_horizon < 0:
            raise ValueError(f"time_horizon must be >= 0, got {time_horizon}")
        missing = [str(k) for k, rhs in dynamics.items()
                   if getattr(rhs, "free_symbols", set()) - set(symbolic_state)]
        if missing:
            raise ValueError(
                "dynamics reference symbols that are not part of the state: "
                f"{missing}; add them to the state or substitute their values"
            )

        state = {k: self._as_number(v) for k, v in symbolic_state.items()}
        trajectory = [dict(state)]

        n_steps = round(time_horizon / dt)
        for _ in range(n_steps):
            derivatives = {
                var: self._as_number(rhs.subs(state)) if isinstance(rhs, sp.Basic)
                else self._as_number(rhs)
                for var, rhs in dynamics.items()
            }
            for var, deriv in derivatives.items():
                state[var] = state[var] + dt * deriv
            trajectory.append(dict(state))

        return trajectory

    def compute_lyapunov_exponents(self,
                                   dynamics: Dict[sp.Symbol, sp.Expr],
                                   steady_state: Dict[sp.Symbol, float]) -> List[float]:
        """
        Growth rates of the linearized system at ``steady_state``.

        For the linearization dx/dt = J x the exponents are the real parts of
        the eigenvalues of the Jacobian J, sorted from most to least unstable.
        """
        variables = list(dynamics.keys())
        if not variables:
            return []
        n = len(variables)
        jacobian = sp.Matrix(
            [[sp.diff(dynamics[a], b) for b in variables] for a in variables]
        )
        subs = {v: self._as_number(steady_state.get(v, 0.0)) for v in variables}
        numeric = np.array([[float(entry) for entry in row]
                            for row in jacobian.subs(subs).tolist()], dtype=float)
        if numeric.shape != (n, n):
            raise ValueError(
                f"the linearization is {numeric.shape}, expected ({n}, {n}); every "
                "dynamics key must be a state variable"
            )
        exponents = sorted((float(r) for r in np.linalg.eigvals(numeric).real), reverse=True)
        return exponents

    def forecast_trajectory(self,
                           historical_states: List[Dict[sp.Symbol, float]],
                           steps_ahead: int) -> List[Dict[sp.Symbol, float]]:
        """
        Extrapolate linearly from the most recent observed step.

        With a single historical state the forecast is flat (no trend to use).
        """
        if steps_ahead < 0:
            raise ValueError(f"steps_ahead must be >= 0, got {steps_ahead}")
        if not historical_states:
            raise ValueError("forecast_trajectory() needs at least one historical state")
        if steps_ahead == 0:
            return []

        last = historical_states[-1]
        if len(historical_states) >= 2:
            prev = historical_states[-2]
            delta = {k: last[k] - prev.get(k, last[k]) for k in last}
        else:
            delta = dict.fromkeys(last, 0.0)

        forecast = []
        current = dict(last)
        for _ in range(steps_ahead):
            current = {k: current[k] + delta[k] for k in current}
            forecast.append(dict(current))
        return forecast


class MockTopo(TopologicalReasoner):
    """
    Self-contained :class:`TopologicalReasoner` for tests and offline analysis.

    Real Topo integration is optional, so this mock derives its answers from the
    symbolic expressions themselves:

    * ``find_critical_points`` solves the symbolic gradient with
      :func:`sympy.solve` and keeps the real, finite solutions;
    * ``compute_manifold_topology`` samples the level set ``f = 0`` on a grid and
      analyses the resulting contour *as a graph*, which yields the number of
      connected components and the first Betti number (independent cycles) --
      i.e. how many closed loops the curve has;
    * ``compute_homology`` returns the Betti numbers of a graph (a 1-dimensional
      simplicial complex): ``b0`` from the connected components, ``b1`` from the
      cycle rank ``E - V + C``.

    Both grid-based answers are approximations that improve with ``resolution``;
    they are reported together with the resolution used so a caller can tell how
    much to trust them.
    """

    def __init__(self, resolution: int = 121, bounds: Tuple[float, float] = (-3.0, 3.0)):
        if resolution < 5:
            raise ValueError(f"resolution must be >= 5, got {resolution}")
        if bounds[1] <= bounds[0]:
            raise ValueError(f"bounds must be increasing, got {bounds}")
        self.resolution = int(resolution)
        self.bounds = (float(bounds[0]), float(bounds[1]))

    # --- critical points -----------------------------------------------------

    def find_critical_points(self,
                            energy_function: sp.Expr,
                            variables: List[sp.Symbol]) -> List[Dict[sp.Symbol, float]]:
        """
        Real solutions of ``grad(energy) = 0``.

        Returns one dict per stationary point, plus ``classification``
        (``minimum`` / ``maximum`` / ``saddle`` / ``degenerate``) taken from the
        eigenvalues of the Hessian at that point.
        """
        variables = list(variables)
        if not variables:
            raise ValueError("find_critical_points() needs at least one variable")

        energy = sp.sympify(energy_function)
        gradient = [sp.diff(energy, v) for v in variables]
        try:
            raw = sp.solve(gradient, variables, dict=True)
        except Exception as exc:  # sympy raises freely on unsolvable systems
            logger.warning("find_critical_points(): solve() failed: %s", exc)
            return []

        points: List[Dict[sp.Symbol, float]] = []
        seen = set()
        hessian = sp.Matrix([[sp.diff(energy, a, b) for b in variables] for a in variables])
        for sol in raw:
            try:
                values = {}
                for v in variables:
                    val = complex(sp.N(sol.get(v, 0.0)))
                    if abs(val.imag) > 1e-9 or not math.isfinite(val.real):
                        raise ValueError("non-real")
                    values[v] = float(val.real)
            except (TypeError, ValueError, AttributeError):
                continue  # parametric or symbolic branch: not an isolated point
            key = tuple(round(values[v], 9) for v in variables)
            if key in seen:
                continue
            seen.add(key)
            entry: Dict[str, Any] = dict(values)
            entry["classification"] = self._classify(hessian, variables, values)
            points.append(entry)
        return points

    @staticmethod
    def _classify(hessian: "sp.Matrix", variables: List[sp.Symbol],
                  values: Dict[sp.Symbol, float]) -> str:
        """Second-derivative test on the Hessian at a stationary point."""
        subs = {v: values[v] for v in variables}
        try:
            eigs = [complex(float(e)) for e in hessian.subs(subs).eigenvals()]
        except (TypeError, ValueError):
            return "degenerate"
        real_parts = [e.real for e in eigs]
        if all(r > 1e-9 for r in real_parts):
            return "minimum"
        if all(r < -1e-9 for r in real_parts):
            return "maximum"
        if all(abs(r) <= 1e-9 for r in real_parts):
            return "degenerate"
        return "saddle"

    # --- level-set topology ---------------------------------------------------

    def compute_manifold_topology(self,
                                  expression: sp.Expr,
                                  variables: List[sp.Symbol]) -> Dict[str, Any]:
        """
        Grid-based topology of the level set ``expression = 0``.

        Only univariate and bivariate level sets are supported; anything else
        raises ``ValueError`` rather than silently guessing. The returned dict
        reports ``components``, ``betti_numbers`` and ``genus`` (the number of
        independent loops of the curve, which for a planar level set *is* its
        genus in the sense of "how many holes it encloses").
        """
        variables = list(variables)
        if len(variables) not in (1, 2):
            raise ValueError(
                "MockTopo.compute_manifold_topology() supports 1D and 2D level "
                f"sets; got {len(variables)} variables"
            )
        expr = sp.sympify(expression)
        f = sp.lambdify(variables, expr, modules="numpy")

        axis = np.linspace(self.bounds[0], self.bounds[1], self.resolution)
        if len(variables) == 1:
            values = np.asarray(f(axis), dtype=float)
            signs = np.sign(values)
            roots = int(np.sum((signs[:-1] * signs[1:]) < 0)) + int(np.sum(signs == 0))
            return {
                "dimension": 1,
                "components": roots,
                "betti_numbers": {0: roots, 1: 0},
                "genus": 0,
                "resolution": self.resolution,
                "bounds": self.bounds,
                "method": "sign changes of f on a uniform grid",
            }

        gx, gy = np.meshgrid(axis, axis, indexing="ij")
        field = np.asarray(f(gx, gy), dtype=float)
        field = np.nan_to_num(field, nan=0.0, posinf=1e300, neginf=-1e300)
        nodes, edges = self._crossing_graph(field)
        graph = self._nx_graph(nodes, edges)
        components = graph.number_of_nodes() and len(list(nx.connected_components(graph)))
        # the crossing graph is 2-regular wherever the curve closes on itself, so
        # its cycle rank E - V + C counts exactly the closed loops of the level set
        cycle_rank = graph.number_of_edges() - graph.number_of_nodes() + components
        return {
            "dimension": 2,
            "components": components,
            "betti_numbers": {0: components, 1: int(max(cycle_rank, 0))},
            "genus": int(max(cycle_rank, 0)),
            "resolution": self.resolution,
            "bounds": self.bounds,
            "method": "sign-change contour of f(x, y) = 0 traced as a graph",
        }

    @staticmethod
    def _crossing_graph(field: np.ndarray):
        """
        Trace the zero level set of ``field`` as a graph of curve segments.

        A node is one sign change along a grid segment (so the two cells sharing
        that segment see the *same* node, which is what stitches the local
        segments into global curves); an edge joins the two crossings of one
        cell. Cells with four crossings -- the saddle case of marching squares --
        are split by the sign of the cell centre. The result is 2-regular on
        closed loops and 1-regular at the domain boundary, so the graph's cycle
        rank is the number of loops and its component count the number of curve
        branches.
        """
        sign = np.where(field >= 0, 1, -1)
        # horizontal / vertical sign changes between neighbouring grid vertices
        h_cross = (sign[:-1, :-1] * sign[:-1, 1:]) < 0   # along j, at row i
        v_cross = (sign[:-1, :-1] * sign[1:, :-1]) < 0   # along i, at column j

        nodes = ({("h", i, j) for i, j in zip(*np.nonzero(h_cross), strict=True)}
                 | {("v", i, j) for i, j in zip(*np.nonzero(v_cross), strict=True)})

        edges: List[Tuple[Any, Any]] = []
        hi, hj = np.nonzero(h_cross)
        vi, vj = np.nonzero(v_cross)
        # Crossings are attributed to the cell(s) they bound, in (row, col) grid
        # terms: a horizontal crossing at (i, j) is the top side of cell (i - 1, j)
        # and the bottom side of cell (i, j); a vertical one is the right side of
        # cell (i, j - 1) and the left side of cell (i, j).
        bottom = {(int(i), int(j)): ("h", int(i), int(j))
                  for i, j in zip(hi.tolist(), hj.tolist(), strict=True)}
        left = {(int(i), int(j)): ("v", int(i), int(j))
                 for i, j in zip(vi.tolist(), vj.tolist(), strict=True)}
        crossings_by_cell: Dict[Tuple[int, int], List[Any]] = {}
        for (i, j), node in bottom.items():
            crossings_by_cell.setdefault((i, j), []).append(("b", node))
            crossings_by_cell.setdefault((i, j), [])  # cell above shares its top
            crossings_by_cell.setdefault((i - 1, j), []).append(("t", node))
        for (i, j), node in left.items():
            crossings_by_cell.setdefault((i, j), []).append(("l", node))
            crossings_by_cell.setdefault((i, j - 1), []).append(("r", node))

        for (i, j), found in crossings_by_cell.items():
            sides = [node for _, node in found]
            if len(sides) == 2:
                edges.append((sides[0], sides[1]))
            elif len(sides) == 4:
                centre = float(np.mean([field[i, j], field[i + 1, j],
                                         field[i, j + 1], field[i + 1, j + 1]]))
                side = dict(found)
                bottom_n, right_n, top_n, left_n = side["b"], side["r"], side["t"], side["l"]
                if centre >= 0:
                    edges.extend([(bottom_n, right_n), (top_n, left_n)])
                else:
                    edges.extend([(right_n, top_n), (left_n, bottom_n)])
        return sorted(nodes, key=str), edges

    @staticmethod
    def _nx_graph(nodes, edges) -> "nx.Graph":
        graph = nx.Graph()
        graph.add_nodes_from(tuple(n) if isinstance(n, list) else n for n in nodes)
        graph.add_edges_from(tuple(e) for e in edges)
        return graph

    def compute_homology(self, simplicial_complex: Any) -> Dict[int, int]:
        """
        Betti numbers ``{0: b0, 1: b1}`` of a graph-like simplicial complex.

        Accepts a ``networkx.Graph``, a ``(nodes, edges)`` pair or a list of
        simplices (tuples of vertex ids); higher-dimensional simplices are
        reduced to their 1-skeleton, which is what determines ``b0`` and ``b1``.
        """
        if isinstance(simplicial_complex, tuple) and len(simplicial_complex) == 2:
            nodes, edges = simplicial_complex
            graph = self._nx_graph(list(nodes), [tuple(e) for e in edges])
        elif isinstance(simplicial_complex, (list, tuple, set)):
            nodes, edges = [], []
            for simplex in simplicial_complex:
                simplex = list(simplex)
                nodes.extend(simplex)
                for a in range(len(simplex)):
                    for b in range(a + 1, len(simplex)):
                        edges.append((simplex[a], simplex[b]))
            graph = self._nx_graph(nodes, edges)
        else:
            graph = simplicial_complex

        n_vertices = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        if n_vertices == 0:
            return {0: 0, 1: 0}
        components = len(list(nx.connected_components(graph)))
        return {0: components, 1: int(max(n_edges - n_vertices + components, 0))}


class MockMorpho(TransformationEngine):
    """
    Self-contained :class:`TransformationEngine` driven by a SymPy operation set.

    ``morpho_transform`` dispatches to named SymPy operations (``TRANSFORMATIONS``
    is the authoritative list); ``generate_variants`` returns structurally
    different applications of that set, deterministically ordered so the same
    template always yields the same variants; ``learn_transformation`` searches
    the same set for an operation reproducing a set of example pairs and falls
    back to an affine relation ``a * source + b`` when no single operation fits.
    """

    #: name -> callable applying the transformation
    TRANSFORMATIONS: ClassVar[Dict[str, Callable[..., sp.Expr]]] = {
        "simplify": lambda e, **_: sp.simplify(e),
        "expand": lambda e, **_: sp.expand(e),
        "factor": lambda e, **_: sp.factor(e),
        "trigsimp": lambda e, **_: sp.trigsimp(e),
        "powsimp": lambda e, **_: sp.powsimp(e),
        "logcombine": lambda e, **_: sp.logcombine(e, force=True),
        "expand_log": lambda e, **_: sp.expand_log(e, force=True),
        "expand_power_base": lambda e, **_: sp.expand_power_base(e, force=True),
        "cancel": lambda e, **_: sp.cancel(e),
        "apart": lambda e, **_: sp.apart(e) if e.atoms(sp.Add) else sp.together(e),
        "together": lambda e, **_: sp.together(e),
        "nsimplify": lambda e, rational=False, **_: sp.nsimplify(
            e, rational=bool(rational)),
        "collect": lambda e, vars=None, **_: sp.collect(
            e, [sp.Symbol(v) if not isinstance(v, sp.Basic) else v
                for v in (vars or sorted(e.free_symbols, key=str) or [sp.Symbol("x")])]),
        "rewrite": lambda e, form="exp", **_: e.rewrite(str(form)),
        "diff": lambda e, var=None, order=1, **_: sp.diff(
            e, sp.Symbol(var) if isinstance(var, str) else var, order),
        "subs": lambda e, mapping=None, **_: e.subs(
            {sp.Symbol(k) if isinstance(k, str) else k: v
             for k, v in (mapping or {}).items()}),
        "series": lambda e, var=None, point=0.0, n=4, **_: sp.series(
            e, sp.Symbol(var) if isinstance(var, str) else var, point, n).removeO(),
    }

    #: Transformations that are meaningless without one specific parameter;
    #: checked before dispatch so the caller gets a clear message instead of a
    #: SymPy-internal ``AttributeError``.
    REQUIRED_PARAMETERS: ClassVar[Dict[str, Tuple[str, ...]]] = {
        "diff": ("var",),
        "series": ("var",),
        "subs": ("mapping",),
    }

    @classmethod
    def available(cls) -> List[str]:
        """Names accepted by :meth:`morpho_transform`."""
        return sorted(cls.TRANSFORMATIONS)

    def morpho_transform(self,
                        source_expr: sp.Expr,
                        transformation_type: str,
                        parameters: Optional[Dict[str, Any]] = None) -> sp.Expr:
        """Apply one named SymPy transformation to ``source_expr``."""
        op = self.TRANSFORMATIONS.get(str(transformation_type))
        if op is None:
            raise ValueError(
                f"unknown transformation {transformation_type!r}; "
                f"available: {self.available()}"
            )
        params = dict(parameters or {})
        for required in self.REQUIRED_PARAMETERS.get(str(transformation_type), ()):
            if params.get(required) is None:
                raise ValueError(
                    f"transformation {transformation_type!r} requires the "
                    f"parameter {required!r}"
                )
        expr = sp.sympify(source_expr)
        try:
            return op(expr, **params)
        except TypeError as exc:
            raise ValueError(
                f"transformation {transformation_type!r} rejected parameters "
                f"{sorted(params)}: {exc}"
            ) from exc

    def generate_variants(self,
                         template_expr: sp.Expr,
                         n_variants: int,
                         constraints: Optional[List[sp.Expr]] = None) -> List[sp.Expr]:
        """
        Produce up to ``n_variants`` structurally distinct rewrites of a template.

        Variants are the transformations that actually change the expression, in
        :meth:`available` order (so the result is deterministic). ``constraints``
        are predicates over the same symbols: a variant is dropped unless every
        constraint is nonzero at each of a fixed set of sample points -- a
        syntactic check would reject semantically different but equal expressions,
        numeric sampling is what a variant *means* here.
        """
        if n_variants < 1:
            raise ValueError(f"n_variants must be >= 1, got {n_variants}")
        template = sp.sympify(template_expr)
        constraints = [sp.sympify(c) for c in (constraints or [])]

        variants: List[sp.Expr] = []
        for name in self.available():
            if len(variants) >= n_variants:
                break
            try:
                candidate = self.morpho_transform(template, name)
            except Exception:  # a rewrite may be undefined for this expression
                continue
            if candidate == template or candidate in variants:
                continue
            if self._satisfies_constraints(candidate, constraints):
                variants.append(candidate)
        return variants

    #: Deterministic sample points used to evaluate constraint predicates.
    _CONSTRAINT_SAMPLES = (0.7, 1.35, 2.1)

    @classmethod
    def _satisfies_constraints(cls, expr: sp.Expr, constraints: List[sp.Expr]) -> bool:
        """
        Check that every constraint holds for ``expr`` on a small fixed point set.

        SymPy keeps ``x > 0`` as a ``Relational`` rather than a boolean, so a
        constraint is judged by substituting numbers: a relational must decide to
        ``True``, an ordinary expression is satisfied when it is nonzero. Anything
        that cannot be decided numerically rejects the variant -- failing closed
        is right for a filter whose purpose is to exclude candidates.
        """
        if not constraints:
            return True
        symbols = sorted(set(expr.free_symbols)
                         | {v for c in constraints for v in c.free_symbols}, key=str)
        values = cls._CONSTRAINT_SAMPLES
        # the full Cartesian product only while it stays small; otherwise vary one
        # symbol at a time, which still catches the common sign/range guards
        if len(symbols) <= 2:
            assignments = list(itertools.product(values, repeat=len(symbols)))
        else:
            assignments = [tuple(values[i] if i == j else values[0]
                                 for i in range(len(symbols)))
                           for j in range(len(symbols))]

        for constraint in constraints:
            for assignment in assignments:
                subs = dict(zip(symbols, assignment, strict=True))
                decided = constraint.subs(subs)
                if decided.is_Relational or decided.is_Boolean:
                    try:
                        if not bool(decided):
                            return False
                    except TypeError:
                        return False
                else:
                    try:
                        if abs(float(decided)) <= 1e-12:
                            return False
                    except (TypeError, ValueError):
                        return False
        return True

    def learn_transformation(self,
                            source_exprs: List[sp.Expr],
                            target_exprs: List[sp.Expr]) -> Callable[[sp.Expr], sp.Expr]:
        """
        Find a transformation reproducing every ``source -> target`` pair.

        Returns the matching callable, carrying a ``description`` attribute
        naming the operation that was found. Raises ``ValueError`` when no
        operation in :data:`TRANSFORMATIONS` and no affine relation fits, which
        is the honest answer for a sample set that is not a transformation.
        """
        sources = [sp.sympify(e) for e in source_exprs]
        targets = [sp.sympify(e) for e in target_exprs]
        if not sources or len(sources) != len(targets):
            raise ValueError(
                "learn_transformation() needs equally many source and target "
                f"expressions, got {len(sources)} and {len(targets)}"
            )

        for name in self.available():
            if name in {"diff", "series", "subs", "collect", "rewrite", "nsimplify",
                        "apart"}:  # parameterised: cannot be guessed from pairs alone
                continue
            op = self.TRANSFORMATIONS[name]
            try:
                if all(op(src) == tgt for src, tgt in zip(sources, targets, strict=True)):
                    def transform(expr, _op=op, _name=name):
                        return _op(sp.sympify(expr))
                    transform.description = f"{name} (learned from {len(sources)} pair(s))"
                    return transform
            except Exception:
                continue

        # fall back to an affine relation target = a * source + b
        try:
            a, b = sp.symbols("_a _b")
            eqs = [a * src + b - tgt for src, tgt in zip(sources, targets, strict=True)]
            solution = sp.solve(eqs, [a, b], dict=True)
            if solution:
                mapping = solution[0]
                a_val, b_val = mapping.get(a, 0), mapping.get(b, 0)
                if all(sp.simplify(a_val * src + b_val - tgt) == 0
                       for src, tgt in zip(sources, targets, strict=True)):
                    def transform(expr, _a=a_val, _b=b_val):
                        return sp.expand(_a * sp.sympify(expr) + _b)
                    transform.description = (f"affine: {a_val} * x + {b_val} "
                                             f"(learned from {len(sources)} pair(s))")
                    return transform
        except Exception as exc:
            logger.debug("learn_transformation(): affine fallback failed: %s", exc)

        raise ValueError(
            "could not learn a transformation from the given examples: no SymPy "
            "operation and no affine relation reproduces all pairs"
        )


__all__ = [
    'EcosystemBridge',
    'EncryptionProvider',
    'MockChrono',
    'MockFortArch',
    'MockMorpho',
    'MockTopo',
    'TemporalPropagator',
    'TopologicalReasoner',
    'TransformationEngine',
]
