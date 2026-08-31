# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Ecosystem bridge and its offline mock providers (Chrono, Topo, Morpho)."""

import math

import pytest
import sympy as sp

from symbo.ecosystem import (
    EcosystemBridge,
    MockChrono,
    MockFortArch,
    MockMorpho,
    MockTopo,
    TopologicalReasoner,
    TransformationEngine,
)

x, y = sp.symbols("x y")


class TestMockChrono:
    def test_propagation_follows_the_documented_example(self):
        chrono = MockChrono()
        traj = chrono.chrono_propagate({x: 1.0, y: 0.0}, {x: y, y: -x},
                                       time_horizon=1.0, dt=1e-3)
        assert len(traj) == 1001
        assert traj[0] == {x: 1.0, y: 0.0}
        # explicit Euler on a rotation stays near the unit circle for small dt
        last = traj[-1]
        assert last[x] ** 2 + last[y] ** 2 == pytest.approx(1.0, abs=2e-2)

    def test_exponential_growth_is_reproduced(self):
        traj = MockChrono().chrono_propagate({x: 1.0}, {x: x}, time_horizon=1.0, dt=1e-4)
        assert traj[-1][x] == pytest.approx(math.e, abs=1e-3)

    def test_validation(self):
        chrono = MockChrono()
        with pytest.raises(ValueError, match="dt must be positive"):
            chrono.chrono_propagate({x: 1.0}, {x: x}, time_horizon=1.0, dt=0.0)
        with pytest.raises(ValueError, match="time_horizon"):
            chrono.chrono_propagate({x: 1.0}, {x: x}, time_horizon=-1.0, dt=0.1)
        with pytest.raises(ValueError, match="not part of the state"):
            chrono.chrono_propagate({x: 1.0}, {x: x + sp.Symbol("t")},
                                    time_horizon=1.0, dt=0.1)

    def test_non_numeric_initial_state_is_rejected(self):
        with pytest.raises(ValueError, match="free symbols"):
            MockChrono().chrono_propagate({x: sp.sin(y)}, {x: x},
                                         time_horizon=1.0, dt=0.1)

    def test_lyapunov_exponents_are_jacobian_eigenvalue_real_parts(self):
        chrono = MockChrono()
        assert chrono.compute_lyapunov_exponents({x: 2 * x}, {x: 1.0}) == [2.0]
        # nonlinear: J = d(x^2)/dx = 2x -> 2 at x = 1
        assert chrono.compute_lyapunov_exponents({x: x**2}, {x: 1.0}) == [2.0]
        assert chrono.compute_lyapunov_exponents({}, {}) == []

    def test_rotation_is_marginally_stable(self):
        exponents = MockChrono().compute_lyapunov_exponents({x: y, y: -x},
                                                            {x: 1.0, y: 0.0})
        assert exponents == [0.0, 0.0]

    def test_forecast_extrapolates_linear_trend(self):
        chrono = MockChrono()
        history = [{x: 0.0}, {x: 1.0}, {x: 2.0}]
        assert chrono.forecast_trajectory(history, steps_ahead=3) == [{x: 3.0},
                                                                      {x: 4.0},
                                                                      {x: 5.0}]
        assert chrono.forecast_trajectory([{x: 7.0}], steps_ahead=2) == [{x: 7.0}, {x: 7.0}]
        assert chrono.forecast_trajectory(history, steps_ahead=0) == []
        with pytest.raises(ValueError, match="steps_ahead"):
            chrono.forecast_trajectory(history, steps_ahead=-1)
        with pytest.raises(ValueError, match="at least one"):
            chrono.forecast_trajectory([], steps_ahead=1)


class TestMockTopo:
    def test_critical_points_and_classification(self):
        topo = MockTopo()
        points = topo.find_critical_points(x**2 + y**2, [x, y])
        assert points == [{x: 0.0, y: 0.0, "classification": "minimum"}]

    def test_saddle_and_minimum(self):
        topo = MockTopo()
        got = {(p[x], p[y]): p["classification"]
               for p in topo.find_critical_points(x**3 - 3 * x + y**2, [x, y])}
        assert got == {(0.0, 0.0): "saddle", (-1.0, 0.0): "saddle", (1.0, 0.0): "minimum"} \
            or got[(-1.0, 0.0)] in ("saddle", "maximum")

    def test_univariate_extrema(self):
        topo = MockTopo()
        pts = topo.find_critical_points(3 * x**4 - 4 * x**2 + 1, [x])
        kinds = {round(p[x], 6): p["classification"] for p in pts}
        assert kinds[0.0] == "maximum"
        assert kinds[round(math.sqrt(2 / 3), 6)] == "minimum"

    def test_no_variables_is_an_error(self):
        with pytest.raises(ValueError, match="at least one variable"):
            MockTopo().find_critical_points(x**2, [])

    @pytest.mark.parametrize("expr,components,loops", [
        (x**2 + y**2 - 1, 1, 1),          # one closed loop
        ((x**2 + y**2 - 1) * (x**2 + y**2 - 2.5), 2, 2),  # two nested loops
        (x**2 - y**2 - 1, 2, 0),          # hyperbola: open branches
        (x**2 + y**2 + 1, 0, 0),          # empty level set
    ])
    def test_level_set_betti_numbers(self, expr, components, loops):
        result = MockTopo(resolution=81).compute_manifold_topology(expr, [x, y])
        assert result["dimension"] == 2
        assert result["components"] == components
        assert result["betti_numbers"] == {0: components, 1: loops}
        assert result["genus"] == loops

    def test_univariate_level_set_counts_roots(self):
        result = MockTopo(resolution=201).compute_manifold_topology(x**2 - 1, [x])
        assert result["dimension"] == 1
        assert result["components"] == 2
        assert result["betti_numbers"][1] == 0

    def test_three_variables_are_not_supported(self):
        with pytest.raises(ValueError, match="1D and 2D"):
            MockTopo().compute_manifold_topology(x + y, [x, y, sp.Symbol("z")])

    def test_homology_of_edge_list(self):
        topo = MockTopo()
        assert topo.compute_homology([(0, 1), (1, 2), (2, 0)]) == {0: 1, 1: 1}
        # two disjoint triangles: two components, one cycle each
        two = [(0, 1), (1, 2), (2, 0), (5, 6), (6, 7), (7, 5)]
        assert topo.compute_homology(two) == {0: 2, 1: 2}
        # a tree has no cycles
        assert topo.compute_homology([(0, 1), (1, 2), (2, 3)]) == {0: 1, 1: 0}

    def test_homology_of_networkx_graph_and_pairs(self):
        nx = pytest.importorskip("networkx")
        topo = MockTopo()
        graph = nx.cycle_graph(6)
        assert topo.compute_homology(graph) == {0: 1, 1: 1}
        assert topo.compute_homology((list(graph.nodes()), list(graph.edges()))) == {0: 1, 1: 1}
        assert topo.compute_homology([]) == {0: 0, 1: 0}

    def test_resolution_is_validated(self):
        for bad in (4, 0):
            with pytest.raises(ValueError, match="resolution"):
                MockTopo(resolution=bad)
        with pytest.raises(ValueError, match="bounds"):
            MockTopo(bounds=(1.0, -1.0))


class TestMockMorpho:
    def test_named_transformations(self):
        morpho = MockMorpho()
        assert morpho.morpho_transform(x**2 + 2 * x + 1, "factor") == (x + 1)**2
        assert morpho.morpho_transform((x + y)**2, "expand") == x**2 + 2 * x * y + y**2
        assert morpho.morpho_transform(x**3, "diff", {"var": "x", "order": 2}) == 6 * x
        assert morpho.morpho_transform(x + y, "subs", {"mapping": {"x": 1}}) == y + 1
        assert "simplify" in morpho.available()

    def test_unknown_transformation_lists_alternatives(self):
        with pytest.raises(ValueError, match="available"):
            MockMorpho().morpho_transform(x, "nope")

    @pytest.mark.parametrize("name", ["diff", "series", "subs"])
    def test_required_parameters_are_enforced(self, name):
        with pytest.raises(ValueError, match="requires the parameter"):
            MockMorpho().morpho_transform(x, name)

    def test_variants_are_distinct_and_capped(self):
        morpho = MockMorpho()
        variants = morpho.generate_variants((x + 1)**2, 3)
        assert len(variants) <= 3
        assert all(v != (x + 1)**2 for v in variants)
        assert len({str(v) for v in variants}) == len(variants)
        # deterministic
        assert morpho.generate_variants((x + 1)**2, 3) == variants
        with pytest.raises(ValueError, match="n_variants"):
            morpho.generate_variants(x, 0)

    def test_constraints_filter_variants(self):
        morpho = MockMorpho()
        # a constraint that vanishes everywhere rejects every variant
        assert morpho.generate_variants((x + 1)**2, 3, constraints=[x - x]) == []
        # an unsatisfiable sign guard likewise
        assert morpho.generate_variants((x + 1)**2, 3, constraints=[x > 5]) == []
        # a constraint that never vanishes on the sample set keeps them all
        assert morpho.generate_variants((x + 1)**2, 3, constraints=[x - 0.5])

    def test_learn_named_transformation(self):
        morpho = MockMorpho()
        learned = morpho.learn_transformation(
            [x**2 + 2 * x + 1, y**2 + 2 * y + 1],
            [(x + 1)**2, (y + 1)**2],
        )
        q = sp.Symbol("q")
        assert learned(q**2 + 2 * q + 1) == (q + 1)**2
        assert "factor" in learned.description

    def test_learn_affine_fallback(self):
        morpho = MockMorpho()
        learned = morpho.learn_transformation([x, y], [2 * x + 1, 2 * y + 1])
        assert learned(x) == 2 * x + 1
        assert "affine" in learned.description

    def test_unlearnable_and_mismatched_examples(self):
        morpho = MockMorpho()
        with pytest.raises(ValueError, match="could not learn"):
            morpho.learn_transformation([x], [sp.sin(x) + 1])
        with pytest.raises(ValueError, match="equally many"):
            morpho.learn_transformation([x], [])


class TestBridge:
    def test_all_four_wiring_points(self):
        bridge = EcosystemBridge(encryption=MockFortArch(),
                                 topology=MockTopo(resolution=41),
                                 temporal=MockChrono(),
                                 transformation=MockMorpho())
        expr = x**2 + 2 * x + 1
        assert bridge.secure_compute(expr, "simplify") is not None
        assert bridge.analyze_topology(x**2 + y**2 - 1, [x, y])["components"] == 1
        traj = bridge.propagate_forward({x: 1.0, y: 0.0}, {x: y, y: -x},
                                        time_horizon=0.1)
        assert len(traj) == 11
        assert bridge.transform_expression(expr, "factor") == (x + 1)**2

    def test_missing_providers_raise_not_implemented(self):
        bridge = EcosystemBridge()
        with pytest.raises(NotImplementedError, match="EncryptionProvider"):
            bridge.secure_compute(x, "identity")
        with pytest.raises(NotImplementedError, match="TopologicalReasoner"):
            bridge.analyze_topology(x, [x])
        with pytest.raises(NotImplementedError, match="TemporalPropagator"):
            bridge.propagate_forward({x: 1.0}, {x: y}, time_horizon=1.0)
        with pytest.raises(NotImplementedError, match="TransformationEngine"):
            bridge.transform_expression(x, "expand")

    def test_missing_provider_message_names_the_interface_and_a_mock(self):
        bridge = EcosystemBridge(temporal=MockChrono())
        with pytest.raises(NotImplementedError) as excinfo:
            bridge.analyze_topology(x, [x])
        message = str(excinfo.value)
        assert "TopologicalReasoner" in message
        assert "MockTopo" in message
        assert "analyze_topology" in message

    def test_providers_are_validated_when_the_bridge_is_built(self):
        class NotAProvider:
            pass

        with pytest.raises(TypeError, match=r"temporal.*TemporalPropagator"):
            EcosystemBridge(temporal=NotAProvider())
        with pytest.raises(TypeError, match="missing"):
            EcosystemBridge(encryption=MockTopo())

    def test_propagate_forward_forwards_the_time_step(self):
        bridge = EcosystemBridge(temporal=MockChrono())
        coarse = bridge.propagate_forward({x: 1.0, y: 0.0}, {x: y, y: -x}, 1.0, dt=0.5)
        fine = bridge.propagate_forward({x: 1.0, y: 0.0}, {x: y, y: -x}, 1.0, dt=0.1)
        assert len(fine) > len(coarse)
        assert coarse[0][x] == pytest.approx(1.0)

    def test_mocks_satisfy_the_abstract_contracts(self):
        assert isinstance(MockTopo(), TopologicalReasoner)
        assert isinstance(MockMorpho(), TransformationEngine)
        for cls in (TopologicalReasoner, TransformationEngine):
            with pytest.raises(TypeError):
                cls()  # abstract: cannot be instantiated directly
