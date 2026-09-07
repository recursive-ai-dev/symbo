# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Core NanoTensor behaviour: construction, symbolic algebra, caching, evaluation.

These tests pin the guarantees the engine relies on: structural validation,
that every in-place mutation invalidates the derived caches (a stale lambdify
cache silently returns wrong numbers), that evaluation is exact and batchable,
and that an unassigned coefficient is reported instead of leaking a symbolic
object into the numeric result.
"""

import numpy as np
import pytest
import sympy as sp

from symbo import NanoTensor


class TestConstruction:
    def test_default_shape_and_vars(self):
        nt = NanoTensor((2, 2))
        assert nt.shape == (2, 2)
        assert nt.data.shape == (2, 2)
        assert nt.data[0, 0] == sp.S(0)
        assert [v.name for v in nt.base_vars] == ['k', 'a', 'eps', 'sig']

    def test_int_shape_is_normalised_to_tuple(self):
        assert NanoTensor(3).shape == (3,)

    @pytest.mark.parametrize("bad", [(0,), (-1, 2), (), "abc", (2.5, 'x')])
    def test_invalid_shapes_rejected(self, bad):
        if bad == (2.5, 'x') or bad == "abc":
            with pytest.raises((TypeError, ValueError)):
                NanoTensor(bad)
        else:
            with pytest.raises(ValueError):
                NanoTensor(bad)

    def test_max_order_must_be_positive(self):
        with pytest.raises(ValueError):
            NanoTensor((1,), max_order=0)

    def test_repr_shows_health(self):
        nt = NanoTensor((1,), name='brain')
        assert 'brain' in repr(nt)
        assert 'health=optimal' in repr(nt)

    def test_repr_html_is_escaped(self):
        nt = NanoTensor((1,), name='<img>')
        html = nt._repr_html_()
        assert '<img>' not in html
        assert '&lt;img&gt;' in html


class TestSymbolicAlgebra:
    def test_diff_returns_new_tensor_and_preserves_metadata(self):
        x, y = sp.symbols('x y')
        nt = NanoTensor((1,), max_order=2, base_vars=['x', 'y'], name='orig')
        nt.data[0] = x**3 + x * y
        d = nt.diff(x)
        assert d is not nt
        assert d.name == 'orig'
        assert d.max_order == 2
        assert sp.simplify(d.data[0] - (3 * x**2 + y)) == 0
        # diff must not touch the source tensor
        assert sp.simplify(nt.data[0] - (x**3 + x * y)) == 0

    def test_diff_recovery_does_not_mutate_self(self):
        """A failed derivative may simplify a *copy*, never the caller's tensor."""
        x = sp.symbols('x')
        nt = NanoTensor((1,), base_vars=['x'])
        expr = (x**8 + 2 * x**3) / (x**4 + x) + sp.sin(x) ** 2
        nt.data[0] = expr
        before = nt.data[0]
        result = nt.diff(x)
        assert isinstance(result, NanoTensor)
        assert nt.data[0] == before

    def test_diff_of_numeric_element_is_zero(self):
        nt = NanoTensor((1,), base_vars=['x'])
        nt.data[0] = 3.0  # a plain float, not a SymPy object
        assert nt.diff(sp.Symbol('x')).data[0] == 0

    def test_diff_failure_raises_runtime_error(self):
        nt = NanoTensor((1,))
        nt.data[0] = sp.Function('f')(sp.Symbol('x'))
        with pytest.raises(RuntimeError):
            nt.diff('not-a-symbol')

    def test_subs_accepts_names_and_symbols(self):
        x, y = sp.symbols('x y')
        nt = NanoTensor((1,), base_vars=['x', 'y'])
        nt.data[0] = x**2 + y
        assert nt.subs({'x': 3}).data[0] == 9 + y
        assert nt.subs({x: 3}).data[0] == 9 + y

    def test_subs_with_float_becomes_exact_rational(self):
        x = sp.symbols('x')
        nt = NanoTensor((1,), base_vars=['x'])
        nt.data[0] = x + sp.Rational(1, 3)
        assert nt.subs({'x': 0.5}).data[0] == sp.Rational(5, 6)

    def test_subs_drops_substituted_coefficient_vars(self):
        nt = NanoTensor((1,), max_order=1, base_vars=['k'])
        nt.generate_taylor({'k': 0.0}, ss_value=sp.S(0), include_bias=False)
        assert 'g_k' in [c.name for c in nt.coeff_vars]
        fitted = nt.subs({'g_k': 2.0})
        assert 'g_k' not in [c.name for c in fitted.coeff_vars]
        assert float(fitted.eval_numeric({'k': 3.0})[0]) == pytest.approx(6.0)

    def test_simplify_invalidates_lambdify_cache(self):
        """Regression: simplify() mutated data without clearing the compiled cache."""
        x = sp.symbols('x')
        nt = NanoTensor((1,), base_vars=['x'])
        nt.data[0] = (x + 1) ** 2 - x ** 2 - 2 * x  # == 1 after simplification
        assert float(nt.eval_numeric({'x': 5.0})[0]) == pytest.approx(1.0)
        nt.data[0] = x * (x + 1) - x**2  # == x
        nt.simplify()
        assert nt._lambdify_cache == {}
        assert float(nt.eval_numeric({'x': 7.0})[0]) == pytest.approx(7.0)

    def test_diff_cached_and_subs_cached(self):
        x = sp.symbols('x')
        nt = NanoTensor((1,), base_vars=['x'])
        nt.data[0] = x**2
        first = nt.diff_cached('x', 1)
        second = nt.diff_cached('x', 1)
        assert first is second
        assert nt._cache_hits >= 1

        point = tuple(sorted({'x': 2.0}.items()))
        sub_a = nt.subs_cached(point)
        sub_b = nt.subs_cached(point)
        assert sub_a is sub_b
        assert float(sub_a.data[0]) == 2.0 ** 2

    def test_subs_cache_is_bounded_and_instance_local(self):
        nt = NanoTensor((1,), base_vars=['x'])
        nt.data[0] = sp.Symbol('x')
        nt._max_cached_tensors = 4
        for i in range(10):
            nt.subs_cached((('x', float(i)),))
        assert len(nt._subs_cache) <= 4

    def test_deriv_tree_as_method_and_function(self):
        from symbo import deriv_tree

        x, y = sp.symbols('x y')
        nt = NanoTensor((1,), base_vars=['x', 'y'])
        nt.data[0] = x**2 * y
        method = nt.deriv_tree(['x', 'y'])
        assert sp.simplify(method['x'] - 2 * x * y) == 0
        assert sp.simplify(method['y'] - x**2) == 0
        # the historic free-function form keeps working
        assert sp.simplify(deriv_tree(nt, ['x', 'y'])['x'] - 2 * x * y) == 0


class TestEvaluation:
    def test_scalar_evaluation(self):
        x, y = sp.symbols('x y')
        nt = NanoTensor((2,), base_vars=['x', 'y'])
        nt.data[0] = x**2 + y
        nt.data[1] = x * y
        result = nt.eval_numeric({'x': 2.0, 'y': 3.0})
        assert result.shape == (2,)
        np.testing.assert_allclose(result, [7.0, 6.0])

    def test_batched_evaluation_adds_trailing_axis(self):
        x = sp.symbols('x')
        nt = NanoTensor((1,), base_vars=['x'])
        nt.data[0] = x**2
        out = nt.eval_numeric({'x': [1.0, 2.0, 3.0]})
        assert out.shape == (1, 3)
        np.testing.assert_allclose(out[0], [1.0, 4.0, 9.0])

    def test_batched_evaluation_of_2d_tensor(self):
        x, y = sp.symbols('x y')
        nt = NanoTensor((2, 2), base_vars=['x', 'y'])
        nt.data[0, 0] = x + y
        nt.data[1, 1] = x * y
        out = nt.eval_numeric({'x': np.array([1.0, 2.0]), 'y': np.array([3.0, 4.0])})
        assert out.shape == (2, 2, 2)
        np.testing.assert_allclose(out[0, 0], [4.0, 6.0])
        np.testing.assert_allclose(out[1, 1], [3.0, 8.0])

    def test_mismatched_batch_lengths_raise(self):
        nt = NanoTensor((1,), base_vars=['x', 'y'])
        nt.data[0] = sp.Symbol('x') + sp.Symbol('y')
        with pytest.raises(ValueError, match="same length"):
            nt.eval_numeric({'x': [1.0, 2.0], 'y': [3.0]})

    def test_unresolved_symbols_raise_value_error(self):
        nt = NanoTensor((1,), max_order=1, base_vars=['k'])
        nt.generate_taylor({'k': 0.0}, ss_value=sp.S(0))
        with pytest.raises(ValueError, match="no numeric value"):
            nt.eval_numeric({'k': 1.0})

    def test_use_lambdify_false_matches_lambdify_path(self):
        x, y = sp.symbols('x y')
        nt = NanoTensor((2,), base_vars=['x', 'y'])
        nt.data[0] = sp.sin(x) + y**2
        nt.data[1] = sp.exp(x * y)
        fast = nt.eval_numeric({'x': 0.5, 'y': 1.5})
        slow = nt.eval_numeric({'x': 0.5, 'y': 1.5}, use_lambdify=False)
        np.testing.assert_allclose(fast, slow, rtol=1e-12)

    def test_batched_symbolic_path_matches(self):
        x = sp.symbols('x')
        nt = NanoTensor((1,), base_vars=['x'])
        nt.data[0] = x**3
        np.testing.assert_allclose(
            nt.eval_numeric({'x': [1.0, 2.0, 3.0]}, use_lambdify=False)[0],
            [1.0, 8.0, 27.0],
        )

    def test_cache_is_reused_for_repeated_points(self):
        x = sp.symbols('x')
        nt = NanoTensor((1,), base_vars=['x'])
        nt.data[0] = x**2
        nt.eval_numeric({'x': 1.0})
        assert len(nt._lambdify_cache) == 1
        nt.eval_numeric({'x': 2.0})
        assert len(nt._lambdify_cache) == 1  # same symbol signature -> same entry

    def test_symvars_is_deterministically_ordered(self):
        x, y, z = sp.symbols('x y z')
        nt = NanoTensor((1,), base_vars=['x'])
        nt.data[0] = z + y + x
        assert [v.name for v in nt.symvars] == ['x', 'y', 'z']

    def test_simplify_invalidates_symvars_cache(self):
        x, y = sp.symbols('x y')
        nt = NanoTensor((1,), base_vars=['x'])
        nt.data[0] = sp.sin(y) ** 2 + sp.cos(y) ** 2 + x
        assert sorted(v.name for v in nt.symvars) == ['x', 'y']
        nt.simplify()
        assert nt._symvars_cache is None
        assert [v.name for v in nt.symvars] == ['x']


class TestValidationAndSecurity:
    def test_bounds_violation_raises(self):
        nt = NanoTensor((1,), base_vars=['x'])
        nt.data[0] = sp.Symbol('x')
        nt.set_validation_bounds('x', -1.0, 1.0)
        assert float(nt.eval_numeric({'x': 0.5})[0]) == 0.5
        with pytest.raises(ValueError, match="outside valid bounds"):
            nt.eval_numeric({'x': 5.0})

    def test_nan_and_inf_rejected(self):
        nt = NanoTensor((1,), base_vars=['x'])
        nt.data[0] = sp.Symbol('x')
        for bad in (float('nan'), float('inf'), -float('inf')):
            with pytest.raises(ValueError):
                nt.eval_numeric({'x': bad})

    def test_non_numeric_value_rejected(self):
        nt = NanoTensor((1,), base_vars=['x'])
        nt.data[0] = sp.Symbol('x')
        with pytest.raises(ValueError, match="Non-numeric"):
            nt.eval_numeric({'x': 'abc'})

    def test_symbol_value_accepted_when_exact(self):
        nt = NanoTensor((1,), base_vars=['x'])
        nt.data[0] = sp.Symbol('x')
        assert float(nt.eval_numeric({'x': sp.Rational(1, 4)})[0]) == 0.25

    def test_anomaly_detection_warns_on_outliers(self):
        nt = NanoTensor((1,), base_vars=['x'])
        nt.data[0] = sp.Symbol('x')
        for i in range(30):
            nt.eval_numeric({'x': 1.0 + 1e-3 * i})
        with pytest.warns(UserWarning, match="anomaly"):
            nt.eval_numeric({'x': 1000.0})

    def test_no_anomaly_before_enough_samples(self):
        nt = NanoTensor((1,), base_vars=['x'])
        nt.data[0] = sp.Symbol('x')
        nt._update_input_stats('x', 1.0)
        assert nt._check_input_anomaly('x', 1e9) is False  # n == 1 guard


class TestGridsAndPathfinding:
    def test_compute_grid_matches_pointwise_evaluation(self):
        x, y = sp.symbols('x y')
        nt = NanoTensor((1,), base_vars=['x', 'y'])
        nt.data[0] = x**2 + y
        X, Y, Z = nt.compute_grid('x', 'y', n1=4, n2=3)
        assert X.shape == Y.shape == Z.shape == (4, 3)  # axis 0 <-> var1
        np.testing.assert_allclose(X[:, 0], np.linspace(-0.5, 1.5, 4))
        np.testing.assert_allclose(Y[0, :], np.linspace(-0.2, 0.2, 3))
        for i in range(4):
            for j in range(3):
                expected = float(nt.eval_numeric({'x': float(X[i, j]),
                                                   'y': float(Y[i, j])})[0])
                assert Z[i, j] == pytest.approx(expected, rel=1e-9)

    def test_compute_grid_requires_scalar_tensor(self):
        nt = NanoTensor((2, 2), base_vars=['x'])
        nt.data[0, 0] = sp.Symbol('x')
        with pytest.raises(ValueError, match="scalar tensor"):
            nt.compute_grid('x', 'x')

    def test_find_path_on_grid_minimises_cost(self):
        Z = np.array([[0.0, 100.0, 0.0],
                      [0.0, 100.0, 0.0],
                      [0.0, 0.0, 0.0]])
        path = NanoTensor.find_path_on_grid(Z, (0, 0), (2, 2), mode="min")
        assert path[0] == (0, 0) and path[-1] == (2, 2)
        # the cheap route stays in column 0 until the bottom row
        assert (0, 1) not in path and (1, 1) not in path
        # non-negative grid + admissible heuristic => Dijkstra-optimal
        import heapq
        rng = np.random.default_rng(1)
        G = rng.random((5, 5))
        astar = NanoTensor.find_path_on_grid(G, (0, 0), (4, 4), mode="min")
        assert astar and astar[0] == (0, 0) and astar[-1] == (4, 4)

        def dijkstra(grid, start, goal):
            rows, cols = grid.shape
            pq = [(0.0, start)]
            dist = {start: 0.0}
            came = {}
            while pq:
                d, cur = heapq.heappop(pq)
                if cur == goal:
                    break
                if d != dist[cur]:
                    continue
                i, j = cur
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        nd = d + float(grid[ni, nj])
                        if nd < dist.get((ni, nj), float("inf")):
                            dist[(ni, nj)] = nd
                            came[(ni, nj)] = cur
                            heapq.heappush(pq, (nd, (ni, nj)))
            return dist[goal]

        astar_cost = sum(float(G[i, j]) for i, j in astar[1:])
        assert astar_cost == pytest.approx(dijkstra(G, (0, 0), (4, 4)))

    def test_find_path_on_grid_terminates_on_negative_landscape(self):
        # Negative entries make the naive A* re-open expanded nodes, which can
        # write a parent pointer back into an ancestor and make the goal-path
        # reconstruction loop forever (OOM). The search must terminate and
        # return a simple path.
        rng = np.random.default_rng(0)
        Z = rng.normal(size=(12, 12))
        path = NanoTensor.find_path_on_grid(Z, (0, 0), (5, 7))
        assert path
        assert path[0] == (0, 0) and path[-1] == (5, 7)
        assert len(set(path)) == len(path)  # no repeated nodes

    def test_find_path_out_of_bounds(self):
        with pytest.raises(ValueError, match="out of bounds"):
            NanoTensor.find_path_on_grid(np.zeros((2, 2)), (0, 0), (5, 5))

    def test_reason_path_returns_coordinate_path(self):
        x, y = sp.symbols('x y')
        nt = NanoTensor((1,), base_vars=['x', 'y'])
        nt.data[0] = x**2 + y**2
        result = nt.reason_path('x', 'y',
                                start_state={'x': -1.0, 'y': -1.0},
                                goal_state={'x': 1.0, 'y': 1.0},
                                range1=(-1.0, 1.0), range2=(-1.0, 1.0),
                                n1=7, n2=7)
        assert len(result['path']) > 1
        assert result['path'][0]['x'] == pytest.approx(-1.0, abs=0.4)

    def test_stream_groebner_basis_yields_json_chunks(self):
        import json

        x, y = sp.symbols('x y')
        chunks = [json.loads(line) for line in
                  NanoTensor.stream_groebner_basis([x + y - 2, x - y], [x, y])]
        assert all(c['type'] == 'groebner_chunk' for c in chunks)
        polys = [item['poly_str'] for c in chunks for item in c['items']]
        assert any('x - 1' in p for p in polys)


class TestPolynomialAndCurve:
    def test_solve_poly_returns_real_roots(self):
        x = sp.symbols('x')
        nt = NanoTensor((1,))
        roots = nt.solve_poly(x**2 - 4, x)
        assert sorted(roots) == [-2.0, 2.0]

    def test_solve_poly_ignores_complex_roots_and_reports_failure(self):
        x = sp.symbols('x')
        nt = NanoTensor((1,))
        assert nt.solve_poly(x**2 + 1, x) == []
        assert nt.solve_poly(sp.sin(x), x) == []  # not a polynomial

    def test_groebner_solve_filters_real_solutions(self):
        x, y = sp.symbols('x y')
        nt = NanoTensor((1,))
        sols = nt.groebner_solve([x**2 + y**2 - 1, x - y], [x, y])
        assert len(sols) == 2
        for sol in sols:
            assert abs(float(sol['x']) - float(sol['y'])) < 1e-10

    def test_resultant_and_parametrize_curve(self):
        x, y, t = sp.symbols('x y t')
        nt = NanoTensor((1,))
        # eliminating y from {x+y=2, x=y} leaves 2*x - 2
        got = sp.expand(nt.resultant(x + y - 2, x - y, y))
        assert got == sp.expand(sp.resultant(x + y - 2, x - y, y))
        assert y not in got.free_symbols  # y really was eliminated

        folium = x**3 + y**3 - 3 * x * y
        px, py = nt.parametrize_curve(folium, t=t)
        # the parametrisation must satisfy the implicit equation identically
        assert sp.simplify(px**3 + py**3 - 3 * px * py) == 0
        yp = sp.symbols("yp")
        with pytest.raises(ValueError, match="extra symbols"):
            nt.parametrize_curve(yp**2 + 3 * yp - 2 * y - 3 * x, t=t)
        with pytest.raises(ValueError, match="singular"):
            nt.parametrize_curve(x**2 + y**2 - 1, t=t)

    def test_solve_poly_failure_is_logged_not_printed(self, caplog):
        import logging

        x = sp.symbols('x')
        with caplog.at_level(logging.DEBUG, logger='symbo.nanotensor'):
            NanoTensor((1,)).solve_poly(sp.sin(x), x)
        assert any('no polynomial form' in r.message for r in caplog.records)


class TestOptimizationAndPersistence:
    def test_optimize_storage_speeds_up_but_stays_correct(self):
        x, y = sp.symbols('x y')
        nt = NanoTensor((1,), base_vars=['x', 'y'])
        nt.data[0] = (x + y) ** 6 + (x + y) ** 3
        assert nt.optimize_storage() is True
        assert nt._optimized_func is not None
        assert float(nt.eval_numeric({'x': 1.0, 'y': 2.0})[0]) == pytest.approx(3**6 + 3**3)

    def test_optimize_storage_noop_for_constants(self):
        nt = NanoTensor((1,))
        nt.data[0] = sp.S(5)
        assert nt.optimize_storage() is False

    def test_simplify_after_optimize_clears_optimized_callable(self):
        x = sp.symbols('x')
        nt = NanoTensor((1,), base_vars=['x'])
        nt.data[0] = (x + 1) ** 2 - x**2 - 2 * x
        nt.optimize_storage()
        nt.simplify()
        assert nt._optimized_func is None
        assert float(nt.eval_numeric({'x': 3.0})[0]) == pytest.approx(1.0)

    def test_save_and_load_brain_roundtrip(self, tmp_path):
        x = sp.symbols('x')
        nt = NanoTensor((1,), max_order=1, base_vars=['x'], name='saved')
        nt.data[0] = 3 * x**2
        nt.eval_numeric({'x': 2.0})
        path = tmp_path / "brain.pkl"
        ops_before_save = nt._operation_count
        nt.save_brain(str(path))
        assert path.exists()

        loaded = NanoTensor.load_brain(str(path))
        assert loaded.name == 'saved'
        assert loaded.data[0] == nt.data[0]
        # operation metrics survive the round trip ...
        assert loaded._operation_count >= ops_before_save
        # ... while derived state is rebuilt on demand, never serialized
        assert loaded._lambdify_cache == {}
        assert float(loaded.eval_numeric({'x': 2.0})[0]) == pytest.approx(12.0)
        assert loaded._lambdify_cache != {}  # and gets repopulated by the next eval

    def test_plain_pickle_roundtrip_without_dill(self):
        """``pickle`` alone must work: compiled callables stay out of the state."""
        import pickle

        x = sp.symbols('x')
        nt = NanoTensor((1,), base_vars=['x'])
        nt.data[0] = sp.cos(x) + x**2
        nt.eval_numeric({'x': 0.7})  # populates the lambdify cache
        payload = pickle.dumps(nt)
        restored = pickle.loads(payload)
        assert restored._lambdify_cache == {}
        np.testing.assert_allclose(restored.eval_numeric({'x': 0.7}),
                                   nt.eval_numeric({'x': 0.7}))

    def test_load_brain_survives_missing_optional_dill(self, tmp_path, monkeypatch):
        """Without dill the stdlib pickle module is used instead."""
        from symbo import nanotensor as nt_module

        x = sp.symbols('x')
        nt = NanoTensor((1,), base_vars=['x'])
        nt.data[0] = 2 * x
        monkeypatch.setattr(nt_module, "dill", None)
        path = tmp_path / "brain2.pkl"
        nt.save_brain(str(path))
        loaded = NanoTensor.load_brain(str(path))
        assert float(loaded.eval_numeric({'x': 4.0})[0]) == pytest.approx(8.0)

    def test_load_brain_reports_corrupt_file(self, tmp_path):
        path = tmp_path / "broken.pkl"
        path.write_bytes(b"not a serialized brain")
        with pytest.raises(RuntimeError, match="Failed to load brain"):
            NanoTensor.load_brain(str(path))


class TestAgencyReporting:
    def test_operations_are_recorded(self, tiny_tensor):
        nt, x = tiny_tensor
        nt.eval_numeric({'x': 1.0})
        nt.diff(x)
        health = nt.health_check()
        assert health['metrics']['total_operations'] == 2
        assert health['metrics']['success_rate'] == 1.0
        assert set(health['learned_patterns']) == {'evaluation', 'differentiation'}

    def test_failures_are_recorded_and_recommend(self, tiny_tensor):
        nt, _ = tiny_tensor
        with pytest.raises(ValueError):
            nt.eval_numeric({'y': 1.0})  # 'y' is not a variable of this tensor
        status = nt.get_agency_status()
        assert status['health'] in ('degraded', 'critical')
        assert any('simplif' in r.lower() for r in status['recommendations'])

    def test_unknown_point_keys_are_ignored_with_debug_log(self, caplog):
        import logging

        nt = NanoTensor((1,), base_vars=['x'])
        nt.data[0] = 2 * sp.Symbol('x')
        with caplog.at_level(logging.DEBUG, logger='symbo.nanotensor'):
            # 'other' is not used by the expression, so it is simply ignored
            assert float(nt.eval_numeric({'x': 2.0, 'other': 9.0})[0]) == pytest.approx(4.0)

    def test_self_optimization_does_not_recurse(self):
        """A degraded tensor must not recurse through simplify -> health -> optimize."""
        x = sp.symbols('x')
        nt = NanoTensor((1,), base_vars=['x'])
        nt.data[0] = x**2
        nt._auto_optimize = True
        # force the degraded path with a poor success rate and many operations
        for i in range(120):
            nt._record_operation("evaluation", 0.0001, success=(i % 5 == 0))
        assert nt._health_status in ('degraded', 'critical')
        nt._attempt_self_optimization()  # would raise RecursionError before the fix
