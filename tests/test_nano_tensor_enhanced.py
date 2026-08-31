# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""The standalone hardened tensor in ``symbo.nano_tensor_enhanced``.

This class exists so a single file can be vendored into an agent runtime; the
tests below pin the parts that make it safe to run unattended: input validation,
constraint checking, cache correctness under mutation, health reporting and
picklability.
"""

import pickle

import numpy as np
import pytest
import sympy as sp

from symbo.nano_tensor_enhanced import (
    AgencyCore,
    Experience,
    HealthStatus,
    MilitaryGradeNanoTensor,
    OperationType,
    PerformanceMetrics,
)

x, y = sp.symbols("x y")


@pytest.fixture()
def brain():
    nt = MilitaryGradeNanoTensor((2,), base_vars=["x", "y"])
    nt.data[0] = x**2 + y
    nt.data[1] = sp.sin(x) * y
    return nt


class TestConstruction:
    def test_shape_and_defaults(self, brain):
        assert brain.shape == (2,)
        assert brain.name == "mgnt"
        assert brain.max_order == 2
        assert brain.base_vars == [x, y]
        assert all(e == 0 for e in MilitaryGradeNanoTensor((3,)).data)

    @pytest.mark.parametrize("bad_shape", [(0,), (2.5,), (-1,), "xy"])
    def test_invalid_shapes_are_rejected(self, bad_shape):
        with pytest.raises((TypeError, ValueError)):
            MilitaryGradeNanoTensor(bad_shape)

    def test_max_order_must_be_positive(self):
        with pytest.raises(ValueError, match="max_order"):
            MilitaryGradeNanoTensor((1,), max_order=0)

    def test_agency_and_security_can_be_disabled(self):
        plain = MilitaryGradeNanoTensor((1,), enable_agency=False, enable_security=False)
        assert plain.agency is None
        assert plain._validation_enabled is False


class TestSymbolicOperations:
    def test_symvars_tracks_the_current_data(self, brain):
        assert brain.symvars == [x, y]
        brain2 = MilitaryGradeNanoTensor((1,))
        assert brain2.symvars == []
        brain2.data[0] = x * y
        assert brain2.symvars == [x, y]  # direct assignment must not go stale

    def test_diff_returns_a_new_tensor(self, brain):
        derivative = brain.diff(x)
        assert derivative is not brain
        assert derivative.data[0] == 2 * x
        assert brain.data[0] == x**2 + y  # original untouched

    def test_diff_second_order(self, brain):
        assert brain.diff(x, order=2).data[0] == 2

    def test_subs_returns_a_new_tensor(self, brain):
        substituted = brain.subs({x: y})
        assert substituted is not brain
        assert substituted.data[0] == y**2 + y
        assert brain.data[0] == x**2 + y

    def test_simplify_mutates_in_place_and_returns_self(self, brain):
        brain.data[0] = (x + x) ** 2
        assert brain.simplify() is brain
        assert brain.data[0] == 4 * x**2

    def test_caches_are_invalidated_by_simplify(self, brain):
        brain.eval_numeric({"x": 1.0, "y": 1.0})
        brain.data[0] = x**2 + y
        brain.simplify()
        assert brain.eval_numeric({"x": 2.0, "y": 1.0}) == pytest.approx([5.0, 0.9092974268256817], rel=1e-9)


class TestNumericEvaluation:
    def test_evaluates_every_entry(self, brain):
        result = brain.eval_numeric({"x": 2.0, "y": 1.0})
        assert result.shape == (2,)
        assert result[0] == pytest.approx(5.0)
        assert result[1] == pytest.approx(float(sp.sin(2)))

    def test_accepts_symbol_keys(self, brain):
        assert brain.eval_numeric({x: 2.0, y: 1.0})[0] == pytest.approx(5.0)

    def test_declared_variables_default_to_zero(self):
        nt = MilitaryGradeNanoTensor((1,), base_vars=["x", "a"])
        nt.data[0] = x**2
        assert nt.eval_numeric({})[0] == 0.0

    def test_undeclared_symbols_are_reported(self):
        nt = MilitaryGradeNanoTensor((1,), base_vars=["q"])
        nt.data[0] = x**2 + y
        with pytest.raises(ValueError, match=r"no value for y"):
            nt.eval_numeric({"x": 1.0})

    def test_cache_is_keyed_on_the_expressions_not_just_the_point(self, brain):
        first = brain.eval_numeric({"x": 2.0, "y": 1.0})
        assert brain.eval_numeric({"x": 2.0, "y": 1.0}) == pytest.approx(first)
        hits_before = brain.metrics.cache_hits
        brain.data[0] = x**2 + 10 * y          # mutate *after* caching
        fresh = brain.eval_numeric({"x": 2.0, "y": 1.0})
        assert fresh[0] == pytest.approx(14.0)
        assert brain.metrics.cache_hits == hits_before   # stale entry was not used

    def test_repeated_identical_evaluation_hits_the_cache(self, brain):
        brain.eval_numeric({"x": 1.0, "y": 1.0})
        misses = brain.metrics.cache_misses
        brain.eval_numeric({"x": 1.0, "y": 1.0})
        assert brain.metrics.cache_hits == 1
        assert brain.metrics.cache_misses == misses

    def test_cache_eviction_keeps_the_bound_constant(self, brain):
        nt = MilitaryGradeNanoTensor((1,), base_vars=["x"], max_cache_size=4)
        nt.data[0] = x
        for i in range(10):
            nt.eval_numeric({"x": float(i)})
        assert len(nt._eval_cache) <= 4


class TestValidation:
    def test_bounds_are_enforced(self, brain):
        brain.set_bounds("x", 0.0, 1.0)
        assert brain.eval_numeric({"x": 0.5, "y": 0.0})[0] == pytest.approx(0.25)
        with pytest.raises(ValueError, match="outside bounds"):
            brain.eval_numeric({"x": 5.0, "y": 0.0})

    def test_nan_and_inf_are_rejected(self, brain):
        for bad in (float("nan"), float("inf")):
            with pytest.raises(ValueError):
                brain.eval_numeric({"x": bad, "y": 0.0})

    def test_constraints_are_checked(self, brain):
        brain.add_constraint(lambda point: point["x"] >= point["y"])
        assert brain.eval_numeric({"x": 2.0, "y": 1.0}).shape == (2,)
        with pytest.raises(ValueError, match="Constraint violation"):
            brain.eval_numeric({"x": 1.0, "y": 2.0})

    def test_security_can_be_switched_off(self, brain):
        brain.set_bounds("x", 0.0, 1.0)
        brain._validation_enabled = False
        assert brain.eval_numeric({"x": 5.0, "y": 0.0})[0] == pytest.approx(25.0)


class TestHealthAndMetrics:
    def test_metrics_track_success_and_cache_use(self, brain):
        brain.eval_numeric({"x": 1.0, "y": 1.0})
        brain.eval_numeric({"x": 1.0, "y": 1.0})
        metrics = brain.metrics
        assert metrics.total_operations == 2
        assert metrics.successful_operations == 2
        assert metrics.success_rate == 1.0
        assert metrics.cache_misses == 1 and metrics.cache_hits == 1
        assert metrics.avg_operation_time >= 0.0

    def test_failed_operations_are_counted(self, brain):
        brain.set_bounds("x", 0.0, 1.0)
        with pytest.raises(ValueError):
            brain.eval_numeric({"x": 9.0, "y": 0.0})
        assert brain.metrics.failed_operations == 1
        assert brain.metrics.success_rate < 1.0

    def test_health_check_schema(self, brain):
        brain.eval_numeric({"x": 1.0, "y": 1.0})
        health = brain.health_check()
        assert health["status"] in {"optimal", "good", "degraded", "critical", "failed"}
        assert set(health["metrics"]) >= {"total_operations", "success_rate",
                                         "cache_hit_rate", "avg_operation_time"}
        assert set(health["cache_sizes"]) >= {"diff", "subs", "eval"}
        assert health["security"]["validation_enabled"] is True
        assert health["agency_status"]["total_experiences"] >= 1

    def test_degraded_health_when_operations_fail(self, brain):
        brain.set_bounds("x", 0.0, 1.0)
        for _ in range(20):
            with pytest.raises(ValueError):
                brain.eval_numeric({"x": 9.0, "y": 0.0})
        assert brain.health_status != HealthStatus.OPTIMAL
        assert brain.health_check()["status"] in {"degraded", "critical", "failed"}

    def test_optimize_reports_a_strategy(self, brain):
        brain.eval_numeric({"x": 1.0, "y": 1.0})
        brain.optimize()   # must not raise even with few samples
        assert brain.metrics.total_operations >= 1


class TestAgencyCore:
    def test_goals_are_recorded_by_priority(self):
        agency = AgencyCore()
        agency.add_goal("fit", "g_k_a", priority=0.5)
        agency.add_goal("solve", "groebner", priority=2.0)
        status = agency.get_status()
        assert status["active_goals"] == 2
        assert status["current_goal"]["priority"] == 2.0
        assert agency.goals[0]["type"] == "solve"
        # completing it promotes the remaining goal
        finished = agency.complete_goal()
        assert finished["type"] == "solve" and finished["progress"] == 1.0
        assert agency.get_status()["current_goal"]["type"] == "fit"
        assert agency.complete_goal()["type"] == "fit"
        assert agency.current_goal is None
        assert agency.complete_goal() is None

    def test_experience_learning_and_recommendation(self):
        agency = AgencyCore()
        for _ in range(12):
            agency.record_experience(Experience(operation=OperationType.EVALUATION,
                                               inputs={}, outputs=0.0, duration=0.5,
                                               success=True))
        summary = agency.get_status()["pattern_summary"]["eval"]
        assert summary["count"] == 12
        assert summary["success_rate"] == 1.0
        recommendation = agency.recommend_optimization(OperationType.EVALUATION)
        assert recommendation["use_cache"] is True
        assert recommendation["confidence"] > 0.0

    def test_anomaly_needs_both_scale_and_samples(self):
        agency = AgencyCore()
        # sub-millisecond operations are noise, however they are distributed
        for _ in range(20):
            agency.record_experience(Experience(operation=OperationType.EVALUATION,
                                               inputs={}, outputs=0.0, duration=1e-6,
                                               success=True))
        assert agency.detect_anomaly(OperationType.EVALUATION, 1e-5) is False
        for _ in range(20):
            agency.record_experience(Experience(operation=OperationType.DIFFERENTIATION,
                                               inputs={}, outputs=0.0, duration=0.5,
                                               success=True))
        assert agency.detect_anomaly(OperationType.DIFFERENTIATION, 5.0) is True
        assert agency.detect_anomaly(OperationType.DIFFERENTIATION, 0.5) is False

    def test_memory_is_bounded(self):
        agency = AgencyCore(max_memory_size=5)
        for i in range(20):
            agency.record_experience(Experience(operation=OperationType.EVALUATION,
                                               inputs={}, outputs=float(i),
                                               duration=float(i), success=True))
        assert len(agency.experience_buffer) == 5


class TestSerialisation:
    def test_pickle_round_trip_drops_transient_state(self, brain):
        brain.eval_numeric({"x": 1.0, "y": 1.0})
        clone = pickle.loads(pickle.dumps(brain))
        assert [str(e) for e in clone.data] == [str(e) for e in brain.data]
        assert clone.base_vars == brain.base_vars
        assert clone._diff_cache == {} and clone._lambdify_cache == {}
        # ... and it still evaluates identically
        assert clone.eval_numeric({"x": 2.0, "y": 1.0})[0] == pytest.approx(5.0)

    def test_numpy_data_survives_the_round_trip(self, brain):
        clone = pickle.loads(pickle.dumps(brain))
        assert isinstance(clone.data, np.ndarray)
        assert clone.data.dtype == object
        assert clone.symvars == brain.symvars


class TestMetricsDataclass:
    def test_rates_are_safe_when_nothing_happened(self):
        metrics = PerformanceMetrics()
        assert metrics.success_rate == 1.0
        assert metrics.cache_hit_rate == 0.0
        assert metrics.avg_operation_time == 0.0

    def test_rates_after_work(self):
        metrics = PerformanceMetrics(total_operations=4, successful_operations=3,
                                    cache_hits=1, cache_misses=3, total_compute_time=2.0)
        assert metrics.success_rate == 0.75
        assert metrics.cache_hit_rate == 0.25
        assert metrics.avg_operation_time == 0.5
