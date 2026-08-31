# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Wire formats: SymboSerializer (msgpack / JSON / Arrow) and the self-check."""

import json

import pytest
import sympy as sp

from symbo import SymbolicTensor
from symbo.generative.taylor import TaylorExpansion
from symbo.io.serialization import (
    SerializationError,
    SymboSerializer,
    verify_round_trip_serialization,
)
from symbo.solver.groebner import GröbnerBasisState

x, y = sp.symbols("x y")


class TestExpressions:
    @pytest.mark.parametrize("fmt", ["msgpack", "json"])
    def test_round_trip(self, fmt):
        if fmt == "msgpack":
            pytest.importorskip("msgpack")
        expr = x**2 + sp.sin(y) + sp.Rational(1, 3)
        payload = SymboSerializer.serialize_expression(expr, fmt)
        assert isinstance(payload, bytes)
        assert SymboSerializer.verify_equivalence(expr,
                                                  SymboSerializer.deserialize_expression(payload, fmt))

    def test_json_payload_is_readable_by_a_browser(self):
        payload = json.loads(SymboSerializer.serialize_expression(x + 1, "json"))
        assert set(payload) >= {"string_repr", "type"}
        assert payload["string_repr"] == "x + 1"

    def test_unsupported_format(self):
        with pytest.raises(SerializationError, match="unsupported format"):
            SymboSerializer.serialize_expression(x, "pickle")

    def test_corrupt_payload_raises_serialization_error(self):
        with pytest.raises(SerializationError):
            SymboSerializer.deserialize_expression(b"\xff\xff not msgpack", "msgpack")

    def test_wrong_type_marker_is_refused(self):
        fake = json.dumps({"string_repr": "x"}).encode("utf-8")
        with pytest.raises(SerializationError, match="expected"):
            SymboSerializer.deserialize_expression(fake, "json")


class TestTensors:
    @pytest.fixture()
    def tensor(self):
        t = SymbolicTensor((2, 2), name="M")
        t.fill_with_symbols("m")
        return t

    @pytest.mark.parametrize("fmt", ["arrow", "msgpack"])
    def test_round_trip(self, tensor, fmt):
        if fmt == "arrow":
            pytest.importorskip("pyarrow")
        payload = SymboSerializer.serialize_tensor(tensor, fmt)
        restored = SymboSerializer.deserialize_tensor(payload, fmt)
        assert restored.shape == tensor.shape
        assert restored.name == tensor.name
        assert SymboSerializer.verify_equivalence(tensor, restored)

    def test_wrong_input_type(self):
        with pytest.raises(SerializationError):
            SymboSerializer.serialize_tensor([[1, 2], [3, 4]], "msgpack")


class TestPolicyFunctions:
    @pytest.fixture()
    def policy(self):
        te = TaylorExpansion([x, y], {x: 0.0, y: 0.0}, max_order=2)
        te.generate("g")
        return te.to_policy_function({n: float(i + 1)
                                     for i, n in enumerate(te.coefficient_names)})

    @pytest.mark.parametrize("fmt", ["msgpack", "json"])
    def test_round_trip_keeps_evaluation_identical(self, policy, fmt):
        if fmt == "msgpack":
            pytest.importorskip("msgpack")
        payload = SymboSerializer.serialize_policy_function(policy, fmt)
        restored = SymboSerializer.deserialize_policy_function(payload, fmt)
        assert restored(x=0.4, y=-0.25) == pytest.approx(policy(x=0.4, y=-0.25))
        assert restored.coefficients.keys() == policy.coefficients.keys()

    def test_legacy_key_format_is_still_readable(self):
        legacy = {
            "type": "PolicyFunction",
            "expansion": "g_0 + g_x*x + g_y*y",
            "variables": ["x", "y"],
            "center": {"x": 0.0, "y": 0.0},
            "coefficients": {"()": "g_0", "(x,)": "g_x", "(y,)": "g_y"},
            "coeff_values": {"g_0": 1.0, "g_x": 2.0, "g_y": 3.0},
        }
        restored = SymboSerializer.deserialize_policy_function(
            json.dumps(legacy).encode("utf-8"), "json")
        assert restored(x=0.5, y=-1.0) == pytest.approx(1.0 + 2.0 * 0.5 - 3.0)

    def test_other_objects_are_refused(self):
        with pytest.raises(SerializationError, match="Expected PolicyFunction"):
            SymboSerializer.serialize_policy_function(x + 1)


class TestGroebnerState:
    @pytest.mark.parametrize("fmt", ["msgpack", "json"])
    def test_round_trip(self, fmt):
        if fmt == "msgpack":
            pytest.importorskip("msgpack")
        state = GröbnerBasisState([x**2 + y**2 - 1, x - y], [x, y], "lex")
        state.status = "completed"
        state.solutions = [{x: sp.Integer(1), y: sp.Integer(1)}]
        restored = SymboSerializer.deserialize_groebner_state(
            SymboSerializer.serialize_groebner_state(state, fmt), fmt)
        assert restored.order == "lex"
        assert restored.status == "completed"
        assert str(restored.polynomials[1]) == "x - y"
        assert restored.solutions[0][x] == 1
        # the basis object itself is not portable and must be recomputed
        assert restored.basis is None

    def test_other_objects_are_refused(self):
        with pytest.raises(SerializationError, match="GröbnerBasisState"):
            SymboSerializer.serialize_groebner_state({"a": 1})


class TestHelpers:
    def test_verify_equivalence(self):
        assert SymboSerializer.verify_equivalence(x + x, 2 * x) is True
        assert SymboSerializer.verify_equivalence(x, x + 1) is False
        assert SymboSerializer.verify_equivalence(x, "x") is False

    def test_round_trip_test_reports_failure_instead_of_raising(self, caplog):
        import logging

        def broken(_obj):
            raise RuntimeError("boom")

        with caplog.at_level(logging.WARNING, logger="symbo.io.serialization"):
            assert SymboSerializer.round_trip_test(x, broken, lambda d: d) is False
        assert any("round-trip" in r.getMessage().lower() for r in caplog.records)

    def test_self_check_passes_for_every_available_backend(self):
        results = verify_round_trip_serialization()
        assert results, "the self-check found nothing to check"
        assert all(results.values()), results
        if pytest.importorskip("msgpack", reason="msgpack backend optional"):
            assert results["Expression"] is True

    def test_arrow_table_from_solution_set(self):
        pa = pytest.importorskip("pyarrow")
        from symbo.io.serialization import ArrowTableBuilder

        table = ArrowTableBuilder.from_solution_set(
            [{x: 1.0, y: -1.0}, {x: 2.5, y: 0.5}])
        assert isinstance(table, pa.Table)
        assert table.num_rows == 2
        assert set(table.column_names) >= {"x", "y"}
