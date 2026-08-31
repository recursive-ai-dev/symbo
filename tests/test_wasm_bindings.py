# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""The browser-facing boundary: WASM interface plus the MessagePack codecs."""

import json

import numpy as np
import pytest
import sympy as sp

from symbo.nanotensor import serialize_basis_arrow, serialize_basis_msgpack
from symbo.wasm_bindings import (
    MessagePackSerializer,
    WASMCompatibilityError,
    WASMInterface,
    create_browser_test_payload,
    verify_wasm_compatibility,
)

x, y = sp.symbols("x y")


class TestWASMInterface:
    def test_eval_expression(self):
        assert WASMInterface.eval_expression("x**2 + y", {"x": 2.0, "y": 3.0}) == pytest.approx(7.0)

    def test_eval_expression_rejects_unresolved_symbols(self):
        with pytest.raises(WASMCompatibilityError):
            WASMInterface.eval_expression("x + z", {"x": 1.0})

    @pytest.mark.parametrize("payload", ["eval('1')", "__import__('os').system('id')",
                                         "open('/tmp/nope','w')", "x.__class__", "y[0]"])
    def test_injection_is_rejected(self, payload):
        with pytest.raises(WASMCompatibilityError):
            WASMInterface.eval_expression(payload, {"x": 1.0, "y": 1.0})

    def test_differentiate_returns_a_string(self):
        assert WASMInterface.differentiate("x**3 + x", "x") == str(3 * x**2 + 1)
        assert WASMInterface.differentiate("sin(x)*y", "y") == str(sp.sin(x))
        assert WASMInterface.differentiate("x**4", "x", order=2) == str(12 * x**2)

    def test_simplify(self):
        assert WASMInterface.simplify("(x+1)**2 - x**2 - 2*x - 1") == "0"

    def test_solve_equation(self):
        assert sorted(WASMInterface.solve_equation("x**2 - 4", "x")) == ["-2", "2"]

    def test_expand_taylor(self):
        result = WASMInterface.expand_taylor("exp(x)", "x", 0.0, 3)
        assert result["order"] == 3
        # 1 + x + x^2/2 + x^3/6
        assert result["coefficients"][0] == pytest.approx(1.0)
        assert result["coefficients"][2] == pytest.approx(0.5)
        assert result["coefficients"][3] == pytest.approx(1 / 6)

    def test_compute_jacobian(self):
        jac = WASMInterface.compute_jacobian(["x**2 + y", "x - y**2"], ["x", "y"])
        assert jac == [[str(2 * x), "1"], ["1", str(-2 * y)]]

    @pytest.mark.parametrize("bad", ["x**", "x +* 1", "'not an expression' + 1"])
    def test_malformed_input_is_wrapped_in_a_compatibility_error(self, bad):
        with pytest.raises(WASMCompatibilityError):
            WASMInterface.differentiate(bad, "x")


class TestMessagePack:
    """The msgpack codecs degrade to a clear ImportError when not installed."""

    def test_check_available_is_a_guard(self):
        if pytest.importorskip("msgpack", reason="msgpack is optional"):
            assert MessagePackSerializer.check_available() is None

    def test_missing_backend_is_reported_with_install_hint(self, monkeypatch):
        import symbo.wasm_bindings as wb

        monkeypatch.setattr(wb, "msgpack", None)
        with pytest.raises(ImportError, match="pip install msgpack"):
            wb.MessagePackSerializer.serialize_expression(x + 1)

    def test_expression_round_trip(self):
        pytest.importorskip("msgpack")
        expr = sp.sin(x) + x * y**2
        packed = MessagePackSerializer.serialize_expression(expr)
        restored = MessagePackSerializer.deserialize_expression(packed)
        assert sp.simplify(restored - expr) == 0

    def test_tensor_round_trip(self):
        pytest.importorskip("msgpack")
        data = np.array([x + 1, y - 2], dtype=object)
        tensor_data, shape = MessagePackSerializer.deserialize_tensor(
            MessagePackSerializer.serialize_tensor(data, (2,)))
        assert shape == (2,)
        assert sp.simplify(tensor_data[0] - (x + 1)) == 0
        assert sp.simplify(tensor_data[1] - (y - 2)) == 0

    def test_solution_round_trip(self):
        msgpack = pytest.importorskip("msgpack")
        payload = MessagePackSerializer.serialize_solution(
            {"g_k_a": 1.5, "exact": sp.Rational(1, 2), "nested": {x: sp.pi}})
        assert msgpack.unpackb(payload, raw=False) == {
            "g_k_a": 1.5, "exact": "1/2", "nested": {"x": "pi"}}


@pytest.fixture()
def basis():
    return sp.groebner([x + y - 2, x**2 + y**2 - 2], x, y, order="lex")


class TestBasisCodecs:
    def test_msgpack_basis_payload(self, basis):
        msgpack = pytest.importorskip("msgpack")
        G = basis
        unpacked = msgpack.unpackb(serialize_basis_msgpack(G), raw=False)
        assert unpacked["gens"] == ["x", "y"]
        assert unpacked["order"] == "lex"
        assert [str(p) for p in G.polys] == unpacked["polys"]

    def test_arrow_basis_is_an_arrow_stream(self, basis):
        pa = pytest.importorskip("pyarrow")
        G = basis
        payload = serialize_basis_arrow(G)
        assert isinstance(payload, bytes) and len(payload) > 0
        with pa.BufferReader(payload) as reader:
            table = pa.ipc.open_stream(reader).read_all()
        assert table.column_names == ["poly", "gens"]
        assert table.num_rows == len(G.polys)

    def test_missing_backends_raise_runtime_error(self, basis, monkeypatch):
        import symbo.nanotensor as nt

        G = basis
        monkeypatch.setattr(nt, "msgpack", None)
        with pytest.raises(RuntimeError, match="msgpack"):
            serialize_basis_msgpack(G)
        monkeypatch.setattr(nt, "msgpack", object())
        monkeypatch.setattr(nt, "pa", None)
        with pytest.raises(RuntimeError, match="pyarrow"):
            serialize_basis_arrow(G)


class TestCompatibilityHelpers:
    def test_payload_is_json_serialisable(self):
        payload = create_browser_test_payload()
        assert json.dumps(payload)
        assert verify_wasm_compatibility(payload) is True

    def test_verify_rejects_non_serialisable_objects(self):
        assert verify_wasm_compatibility({"f": lambda v: v}) is False
        assert verify_wasm_compatibility(np.ones(3)) is False
        assert verify_wasm_compatibility([1, 2.5, "x", None, True]) is True
