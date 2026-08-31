# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Symbolic Taylor expansion, policy functions and their WASM interchange."""

import json

import numpy as np
import pytest
import sympy as sp

from symbo.generative.taylor import (
    TaylorExpansion,
    generate_multivariate_taylor,
)


@pytest.fixture()
def expansion():
    x, y = sp.symbols("x y")
    te = TaylorExpansion([x, y], {x: 0.0, y: 0.0}, max_order=2)
    return te, x, y


class TestTaylorExpansion:
    def test_coefficient_naming_scheme(self, expansion):
        te, x, y = expansion
        expr = te.generate("g")
        assert te.coefficient_names == ['g_0', 'g_x', 'g_y',
                                        'g_x_x', 'g_y_y', 'g_x_y']
        assert expr.free_symbols == ({sp.Symbol(n) for n in te.coefficient_names}
                                     | {x, y})
        # 2nd-order diagonal terms carry the 1/2! Taylor factor, cross terms do not
        assert expr.coeff(sp.Symbol("g_x_x")) == sp.Rational(1, 2) * x**2
        assert expr.coeff(sp.Symbol("g_x_y")) == x * y

    def test_max_order_must_be_positive(self, expansion):
        te, _, _ = expansion
        for bad in (0, -3):
            with pytest.raises(ValueError):
                TaylorExpansion(te.variables, te.center, max_order=bad)

    def test_unknown_variable_in_center_is_rejected(self):
        x, z = sp.symbols("x z")
        with pytest.raises(ValueError, match="z"):
            TaylorExpansion([x], {x: 0.0, z: 1.0}, max_order=1)

    def test_evaluate_at_point_is_linear_part(self, expansion):
        te, x, y = expansion
        te.generate("g")
        values = dict(zip(te.coefficient_names, [0.0, 2.0, -1.0, 0.0, 0.0, 0.0],
                          strict=True))
        expr = te.substitute_coefficients(values, apply=True)
        assert expr.free_symbols == {x, y}  # only the coefficients are numeric
        point = {x: 0.5, y: 2.0}
        assert float(te.evaluate_at_point(point)) == pytest.approx(2.0 * 0.5 - 1.0 * 2.0)

    def test_substitute_coefficients_returns_expr_without_applying(self, expansion):
        te, x, _y = expansion
        te.generate("g")
        substituted = te.substitute_coefficients({"g_x": 5.0})
        assert substituted.has(x) and not substituted.has(sp.Symbol("g_x"))
        # the expansion itself stays symbolic unless apply=True
        assert te.expansion.has(sp.Symbol("g_x"))
        # ... but the fitted value is recorded and reused
        assert te.coeff_values["g_x"] == 5.0
        policy = te.to_policy_function()
        assert policy(x=1.0, y=0.0) == pytest.approx(5.0)

    def test_substitute_coefficients_rejects_unknown_name(self, expansion):
        te, _, _ = expansion
        te.generate("g")
        with pytest.raises(ValueError, match="g_xx"):
            te.substitute_coefficients({"g_xx": 1.0})

    def test_get_coefficient_vector_matches_names(self, expansion):
        te, _, _ = expansion
        te.generate("f")
        assert [c.name for c in te.get_coefficient_vector()] == te.coefficient_names

    def test_generate_multivariate_taylor_matches_manual_second_order(self):
        x = sp.Symbol("x")
        expr = generate_multivariate_taylor(sp.exp(x), [x], {x: 0.0}, max_order=2)
        # exp(x) ~ 1 + x + x^2/2, with symbolic coefficients substituted by numbers
        assert isinstance(expr, sp.Expr)
        # numeric behaviour: compare the expansion of a quadratic polynomial,
        # which is exact at order >= 2
        poly = 3 + 2 * x + 5 * x**2
        exact = generate_multivariate_taylor(poly, [x], {x: 1.0}, max_order=3)
        assert sp.simplify(exact - poly.subs(x, 1 + (x - 1))) == 0

    def test_wasm_json_roundtrip_preserves_coefficients(self, expansion):
        te, _x, _y = expansion
        te.generate("p")
        values = {n: float(i + 1) for i, n in enumerate(te.coefficient_names)}
        te.substitute_coefficients(values)
        payload = json.loads(te.to_wasm_json())
        assert payload["max_order"] == 2
        assert payload["coefficients"] == te.coefficient_names
        # coefficient_map records which variables each coefficient multiplies ...
        assert payload["coefficient_map"]["p_x"] == ["x"]
        assert payload["coefficient_map"]["p_0"] == []
        # ... while coefficient_values carries the numbers a VM needs
        assert payload["coefficient_values"]["p_x"] == pytest.approx(2.0)

        restored = TaylorExpansion.from_wasm_json(json.dumps(payload))
        assert restored.coefficient_names == te.coefficient_names
        assert restored.coeff_values == te.coeff_values
        assert float(restored.to_policy_function()(x=0.0, y=0.0)) == pytest.approx(values["p_0"])


class TestPolicyFunction:
    def test_call_with_numeric_coefficients(self):
        x, y = sp.symbols("x y")
        te = TaylorExpansion([x, y], {x: 0.0, y: 0.0}, max_order=1)
        te.generate("g")
        policy = te.to_policy_function({"g_0": 1.0, "g_x": 2.0, "g_y": 3.0})
        assert policy(x=1.0, y=0.0) == pytest.approx(3.0)
        assert policy(x=0.0, y=1.0) == pytest.approx(4.0)
        assert policy(x=[1.0, 2.0], y=[0.0, 0.0]) == pytest.approx([3.0, 5.0])

    def test_missing_coefficient_names_are_listed(self):
        x = sp.Symbol("x")
        te = TaylorExpansion([x], {x: 0.0}, max_order=1)
        te.generate("g")
        policy = te.to_policy_function()
        with pytest.raises(ValueError, match="g_x"):
            policy(x=1.0, g_x=1.0, g_xx=2.0)

    def test_unspecified_coefficients_default_to_zero(self):
        x = sp.Symbol("x")
        te = TaylorExpansion([x], {x: 0.0}, max_order=1)
        te.generate("g")
        policy = te.to_policy_function({"g_x": 4.0})
        assert policy(x=2.0) == pytest.approx(8.0)

    def test_update_coefficients(self):
        x = sp.Symbol("x")
        te = TaylorExpansion([x], {x: 0.0}, max_order=1)
        te.generate("g")
        policy = te.to_policy_function({"g_0": 0.0, "g_x": 1.0})
        assert policy(x=3.0) == pytest.approx(3.0)
        policy.update_coefficients({"g_x": 10.0})
        assert policy(x=3.0) == pytest.approx(30.0)

    def test_get_partial_derivative(self):
        x, y = sp.symbols("x y")
        te = TaylorExpansion([x, y], {x: 0.0, y: 0.0}, max_order=2)
        te.generate("g")
        policy = te.to_policy_function()
        d = policy.get_partial_derivative(x)
        assert d.has(sp.Symbol("g_x")) and not d.has(sp.Symbol("g_y_y"))

    def test_repr_mentions_center(self):
        x = sp.Symbol("x")
        te = TaylorExpansion([x], {x: 1.0}, max_order=1)
        te.generate("g")
        policy = te.to_policy_function({"g_0": 0.0, "g_x": 1.0})
        assert "x" in repr(policy)


class TestNumericalConsistency:
    def test_second_order_matches_finite_difference(self):
        """The order-2 expansion of a smooth scalar function must reproduce f."""
        x = sp.Symbol("x")
        f = sp.sin(x) + x**2
        center = {x: 0.7}
        te = TaylorExpansion([x], center, max_order=3)
        te.generate("c")
        # substitute each symbolic coefficient with the true derivative value
        # The expansion already divides by k! (see c_x_x * dx**2 / 2), so the
        # coefficient symbols are the raw derivatives at the center.
        true = {}
        for order, name in enumerate(["c_0", "c_x", "c_x_x", "c_x_x_x"]):
            true[name] = float(sp.diff(f, x, order).subs(x, center[x]))
        te.substitute_coefficients(true, apply=True)
        for delta in (-0.05, 0.03):
            got = float(te.evaluate_at_point({x: center[x] + delta}))
            assert got == pytest.approx(float(f.subs(x, center[x] + delta)), abs=1e-6)

    def test_batched_policy_evaluation_shape(self):
        x, y = sp.symbols("x y")
        te = TaylorExpansion([x, y], {x: 0.0, y: 0.0}, max_order=1)
        te.generate("g")
        policy = te.to_policy_function({"g_0": 0.5, "g_x": 1.0, "g_y": -2.0})
        out = np.asarray(policy(x=[0.0, 1.0, 2.0], y=[0.0, 0.0, 0.0]))
        np.testing.assert_allclose(out, [0.5, 1.5, 2.5])
