# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Atomic primitives: the semantics every higher-level module is built on."""

import numpy as np
import pytest
import sympy as sp

from symbo.primitives import AtomicPrimitives as P
from symbo.primitives import add, diff, mul

x, y, z = sp.symbols("x y z")


class TestAlgebra:
    def test_symbolic_add_simplifies(self):
        assert P.symbolic_add(x + y, 2 * x) == 3 * x + y
        assert P.symbolic_add(1, 2.0) == 3.0

    def test_symbolic_mul_expands(self):
        assert P.symbolic_mul(x, y) == x * y
        assert P.symbolic_mul(x * (x + 1), x) == x**2 * (x + 1)

    def test_symbolic_pow(self):
        assert P.symbolic_pow(x, 2) == x**2
        assert P.symbolic_pow(x, 0) == 1
        assert P.symbolic_pow(x, -1) == 1 / x

    def test_symbolic_div_cancels(self):
        assert P.symbolic_div(x**2, x) == x
        assert P.symbolic_div(x**2, sp.Integer(2)) == x**2 / 2

    def test_division_by_zero_is_rejected(self):
        with pytest.raises(ValueError, match="Division by zero"):
            P.symbolic_div(x, 0)
        with pytest.raises(ValueError, match="Division by zero"):
            P.symbolic_div(x, 0.0)

    def test_module_aliases_match_the_class(self):
        assert add(x, y) == P.symbolic_add(x, y)
        assert mul(x, y) == P.symbolic_mul(x, y)
        assert diff(x**3, x) == P.symbolic_diff(x**3, x)


class TestCalculus:
    def test_symbolic_diff_orders(self):
        assert P.symbolic_diff(x**3, x) == 3 * x**2
        assert P.symbolic_diff(x**3, x, order=2) == 6 * x
        assert P.symbolic_diff(x**3, x, order=0) == x**3
        assert P.symbolic_diff(sp.sin(x) * y, y) == sp.sin(x)

    def test_diff_of_a_constant_is_zero(self):
        assert P.symbolic_diff(sp.Integer(5), x) == 0

    def test_gradient_component_order_follows_the_variable_list(self):
        assert P.gradient(x**2 + y**2, [x, y]) == [2 * x, 2 * y]
        assert P.gradient(x**2 + y**2, [y, x]) == [2 * y, 2 * x]

    def test_hessian_is_symmetric(self):
        H = P.hessian(x**2 * y + y**3, [x, y])
        assert H.tolist() == [[2 * y, 2 * x], [2 * x, 6 * y]]
        assert H[0, 1] == H[1, 0]

    def test_jacobian_rows_are_equations(self):
        J = P.jacobian([x * y, x + y], [x, y])
        assert J.tolist() == [[y, x], [1, 1]]

    def test_integrate_indefinite_and_definite(self):
        assert P.symbolic_integrate(x**2, x) == x**3 / 3
        assert P.symbolic_integrate(x**2, x, 0, 1) == sp.Rational(1, 3)

    def test_substitute_mixing_symbol_and_string_keys(self):
        assert P.substitute(x + y, {x: 1}) == y + 1
        assert P.substitute(x + y, {x: y}) == 2 * y

    def test_evaluate_numeric(self):
        assert P.evaluate_numeric(x * 2, {x: 2.0}) == pytest.approx(4.0)
        assert P.evaluate_numeric(x * 2, {"x": 2.0}) == pytest.approx(4.0)

    def test_evaluate_numeric_rejects_unresolved_symbols(self):
        with pytest.raises(ValueError, match=r"unresolved symbols \['y'\]"):
            P.evaluate_numeric(x + y, {x: 2.0})

    def test_evaluate_numeric_allows_constant_expressions(self):
        assert P.evaluate_numeric(sp.pi, {}) == pytest.approx(float(sp.pi))


class TestTensors:
    def test_contraction_matches_matmul(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        assert np.array_equal(P.tensor_contraction(A, B, [1], [0]), A @ B)

    def test_contraction_is_symbolic_when_the_inputs_are(self):
        A = np.array([[x, 1]], dtype=object)
        B = np.array([[y], [x]], dtype=object)
        result = P.tensor_contraction(A, B, [1], [0])
        assert result.shape == (1, 1)
        assert sp.simplify(result[0, 0] - (x * y + x)) == 0

    def test_full_contraction_gives_a_scalar(self):
        v = np.array([1.0, 2.0])
        assert float(P.tensor_contraction(v, v, [0], [0])) == pytest.approx(5.0)

    def test_outer_product(self):
        assert np.array_equal(P.outer_product(np.array([1, 2]), np.array([3, 4])),
                              np.array([[3, 4], [6, 8]]))

    def test_trace_over_arbitrary_axes(self):
        T = np.ones((2, 2, 2))
        assert np.array_equal(P.tensor_trace(T, 0, 1), np.array([2.0, 2.0]))
        M = np.array([[1, 2], [3, 4]])
        assert float(P.tensor_trace(M)) == pytest.approx(5.0)

    def test_symbolic_tensor_product_is_the_kronecker_product(self):
        A = sp.Matrix([[x]])
        B = sp.Matrix([[y, 1]])
        assert P.symbolic_tensor_product(A, B) == sp.Matrix([[x * y, x]])


class TestPolynomial:
    def test_expand_and_factor_are_inverse(self):
        assert P.polynomial_expand((x + 1) ** 3) == x**3 + 3 * x**2 + 3 * x + 1
        assert P.polynomial_factor(x**2 - 1) == (x - 1) * (x + 1)
        assert sp.expand(P.polynomial_factor(x**2 - 1)) == P.polynomial_expand(x**2 - 1)

    def test_collect_groups_by_the_variable(self):
        assert P.polynomial_collect(x**2 + x * y + x + 2, x) == x**2 + x * (y + 1) + 2

    def test_degree_in_the_given_variable(self):
        assert P.polynomial_degree(x**3 + y * x**5, x) == 5
        assert P.polynomial_degree(x**3 + y * x**5, y) == 1
        # SymPy's convention for the zero polynomial
        assert P.polynomial_degree(sp.Integer(0), x) == -sp.oo

    def test_coeffs_are_padded_from_the_leading_power(self):
        assert P.polynomial_coeffs(2 * x**2 + 3, x) == [2, 0, 3]
        assert P.polynomial_coeffs(x + 1, x) == [1, 1]


class TestMatrix:
    def test_det(self):
        assert P.matrix_det(sp.Matrix([[x, 1], [y, x]])) == x**2 - y

    def test_inv_times_original_is_identity(self):
        M = sp.Matrix([[x, 0], [0, 1]])
        assert sp.simplify(P.matrix_inv(M) * M - sp.eye(2)) == sp.zeros(2, 2)

    def test_singular_matrix_is_rejected(self):
        with pytest.raises(ValueError, match="singular"):
            P.matrix_inv(sp.Matrix([[x, x], [1, 1]]))

    def test_eigenvalues_and_multiplicities(self):
        assert P.matrix_eigenvals(sp.Matrix([[2, 0], [0, 3]])) == {2: 1, 3: 1}

    def test_eigenvectors(self):
        value, multiplicity, vectors = P.matrix_eigenvects(sp.Matrix([[2, 0], [0, 2]]))[0]
        assert value == 2
        assert multiplicity == 2
        assert len(vectors) == 2


class TestSimplification:
    def test_simplify(self):
        assert P.simplify(x + x) == 2 * x
        assert P.simplify((x**2 - 1) / (x - 1)) == x + 1

    def test_trigsimp(self):
        assert P.trigsimp(sp.sin(x) ** 2 + sp.cos(x) ** 2) == 1

    def test_ratsimp_and_cancel(self):
        assert P.ratsimp(x / y + y / x) == (x**2 + y**2) / (x * y)
        assert P.cancel((x**2 + 2 * x + 1) / (x + 1)) == x + 1

    def test_primitives_compose(self):
        """The advertised composability: build a gradient, contract it, evaluate."""
        f = x**2 * y + sp.sin(z)
        grad = P.gradient(f, [x, y, z])
        assert P.evaluate_numeric(sum(g * s for g, s in zip(grad, (x, y, z), strict=True))
                                 .subs({z: 0.0}), {x: 1.0, y: 2.0}) == pytest.approx(6.0)
