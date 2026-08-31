# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""SymbolicTensor: construction, Einstein-style ops and numeric evaluation."""

import numpy as np
import pytest
import sympy as sp

from symbo import SymbolicTensor


@pytest.fixture()
def pair():
    a = SymbolicTensor((2, 3), name="a")
    a.fill_with_symbols("a")
    b = SymbolicTensor((3, 4), name="b")
    b.fill_with_symbols("b")
    return a, b


class TestConstruction:
    def test_int_shape_and_defaults(self):
        t = SymbolicTensor(4, name="v")
        assert t.shape == (4,) and t.rank == 1 and t.size == 4
        assert t.data[0] == 0

    @pytest.mark.parametrize("bad", [(), (0,), (-2,), (2.5, 'x')])
    def test_invalid_shapes(self, bad):
        with pytest.raises((ValueError, TypeError)):
            SymbolicTensor(bad)

    def test_fill_with_symbols_names_by_index(self):
        t = SymbolicTensor((2, 2), name="M").fill_with_symbols("m")
        assert t.get_element((1, 0)) == sp.Symbol("m_1_0")
        assert len(t.free_symbols) == 4

    def test_fill_with_expression_broadcasts(self):
        x = sp.Symbol("x")
        t = SymbolicTensor((2, 3)).fill_with_expression(x**2)
        assert all(t.get_element((i, j)) == x**2 for i in range(2) for j in range(3))

    def test_set_element_returns_self_for_chaining(self):
        t = SymbolicTensor((2,))
        assert t.set_element((1,), sp.Symbol("q")) is t
        assert t.get_element((1,)) == sp.Symbol("q")

    def test_out_of_range_indices(self):
        t = SymbolicTensor((2, 2))
        with pytest.raises(IndexError):
            t.get_element((2, 0))


class TestAlgebra:
    def test_elementwise_ops(self, pair):
        a, _ = pair
        b = SymbolicTensor((2, 3)).fill_with_expression(sp.Symbol("u"))
        assert (a + b).get_element((0, 0)) == a.get_element((0, 0)) + sp.Symbol("u")
        assert (a - a).get_element((0, 0)) == 0
        assert (a * 2).get_element((1, 2)) == 2 * a.get_element((1, 2))
        assert (b / 2).get_element((0, 0)) == sp.Symbol("u") / 2

    def test_shape_mismatch_rejected(self, pair):
        a, b = pair
        with pytest.raises(ValueError, match="Shape mismatch"):
            a + b

    def test_method_aliases_match_operators(self, pair):
        """The documented method spellings must be the operators, not new maths."""
        a, _ = pair
        for name, operator in (("add", a + a), ("sub", a - a),
                               ("mul", a * a), ("div", a / a)):
            method = getattr(a, name)(a)
            assert method.shape == operator.shape
            assert sp.simplify(method.to_matrix() - operator.to_matrix()).is_zero_matrix

    def test_matmul_matches_matrix_product(self, pair):
        a, b = pair
        assert a.matmul(b).shape == (2, 4)
        assert sp.simplify(a.matmul(b).to_matrix() - a.to_matrix() * b.to_matrix()) == sp.zeros(2, 4)
        assert sp.simplify((a @ b).to_matrix() - a.matmul(b).to_matrix()) == sp.zeros(2, 4)

    def test_matmul_validates_ranks_and_shapes(self, pair):
        _a, b = pair
        with pytest.raises(ValueError, match="rank"):
            SymbolicTensor((3,)) @ b
        with pytest.raises(ValueError, match="Shape mismatch for @"):
            SymbolicTensor((2, 3)) @ SymbolicTensor((4, 2))

    def test_outer_product_shape(self, pair):
        a, _ = pair
        v = SymbolicTensor((2,)).fill_with_symbols("v")
        assert a.outer(v).shape == (2, 3, 2)

    def test_contract_is_order_independent(self):
        t = SymbolicTensor((2, 3, 4)).fill_with_symbols("t")
        u = SymbolicTensor((4, 3)).fill_with_symbols("u")
        left = t.contract(u, (2, 1), (0, 1))
        right = t.contract(u, (1, 2), (1, 0))
        assert left.shape == (2,)
        for i in range(2):
            assert sp.simplify(left.get_element((i,)) - right.get_element((i,))) == 0

    def test_contract_matches_numeric_einsum(self):
        e = SymbolicTensor((3, 3)).fill_with_symbols("e")
        f = SymbolicTensor((3, 3)).fill_with_symbols("f")
        ev = np.array([[float(i * 3 + j + 1) for j in range(3)] for i in range(3)])
        fv = np.array([[float(i + j + 2) for j in range(3)] for i in range(3)])
        point = {**{f"e_{i}_{j}": ev[i, j] for i in range(3) for j in range(3)},
                 **{f"f_{i}_{j}": fv[i, j] for i in range(3) for j in range(3)}}
        got = e.contract(f, (0, 1), (1, 0)).eval_numeric(point)
        np.testing.assert_allclose(got[0], np.einsum("ij,ji->", ev, fv))

    def test_contract_rejects_bad_axes(self):
        e = SymbolicTensor((2, 3)).fill_with_symbols("e")
        f = SymbolicTensor((2, 3)).fill_with_symbols("f")
        with pytest.raises(ValueError, match="same number of axes"):
            e.contract(f, (0,), (0, 1))
        with pytest.raises(ValueError, match="same dimension"):
            e.contract(f, (0,), (1,))
        with pytest.raises(ValueError, match="same axis twice"):
            e.contract(f, (0, 0), (0, 1))

    def test_negative_axes_are_normalised(self):
        a, b = SymbolicTensor((2, 3)).fill_with_symbols("a"), SymbolicTensor((3, 4)).fill_with_symbols("b")
        assert sp.simplify(a.contract(b, (-1,), (0,)).to_matrix()
                           - a.to_matrix() * b.to_matrix()) == sp.zeros(2, 4)

    def test_trace(self):
        m = SymbolicTensor((3, 3)).fill_with_symbols("m")
        tr = m.trace(0, 1)
        expected = sum(sp.Symbol(f"m_{i}_{i}") for i in range(3))
        assert sp.simplify(tr.get_element((0,)) - expected) == 0

    def test_transpose(self):
        t = SymbolicTensor((2, 3)).fill_with_symbols("t")
        tt = t.transpose()
        assert tt.shape == (3, 2)
        assert tt.get_element((2, 1)) == t.get_element((1, 2))

    def test_matrix_roundtrip(self):
        m = sp.Matrix([[sp.Symbol("p"), 1], [2, sp.Symbol("q")]])
        t = SymbolicTensor.from_matrix(m, name="M")
        assert t.shape == (2, 2)
        assert (t.to_matrix() - m) == sp.zeros(2, 2)

    def test_simplify_and_diff_and_subs_return_new_tensors(self):
        x = sp.Symbol("x")
        t = SymbolicTensor((2,)).fill_with_expression(sp.simplify((x + 1) ** 2 - x**2 - 2 * x))
        assert t.simplify().get_element((0,)) == 1
        d = t.diff(x)
        assert d is not t
        s = t.subs({x: 2})
        assert s is not t and s.get_element((0,)) == t.get_element((0,))

    def test_diff_order(self):
        x = sp.Symbol("x")
        t = SymbolicTensor((1,)).fill_with_expression(x**4)
        assert t.diff(x, 2).get_element((0,)) == 12 * x**2


class TestEvaluation:
    def test_eval_numeric_matches_substitution(self):
        x, y = sp.symbols("x y")
        t = SymbolicTensor((2, 1)).fill_with_expression(x + 2 * y)
        got = t.eval_numeric({"x": 1.5, "y": -0.5})
        assert got.shape == (2, 1)
        np.testing.assert_allclose(got, [[0.5], [0.5]])

    def test_eval_numeric_accepts_symbol_keys(self):
        x = sp.Symbol("x")
        t = SymbolicTensor((1,)).fill_with_expression(3 * x)
        np.testing.assert_allclose(t.eval_numeric({x: 2.0}), [6.0])

    def test_eval_numeric_raises_on_unresolved_symbols(self):
        x, z = sp.symbols("x z")
        t = SymbolicTensor((1,)).fill_with_expression(x + z)
        with pytest.raises(ValueError, match="z"):
            t.eval_numeric({"x": 1.0})

    def test_eval_numeric_accepts_plain_numbers(self):
        t = SymbolicTensor((2,))
        t.data[0] = 2.5
        t.data[1] = sp.Integer(4)
        np.testing.assert_allclose(t.eval_numeric({}), [2.5, 4.0])

    def test_to_nanotensor_shares_nothing_but_agrees(self):
        x, y = sp.symbols("x y")
        t = SymbolicTensor((2,)).fill_with_expression(x**2 + y)
        nt = t.to_nanotensor(max_order=2)
        assert nt.shape == (2,)
        assert nt is not t
        point = {"x": 1.5, "y": -2.0}
        np.testing.assert_allclose(nt.eval_numeric(point), t.eval_numeric(point))
        # the conversion copies data
        nt.data[0] = sp.S(0)
        assert t.get_element((0,)) == x**2 + y
