# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Shared pytest fixtures for the Symbo test suite."""

import pytest
import sympy as sp


@pytest.fixture()
def symbols():
    """The usual (x, y, z) symbol triple."""
    return sp.symbols('x y z')


@pytest.fixture()
def rbc_model():
    """
    A small complete RBC model, ready for perturbation.

    Returns ``(equations, state_vars, control_vars, shock_vars, params, ss)``.
    Technology enters as ``exp(a)`` so the shock has the mean-zero
    log-deviation interpretation the perturbation solver assumes.
    """
    k, c, a = sp.symbols('k c a')
    kp, cp, ap = sp.symbols('k_next c_next a_next')
    alpha, beta, delta, rho = sp.symbols('alpha beta delta rho')
    params = {alpha: 0.36, beta: 0.99, delta: 0.08, rho: 0.9}

    euler = c**(-1) - beta * cp**(-1) * (alpha * sp.exp(ap) * kp**(alpha - 1) + 1 - delta)
    resource = alpha * sp.exp(a) * k**alpha + (1 - delta) * k - c - kp
    shock_law = ap - rho * a

    return [euler, resource, shock_law], [k], [c], [a], params, {'rho': rho}


@pytest.fixture()
def tiny_tensor():
    """A (1,) NanoTensor holding ``2*x`` over base variable ``x``."""
    from symbo import NanoTensor

    x = sp.Symbol('x')
    nt = NanoTensor((1,), max_order=1, base_vars=['x'])
    nt.data[0] = 2 * x
    return nt, x
