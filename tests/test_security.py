# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Input-validation / sandbox tests for :func:`symbo.security.safe_sympify`.

``safe_sympify`` is the guard in front of every string the library parses
(WASM payloads, REPL input, ecosystem transforms), so these cases are the
security boundary of the package. The regression that matters most is nested
re-entry: SymPy exports ``sympify``, and the parser leaves calls to it in place,
so ``sympify("<code>")`` used to execute the inner string with full builtins.
"""

import os

import pytest
import sympy as sp

from symbo.security import MAX_EXPRESSION_LENGTH, safe_sympify

LEGITIMATE = [
    "x**2 + 3*x - 1",
    "sin(x) + cos(y)",
    "exp(-x**2/2)/sqrt(2*pi)",
    "log(x, 2)",
    "Piecewise((x, x > 0), (0, True))",
    "Derivative(f(x), x)",
    "Sum(n**2, (n, 1, 10))",
    "Integral(x**2, (x, 0, 1))",
    "sqrt(a**2 + b**2)",
    "Matrix([[1, 2], [3, 4]])",
    "1.2e-3 * g_k",
    "Rational(1, 3) + S(1)/6",
]

MALICIOUS = [
    "sympify(\"open('/tmp/symbo_security_marker','w').write('x')\")",
    'lambdify("x", "open(\'/tmp/symbo_security_marker\',\'w\').write(\'x\')")',
    "autowrap(x)",
    "dotprint(x)",
    "test()",
    "init_printing()",
    "lambda: open('/tmp/symbo_security_marker', 'w')",
    "x.__class__",
    "().__class__.__bases__",
    "os.system('id')",
    "eval('1+1')",
    "__import__('os')",
    "a[0]",
    "exec('x=1')",
    "parse_expr('open')",
    "Function('eval')('1+1')",
    "getattr(x, 'y')",
]


class TestLegitimateExpressions:
    @pytest.mark.parametrize("expr_str", LEGITIMATE)
    def test_accepted_and_parsed(self, expr_str):
        result = safe_sympify(expr_str)
        assert isinstance(result, (sp.Basic, sp.MatrixBase))

    def test_symbols_are_created_automatically(self):
        expr = safe_sympify("theta * kap + bq")
        assert expr.free_symbols == {sp.Symbol("theta"), sp.Symbol("kap"), sp.Symbol("bq")}

    def test_names_shadowing_sympy_functions_resolve_to_sympy(self):
        """Known limitation: SymPy's own namespace wins over auto-created symbols.

        ``beta`` is SymPy's beta function, so a model parameter named ``beta``
        cannot be parsed from text -- a documented gotcha for user-facing input.
        """
        assert safe_sympify("beta") is sp.beta

    def test_numeric_input_is_returned_unchanged(self):
        assert safe_sympify(3) == 3
        assert safe_sympify(sp.Rational(1, 2)) == sp.Rational(1, 2)

    def test_already_symbolic_input_passes_through(self):
        expr = sp.Symbol("q") ** 2
        assert safe_sympify(expr) == expr


class TestInjectionIsBlocked:
    @pytest.mark.parametrize("payload", MALICIOUS)
    def test_rejected_with_value_error(self, payload, tmp_path, monkeypatch):
        # any payload writing a file would write it under this name
        marker = tmp_path / "symbo_security_marker"
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("MARKER", str(marker))
        with pytest.raises(ValueError):
            safe_sympify(payload)
        assert not marker.exists()
        assert not os.path.exists("/tmp/symbo_security_marker")

    def test_no_eval_of_nested_string(self):
        """The specific historical escape: a nested sympify() call."""
        with pytest.raises(ValueError, match="sympify"):
            safe_sympify("sympify('__import__(\"os\").getlogin()')")

    def test_unknown_function_becomes_an_inert_symbolic_function(self):
        """Unrecognised names are *not* looked up: they stay unevaluated."""
        expr = safe_sympify("mystery_function(2)")
        assert isinstance(expr, sp.Function) and expr.func == sp.Function("mystery_function")

    def test_code_generating_sympy_calls_are_refused(self):
        """`lambdify` builds a callable from a *string* body, so it is refused."""
        with pytest.raises(ValueError, match="lambdify"):
            safe_sympify('lambdify("x", "1+1")')

    def test_attributes_and_subscripts_are_never_allowed(self):
        for payload in ("x.real", "x[0]", "(1).__class__"):
            with pytest.raises(ValueError):
                safe_sympify(payload)


class TestResourceGuards:
    def test_empty_expression_rejected(self):
        for bad in ("", "   ", "\n\t"):
            with pytest.raises(ValueError, match="Empty"):
                safe_sympify(bad)

    def test_overlong_expression_rejected(self):
        huge = "x+" * (MAX_EXPRESSION_LENGTH // 2 + 10) + "1"
        with pytest.raises(ValueError, match="too long"):
            safe_sympify(huge)

    def test_long_but_valid_expression_accepted(self):
        ok = "+".join(f"x**{i}" for i in range(200))
        assert len(ok) < MAX_EXPRESSION_LENGTH
        assert sp.count_ops(safe_sympify(ok)) > 100


class TestErrorContract:
    def test_syntax_error_is_reported_as_value_error(self):
        with pytest.raises(ValueError, match=r"Invalid syntax|transform"):
            safe_sympify("x ** ")

    def test_original_exception_is_chained(self):
        with pytest.raises(ValueError) as excinfo:
            safe_sympify("parse_expr('x')")
        assert excinfo.value.__cause__ is not None or "parse_expr" in str(excinfo.value)
