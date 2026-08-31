# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0

"""
Security Module for Symbo
=========================

Provides strict validation and safety guards against arbitrary code execution
and other vulnerabilities, particularly around symbolic expression parsing.
"""

import ast

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, stringify_expr

#: Inputs longer than this are rejected outright: AST parsing and SymPy's
#: transformations are super-linear on adversarial input, so a length ceiling is
#: the cheapest denial-of-service guard for a parser exposed to user text.
MAX_EXPRESSION_LENGTH = 10_000

#: Names that can reach the interpreter, the filesystem or the object graph.
#: SymPy exports several of them (``sympify``, ``parse_expr``), and the parser
#: leaves calls to them in place as ordinary ``Call`` nodes -- which means a
#: payload such as ``sympify("__import__('os').system('...')")`` would otherwise
#: execute the nested string with full builtins. They are refused by name.
FORBIDDEN_CALLEES = frozenset({
    # interpreter / filesystem access
    'eval', 'exec', 'execfile', 'compile', 'open', 'input', 'print',
    '__import__', 'importlib', 'import_module',
    # object-graph probing used to escape restricted globals
    'globals', 'locals', 'vars', 'dir', 'getattr', 'setattr', 'delattr',
    'type', 'object', 'super', 'callable', 'memoryview', 'breakpoint',
    'exit', 'quit',
    # SymPy entry points that compile strings or run code. These are exported by
    # ``sympy`` and therefore *would* pass the allowlist check: ``lambdify`` and
    # ``autowrap`` build callables whose body comes from a plain string (which no
    # AST walk can inspect), ``preview``/``dotprint`` write files and spawn
    # programs, and ``test``/``this``/``init_*`` mutate global state.
    'sympify', 'parse_expr', 'auto_symbol', 'lambdify', 'autowrap', 'pycode',
    'cxxcode', 'fcode', 'jscode', 'octave_code', 'mathematica_code', 'dotprint',
    'preview', 'test', 'this', 'interactive', 'init_printing', 'init_session',
    'init_ipython',
})


def safe_sympify(expr_str):
    """
    Safely parse a string into a SymPy expression.

    Parameters
    ----------
    expr_str:
        A mathematical expression in text form. Non-string input is sympified
        directly (a number or an existing SymPy object is trusted; text is not).

    Returns
    -------
    sympy.Basic

    Raises
    ------
    ValueError
        If the input is empty, longer than :data:`MAX_EXPRESSION_LENGTH`, or if
        the parsed form contains attribute access, subscripting, or a call to
        anything that is not a public SymPy name (this is what blocks code
        injection, including nested re-entry through ``sympify``).

    Notes
    -----
    Validation happens on the AST *before* evaluation, because the danger in a
    symbolic parser is that evaluation is the normal code path: a check applied
    to the result would run the payload first.

    Time complexity: O(N) where N is the length of the string (for AST parsing).
    Space complexity: O(N) for the AST representation.
    Thread-safety: stateless, hence thread-safe.
    """
    if not isinstance(expr_str, str):
        # If it's already a SymPy object, just return it
        return sp.sympify(expr_str)

    if not expr_str.strip():
        raise ValueError("Empty expression")
    if len(expr_str) > MAX_EXPRESSION_LENGTH:
        raise ValueError(
            f"Expression too long ({len(expr_str)} > {MAX_EXPRESSION_LENGTH} characters)"
        )

    # Phase 1: Transform the string into Python syntax.
    allowed_names = {k: v for k, v in vars(sp).items() if not k.startswith('_')}
    global_dict = {'__builtins__': {}}

    try:
        transformed_code = stringify_expr(expr_str, allowed_names, global_dict,
                                          standard_transformations)
    except Exception as e:
        raise ValueError(f"Failed to transform expression: {e}") from e

    # Phase 2: Static analysis of the transformed code.
    #
    # Attribute access (``obj.__class__``) and subscripting
    # (``__builtins__["exec"]``) are rejected outright, and every call must be to
    # a public SymPy name that is not a known re-entry point. The allowlist is
    # what makes this robust rather than a growing blacklist: an unfamiliar
    # callable -- including one SymPy gains in a later release -- is refused
    # until it is deliberately permitted.
    try:
        tree = ast.parse(transformed_code, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax in expression: {e}") from e

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            raise ValueError("Attribute access is not allowed in symbolic expressions.")
        if isinstance(node, ast.Subscript):
            raise ValueError("Subscript access is not allowed in symbolic expressions.")
        if isinstance(node, ast.NamedExpr):  # (x := ...)
            raise ValueError("Walrus expressions are not allowed in symbolic expressions.")
        if not isinstance(node, ast.Call):
            continue

        func = node.func
        if isinstance(func, ast.Name):
            if func.id in FORBIDDEN_CALLEES:
                raise ValueError(f"Dangerous function call detected: {func.id}")
            if func.id not in allowed_names:
                raise ValueError(
                    f"Call to unknown function {func.id!r}: only SymPy functions "
                    "may be used in symbolic expressions."
                )
        elif isinstance(func, ast.Call):
            # stringify_expr renders undefined functions as Function("name")(args);
            # screen the quoted name as well so Function("eval") cannot be used to
            # smuggle a callable past the checks above.
            inner = func.func
            if not (isinstance(inner, ast.Name) and inner.id == 'Function'):
                raise ValueError("Unsupported call form in symbolic expression.")
            if func.args and isinstance(func.args[0], ast.Constant):
                func_name = func.args[0].value
                if (not isinstance(func_name, str) or func_name in FORBIDDEN_CALLEES
                        or not func_name.isidentifier()):
                    raise ValueError(f"Dangerous function call detected: {func_name}")
        else:
            raise ValueError("Unsupported call form in symbolic expression.")

    # Phase 3: Restricted parsing environment
    try:
        return parse_expr(
            expr_str,
            local_dict=allowed_names,
            global_dict=global_dict,
            transformations=standard_transformations
        )
    except Exception as e:
        raise ValueError(f"Failed to safely parse expression: {e}") from e
