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
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

def safe_sympify(expr_str):
    """
    Safely parse a string into a SymPy expression.

    Pre-conditions:
    - expr_str is expected to be a string representing a mathematical expression.

    Post-conditions:
    - Returns a valid SymPy expression.
    - Raises ValueError if malicious or disallowed AST nodes (e.g., function calls to eval) are detected.

    Invariants maintained:
    - No arbitrary code execution via Python's eval or other dangerous built-ins.

    Time complexity: O(N) where N is the length of the string (for AST parsing).
    Space complexity: O(N) for the AST representation.
    Thread-safety guarantees: Thread-safe as it relies on stateless parsing functions.
    """
    if not isinstance(expr_str, str):
        # If it's already a SymPy object, just return it
        return sp.sympify(expr_str)

    # Phase 1: Transform string to python syntax
    from sympy.parsing.sympy_parser import stringify_expr

    allowed_names = {k: v for k, v in vars(sp).items() if not k.startswith('_')}
    global_dict = {'__builtins__': {}}

    try:
        transformed_code = stringify_expr(expr_str, allowed_names, global_dict, standard_transformations)
    except Exception as e:
        raise ValueError(f"Failed to transform expression: {e}")

    # Phase 2: AST-based validation (Static Analysis)
    # Reject function calls, attribute access, and subscripting associated with code execution.
    try:
        tree = ast.parse(transformed_code, mode='eval')
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                # We disallow attribute access to prevent obj.__class__ etc.
                raise ValueError("Attribute access is not allowed in symbolic expressions.")
            if isinstance(node, ast.Subscript):
                # We disallow subscripting to prevent __builtins__["exec"] bypassing
                raise ValueError("Subscript access is not allowed in symbolic expressions.")
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ('eval', 'exec', '__import__', 'getattr', 'setattr',
                                        'globals', 'locals', 'open', 'type', 'compile', 'memoryview'):
                        raise ValueError(f"Dangerous function call detected: {node.func.id}")

                # Check for Function('getattr') from stringify_expr transformation
                if isinstance(node.func, ast.Call) and isinstance(node.func.func, ast.Name) and node.func.func.id == 'Function':
                    if node.func.args and isinstance(node.func.args[0], ast.Constant):
                        func_name = node.func.args[0].value
                        if func_name in ('eval', 'exec', '__import__', 'getattr', 'setattr',
                                        'globals', 'locals', 'open', 'type', 'compile', 'memoryview'):
                            raise ValueError(f"Dangerous function call detected: {func_name}")
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax in expression: {e}")

    # Phase 3: Restricted parsing environment
    try:
        return parse_expr(
            expr_str,
            local_dict=allowed_names,
            global_dict=global_dict,
            transformations=standard_transformations
        )
    except Exception as e:
        raise ValueError(f"Failed to safely parse expression: {e}")
