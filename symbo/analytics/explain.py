# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Explainability through Derivative Trees
========================================

This module implements derivative tree construction for explainability,
mapping the influence of all initial symbolic variables on the final result.

The derivative tree includes intermediate symbolic derivatives and
substitutions, and can be easily rendered by external graphing libraries
like Graphviz.

Key Features:
- Full derivative tree construction
- Variable influence mapping
- Graphviz export
- Interactive visualization support
- Chain rule tracking
"""

import sympy as sp
import networkx as nx
from typing import List, Dict, Tuple, Optional, Any
import json


class DerivativeNode:
    """
    Node in a derivative tree.

    Represents either a variable, an intermediate expression,
    or a derivative operation.

    Attributes
    ----------
    expr : sp.Expr
        Symbolic expression at this node
    node_type : str
        'variable', 'expression', or 'derivative'
    label : str
        Human-readable label
    derivative_info : Dict
        Information about derivative operation
    """

    def __init__(self,
                 expr: sp.Expr,
                 node_type: str,
                 label: Optional[str] = None,
                 derivative_info: Optional[Dict] = None):
        """Initialize derivative node."""
        self.expr = expr
        self.node_type = node_type
        self.label = label or str(expr)
        self.derivative_info = derivative_info or {}

        # Content-based identity: two nodes are the same node when they are the
        # same operation on the same expression. ``id(expr)`` used to be used
        # here, which silently merged unrelated nodes whenever CPython recycled
        # the address of a freed expression.
        self.node_id = f"{node_type}:{self.label}:{expr}"

    def __repr__(self) -> str:
        return f"DerivativeNode({self.label}, type={self.node_type})"

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other) -> bool:
        return isinstance(other, DerivativeNode) and self.node_id == other.node_id


class DerivativeTree:
    """
    Derivative tree for explainability analysis.

    This class constructs a directed graph showing how each variable
    influences the final result through the chain of derivatives.

    Parameters
    ----------
    target_expr : sp.Expr
        Target expression to explain
    variables : List[sp.Symbol]
        Input variables to track

    Attributes
    ----------
    graph : nx.DiGraph
        Directed graph representation
    target : DerivativeNode
        Root node (target expression)
    variable_nodes : Dict[sp.Symbol, DerivativeNode]
        Mapping from variables to their nodes

    Examples
    --------
    >>> import sympy as sp, tempfile, os
    >>> x, y = sp.symbols('x y')
    >>> tree = DerivativeTree(x**2 * sp.sin(y) + sp.exp(x*y), [x, y])
    >>> tree.build()
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = tree.export_graphviz(os.path.join(tmp, "tree.dot"))
    ...     print(os.path.basename(path))
    tree.dot
    """

    def __init__(self,
                 target_expr: sp.Expr,
                 variables: List[sp.Symbol],
                 evaluation_point: Optional[Dict] = None):
        """
        Initialize derivative tree.

        Parameters
        ----------
        target_expr:
            Expression to explain.
        variables:
            Variables whose influence is tracked.
        evaluation_point:
            Optional ``{name or Symbol: float}`` mapping. With a point, each
            edge weight is the *magnitude of the derivative at that point*, which
            is what makes the ranking meaningful; without one the score falls back
            to an expression-complexity proxy.
        """
        self.target_expr = target_expr
        self.variables = list(variables)
        self.evaluation_point: Dict[sp.Basic, float] = {
            (k if isinstance(k, sp.Basic) else sp.Symbol(k)): v
            for k, v in (evaluation_point or {}).items()
        }
        unknown = sorted(str(k) for k in self.evaluation_point
                         if k not in self.variables and k not in target_expr.free_symbols)
        if unknown:
            raise ValueError(
                f"evaluation_point contains symbols the expression does not depend "
                f"on and that are not tracked variables: {unknown}"
            )

        # Graph structure
        self.graph = nx.DiGraph()

        # Nodes
        self.target = DerivativeNode(target_expr, 'expression',
                                     label=f"f = {target_expr}")
        self.variable_nodes = {
            var: DerivativeNode(var, 'variable', label=str(var))
            for var in variables
        }

        # Derivative cache
        self._derivative_cache: Dict[Tuple[sp.Expr, sp.Symbol], sp.Expr] = {}

        # Influence scores
        self.influence_scores: Dict[sp.Symbol, float] = {}

    def build(self, max_depth: int = 3):
        """
        Build the complete derivative tree.

        Traces derivatives from target to all input variables,
        creating intermediate nodes for chain rule steps.

        Parameters
        ----------
        max_depth : int
            Maximum depth for recursive derivative expansion
        """
        # Add target node
        self.graph.add_node(self.target, **self._node_attributes(self.target))

        # Add variable nodes
        for var_node in self.variable_nodes.values():
            self.graph.add_node(var_node, **self._node_attributes(var_node))

        # Compute first derivatives
        for var in self.variables:
            self._add_derivative_path(self.target_expr, var,
                                     self.target, self.variable_nodes[var],
                                     depth=0, max_depth=max_depth)

        # Compute influence scores
        self._compute_influence_scores()

    def _add_derivative_path(self,
                            expr: sp.Expr,
                            var: sp.Symbol,
                            parent_node: DerivativeNode,
                            var_node: DerivativeNode,
                            depth: int,
                            max_depth: int):
        """
        Add derivative path from expression to variable.

        Recursively expands derivatives using chain rule.
        """
        if depth > max_depth:
            return

        # Compute derivative
        deriv = self._compute_derivative(expr, var)

        if deriv == 0:
            # No influence
            return

        # Create derivative node
        deriv_node = DerivativeNode(
            deriv,
            'derivative',
            label=f"∂/∂{var}",
            derivative_info={
                'wrt': str(var),
                'order': 1,
                'parent_expr': str(expr)
            }
        )

        # Add node and edge
        if deriv_node not in self.graph:
            self.graph.add_node(deriv_node, **self._node_attributes(deriv_node))

        self.graph.add_edge(parent_node, deriv_node,
                           label=f"∂/∂{var}",
                           derivative=str(deriv))

        # Connect to variable node
        self.graph.add_edge(deriv_node, var_node,
                           label="depends_on",
                           influence=self._compute_local_influence(deriv))

        # Expand further when the derivative is itself a composite expression
        # over more than one variable (chain rule through each component).
        if (not deriv.is_number and depth < max_depth
                and len(deriv.free_symbols) > 1):
            for component_var in deriv.free_symbols:
                if component_var in self.variables:
                    self._add_derivative_path(
                        deriv, component_var,
                        deriv_node, self.variable_nodes[component_var],
                        depth + 1, max_depth
                    )

    def _compute_derivative(self, expr: sp.Expr, var: sp.Symbol) -> sp.Expr:
        """Compute and cache derivative."""
        key = (expr, var)
        if key not in self._derivative_cache:
            self._derivative_cache[key] = sp.diff(expr, var)
        return self._derivative_cache[key]

    def _compute_local_influence(self, deriv_expr: sp.Expr) -> float:
        """
        Compute local influence score for a derivative.

        The score is ``|d(deriv)/d(point)|``: numeric derivatives are used as
        they are, and when an ``evaluation_point`` was supplied the derivative is
        evaluated there first. Expressions that stay symbolic (an unfitted policy,
        a coefficient still to be learned) fall back to an operation-count proxy,
        which compares *structure* rather than sensitivity -- documented because
        the two must not be confused when reading a ranking.
        """
        try:
            # If derivative is numeric, use its magnitude
            if deriv_expr.is_number:
                return float(abs(deriv_expr))

            if self.evaluation_point:
                value = deriv_expr.subs(self.evaluation_point)
                if value.is_number:
                    return float(abs(sp.N(value)))

            # Otherwise, use complexity as proxy
            # Count operations as rough measure
            return float(sp.count_ops(deriv_expr))
        except Exception:
            return 1.0

    def _compute_influence_scores(self):
        """
        Compute overall influence score for each variable.

        Aggregates influence along all paths from target to variable.
        """
        for var in self.variables:
            var_node = self.variable_nodes[var]

            # Find all paths from target to this variable
            try:
                paths = list(nx.all_simple_paths(
                    self.graph, self.target, var_node,
                    cutoff=10
                ))

                # Aggregate influence over paths
                total_influence = 0.0
                for path in paths:
                    # Multiply influences along path
                    path_influence = 1.0
                    for i in range(len(path) - 1):
                        edge_data = self.graph.get_edge_data(path[i], path[i+1])
                        if edge_data and 'influence' in edge_data:
                            path_influence *= edge_data['influence']

                    total_influence += path_influence

                self.influence_scores[var] = total_influence

            except nx.NetworkXNoPath:
                self.influence_scores[var] = 0.0

    def _node_attributes(self, node: DerivativeNode) -> Dict[str, Any]:
        """Get node attributes for graph."""
        return {
            'label': node.label,
            'type': node.node_type,
            'expr': str(node.expr),
            'derivative_info': node.derivative_info
        }

    def direct_sensitivities(self) -> Dict[sp.Symbol, float]:
        """
        First-order sensitivity of the target with respect to each variable.

        Entries are ``None`` when the derivative stays symbolic, i.e. when no
        ``evaluation_point`` was supplied for a non-constant derivative. Unlike :attr:`influence_scores`, which aggregates every path through
        the derivative graph, this is the plain ``|d f / d v|`` evaluated at
        ``evaluation_point`` (absolute values of symbolic results are returned
        unchanged when no point is given). Use it when the question is "which
        input moves the output the most", which the path-based score -- being
        a sum over paths -- is not: variables that appear deep in the graph are
        counted once per path that reaches them.

        Returns
        -------
        Dict[sp.Symbol, Optional[float]]
            ``{variable: |df/dv|}``, in the order the variables were given
        """
        sensitivities: Dict[sp.Symbol, Optional[float]] = {}
        for var in self.variables:
            value = sp.diff(self.target_expr, var)
            if self.evaluation_point:
                value = value.subs(self.evaluation_point)
            try:
                sensitivities[var] = float(abs(sp.N(value)))
            except (TypeError, ValueError):
                # still symbolic (no evaluation point): there is no number to report
                sensitivities[var] = None
        return sensitivities

    def get_influence_ranking(self) -> List[Tuple[sp.Symbol, float]]:
        """
        Get variables ranked by influence on target.

        Returns
        -------
        List[Tuple[sp.Symbol, float]]
            List of (variable, influence_score) sorted by influence. The score
            is summed over all paths, so it grows with the number of paths that
            reach a variable; see :meth:`direct_sensitivities` for the
            path-independent first-order ranking.
        """
        return sorted(
            self.influence_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

    def export_graphviz(self, filename: str = "derivative_tree.dot",
                       include_edge_labels: bool = True):
        """
        Export tree to Graphviz DOT format.

        Parameters
        ----------
        filename : str
            Output file path
        include_edge_labels : bool
            Whether to include edge labels

        Returns
        -------
        str
            The path written, so the call composes with a follow-up render step.

        Examples
        --------
        Render the file with ``dot -Tpng tree.dot -o tree.png``::

            import sympy as sp, tempfile, os
            from symbo.analytics.explain import derivative_tree

            x, y = sp.symbols("x y")
            tree = derivative_tree(x**2 * sp.sin(y), [x, y])
            with tempfile.TemporaryDirectory() as tmp:
                path = tree.export_graphviz(os.path.join(tmp, "tree.dot"))
        """
        with open(filename, 'w') as f:
            f.write("digraph DerivativeTree {\n")
            f.write("  rankdir=TB;\n")
            f.write("  node [shape=box, style=rounded];\n\n")

            # Define node styles by type
            f.write("  // Nodes\n")
            for node in self.graph.nodes():
                node_data = self.graph.nodes[node]

                # Color by type
                if node_data['type'] == 'variable':
                    color = "lightblue"
                    shape = "ellipse"
                elif node_data['type'] == 'derivative':
                    color = "lightgreen"
                    shape = "box"
                else:
                    color = "lightyellow"
                    shape = "box"

                # Escape label for DOT
                label = node_data['label'].replace('"', '\\"')

                f.write(f'  "{node.node_id}" [label="{label}", '
                       f'fillcolor="{color}", style=filled, shape={shape}];\n')

            # Edges
            f.write("\n  // Edges\n")
            for u, v, edge_data in self.graph.edges(data=True):
                if include_edge_labels and 'label' in edge_data:
                    label = edge_data['label'].replace('"', '\\"')
                    f.write(f'  "{u.node_id}" -> "{v.node_id}" '
                           f'[label="{label}"];\n')
                else:
                    f.write(f'  "{u.node_id}" -> "{v.node_id}";\n')

            f.write("}\n")
        return filename

    def to_json(self) -> str:
        """
        Export tree structure to JSON.

        Returns
        -------
        str
            JSON representation suitable for web visualization
        """
        # Convert to JSON-serializable format
        data = {
            "target": {
                "expr": str(self.target_expr),
                "label": self.target.label
            },
            "variables": [str(var) for var in self.variables],
            "influence_scores": {
                str(var): score
                for var, score in self.influence_scores.items()
            },
            "nodes": [],
            "edges": []
        }

        # Add nodes
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            data["nodes"].append({
                "id": node.node_id,
                "label": node_data['label'],
                "type": node_data['type'],
                "expr": node_data['expr']
            })

        # Add edges
        for u, v, edge_data in self.graph.edges(data=True):
            data["edges"].append({
                "source": u.node_id,
                "target": v.node_id,
                "label": edge_data.get('label', ''),
                "influence": edge_data.get('influence', 1.0)
            })

        return json.dumps(data, indent=2)

    def visualize_influence(self) -> str:
        """
        Generate text-based influence summary.

        Returns
        -------
        str
            Formatted influence report
        """
        lines = ["Variable Influence Analysis", "=" * 40]

        for var, score in self.get_influence_ranking():
            # Create simple bar chart
            bar_length = int(min(50, score))
            bar = "█" * bar_length
            lines.append(f"{var!s:10s} {score:10.4f} {bar}")

        return "\n".join(lines)


def derivative_tree(expr: sp.Expr,
                   variables: List[sp.Symbol],
                   evaluation_point: Optional[Dict] = None,
                   max_depth: int = 3) -> DerivativeTree:
    """
    Build a derivative tree for explainability analysis.

    Parameters
    ----------
    expr : sp.Expr
        Expression to analyze
    variables : List[sp.Symbol]
        Variables to track
    evaluation_point : dict, optional
        Point at which derivatives are measured (see :class:`DerivativeTree`).
    max_depth : int
        How far the chain rule is expanded.

    Returns
    -------
    DerivativeTree
        Constructed (already built) derivative tree

    Examples
    --------
    >>> import sympy as sp
    >>> x, y, z = sp.symbols('x y z')
    >>> tree = derivative_tree(x**2 * y + sp.sin(z), [x, y, z],
    ...                        evaluation_point={x: 1.0, y: 1.0, z: 0.0})
    >>> tree.get_influence_ranking()[0][0]
    x
    """
    tree = DerivativeTree(expr, variables, evaluation_point=evaluation_point)
    tree.build(max_depth=max_depth)
    return tree


__all__ = [
    'DerivativeNode',
    'DerivativeTree',
    'derivative_tree',
]
