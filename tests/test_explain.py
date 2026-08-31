# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Derivative-tree explainability: graph shape, rankings, JSON and Graphviz."""

import json

import pytest
import sympy as sp

from symbo.analytics.explain import DerivativeNode, DerivativeTree, derivative_tree

x, y = sp.symbols("x y")


class TestDerivativeNode:
    def test_identity_is_content_based(self):
        a = DerivativeNode(x**2, "derivative", label="∂/∂x")
        b = DerivativeNode(x**2, "derivative", label="∂/∂x")
        c = DerivativeNode(x**2, "derivative", label="∂/∂y")
        assert a == b
        assert hash(a) == hash(b)
        assert a != c
        assert a != "not a node"
        assert "∂/∂x" in repr(a)

    def test_recycled_ids_do_not_collide_nodes(self):
        """Regression: nodes used to be keyed by ``id(expr)``, whose addresses are recycled."""
        nodes = [DerivativeNode(x**2 + i, "derivative", label=f"∂/∂v{i}") for i in range(200)]
        assert len({n.node_id for n in nodes}) == len(nodes)


@pytest.fixture()
def tree():
    return derivative_tree(sp.exp(x) * sp.cos(y), [x, y],
                           evaluation_point={x: 0.5, y: 0.2}, max_depth=2)


@pytest.fixture()
def deep_tree():
    return derivative_tree(sp.exp(x) * sp.cos(y), [x, y],
                           evaluation_point={x: 0.5, y: 0.2}, max_depth=3)


class TestBuild:
    def test_nodes_and_edges(self, tree):
        assert len(tree.graph.nodes) == 9
        assert len(tree.graph.edges) == 16
        assert tree.target in tree.graph.nodes
        for var in (x, y):
            assert tree.variable_nodes[var] in tree.graph.nodes

    def test_attributes_are_json_ready(self, tree):
        for attrs in tree.graph.nodes(data=True):
            assert set(attrs[1]) == {"label", "type", "expr", "derivative_info"}
            json.dumps(attrs[1])

    def test_second_order_derivatives_appear(self):
        t = derivative_tree(x**3 * y**2, [x, y], max_depth=2)
        labels = [n.label for n in t.graph.nodes if n.node_type == "derivative"]
        assert labels.count("∂/∂x") >= 2
        assert "∂/∂y" in labels

    def test_build_is_idempotent(self, tree):
        before = (len(tree.graph.nodes), len(tree.graph.edges))
        tree.build(max_depth=2)
        assert (len(tree.graph.nodes), len(tree.graph.edges)) == before

    def test_constant_expression_has_no_derivative_nodes(self):
        t = derivative_tree(sp.Integer(7), [x, y])
        assert [n for n in t.graph.nodes if n.node_type == "derivative"] == []
        assert t.get_influence_ranking() == [(x, 0.0), (y, 0.0)]

    def test_zero_derivative_branch_is_dropped(self):
        t = derivative_tree(x**2, [x, y])
        assert t.graph.in_degree(t.variable_nodes[y]) == 0
        assert t.graph.in_degree(t.variable_nodes[x]) == 1

    def test_evaluation_point_accepts_string_keys(self):
        t = DerivativeTree(x**2 + y, [x, y], evaluation_point={"x": 3.0, "y": 1.0})
        t.build()
        assert t.direct_sensitivities() == {x: 6.0, y: 1.0}


class TestRanking:
    def test_influence_ranking_is_sorted(self, tree):
        ranking = tree.get_influence_ranking()
        scores = [score for _, score in ranking]
        assert scores == sorted(scores, reverse=True)

    def test_ranking_follows_the_dominant_variable(self):
        t = derivative_tree(x**2 + y**3, [x, y], evaluation_point={x: 1.0, y: 2.0})
        assert [var for var, _ in t.get_influence_ranking()] == [y, x]

    def test_direct_sensitivities_match_manual_differentiation(self):
        t = derivative_tree(x**2 + y**3, [x, y], evaluation_point={x: 1.0, y: 2.0})
        assert t.direct_sensitivities() == {x: 2.0, y: 12.0}

    def test_direct_sensitivities_are_numeric_only(self):
        t = derivative_tree(x**2 + y**3, [x, y])
        assert t.direct_sensitivities() == {x: None, y: None}

    def test_path_aggregation_and_direct_sensitivity_answer_different_questions(self, deep_tree):
        """The path score counts paths, the direct score counts sensitivity."""
        direct = deep_tree.direct_sensitivities()
        assert direct[x] == pytest.approx(float(sp.exp(sp.Rational(1, 2)) * sp.cos(sp.Rational(1, 5))))
        assert direct[x] > direct[y]
        assert deep_tree.influence_scores[y] > deep_tree.influence_scores[x]

    def test_influence_scores_are_positive(self, tree):
        assert all(v > 0 for v in tree.influence_scores.values())
        assert len(tree.influence_scores) == 2


class TestOutputs:
    def test_to_json_schema(self, tree):
        payload = json.loads(tree.to_json())
        assert set(payload) == {"target", "variables", "influence_scores", "nodes", "edges"}
        assert payload["target"] == {"expr": "exp(x)*cos(y)", "label": "f = exp(x)*cos(y)"}
        assert payload["variables"] == ["x", "y"]
        assert len(payload["nodes"]) == len(tree.graph.nodes)
        assert len(payload["edges"]) == len(tree.graph.edges)
        assert set(payload["influence_scores"]) == {"x", "y"}
        assert all(edge["influence"] > 0 for edge in payload["edges"])

    def test_export_graphviz_returns_the_written_path(self, tree, tmp_path):
        target = tmp_path / "tree.dot"
        assert tree.export_graphviz(str(target)) == str(target)
        text = target.read_text()
        assert text.startswith("digraph DerivativeTree {")
        assert text.rstrip().endswith("}")
        assert "exp(x)*cos(y)" in text
        assert "lightblue" in text  # variable nodes are colour-coded

    def test_edge_labels_can_be_omitted(self, tree, tmp_path):
        with_labels = tree.export_graphviz(str(tmp_path / "a.dot"))
        bare = tree.export_graphviz(str(tmp_path / "b.dot"), include_edge_labels=False)
        with open(with_labels) as f:
            labelled = f.read()
        with open(bare) as f:
            unlabelled = f.read()
        assert any('->' in line and '[label=' in line for line in labelled.splitlines())
        assert not any('->' in line and '[label=' in line for line in unlabelled.splitlines())
        assert unlabelled.count("->") == labelled.count("->")

    def test_text_summary_lists_every_variable(self, tree):
        summary = tree.visualize_influence()
        assert "Variable Influence Analysis" in summary
        for var in (x, y):
            assert str(var) in summary

    def test_unbuilt_tree_is_empty(self):
        t = DerivativeTree(x * y, [x, y])
        assert t.evaluation_point == {}
        assert t.graph.number_of_nodes() == 0
        t.build()
        assert t.graph.number_of_nodes() > 0
