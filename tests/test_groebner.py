# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Gröbner solvers: batch, streaming, real-time chunks and infinite families."""

import json
import types

import pytest
import sympy as sp

from symbo.solver.groebner import (
    GröbnerBasisState,
    RealTimeGröbnerSolver,
    StreamingGröbnerSolver,
    handle_infinite_solutions,
    solve_with_groebner,
)

x, y = sp.symbols("x y")


class TestBatchSolve:
    def test_two_by_two_system(self):
        assert solve_with_groebner([x + y - 2, x - y], [x, y]) == \
            [{x: sp.Integer(1), y: sp.Integer(1)}]

    def test_quadratic_system_returns_every_branch(self):
        sols = solve_with_groebner([x**2 - 1, x + y], [x, y])
        assert sorted(sols, key=lambda s: float(s[x])) == [
            {x: sp.Integer(-1), y: sp.Integer(1)},
            {x: sp.Integer(1), y: sp.Integer(-1)},
        ]

    def test_stream_flag_returns_a_generator_not_a_list(self):
        stream = solve_with_groebner([x**2 - 1, x + y], [x, y], stream=True)
        assert isinstance(stream, types.GeneratorType)
        chunks = list(stream)
        assert [c["type"] for c in chunks] == ["solution", "solution", "completion"]
        assert chunks[-1]["total_solutions"] == 2

    def test_inconsistent_system_reports_instead_of_hanging(self):
        assert solve_with_groebner([x**2 + 1, x - 1], [x]) == []

    def test_no_equations_yields_no_solutions(self):
        assert solve_with_groebner([], [x]) == []


class TestStreaming:
    @pytest.fixture()
    def solver(self):
        return StreamingGröbnerSolver([x + y - 2, x**2 + y**2 - 2], [x, y],
                                      order="lex", chunk_size=1)

    def test_basis_is_streamed_one_chunk_at_a_time(self, solver):
        chunks = list(solver.stream_basis())
        assert [c["type"] for c in chunks] == ["basis_chunk", "basis_chunk", "completion"]
        assert chunks[0]["items"][0]["poly_str"].startswith("Poly(")
        assert chunks[0]["items"][0]["degree"] == 1
        assert chunks[0]["items"][0]["variables"] == ["x", "y"]
        assert chunks[-1]["total_basis_polynomials"] == 2
        assert chunks[-1]["status"] == "completed"

    def test_progress_advances_to_completion(self, solver):
        percentages = [c["progress"]["percentage"] for c in solver.stream_basis()
                       if "progress" in c]
        assert percentages == [50.0, 100.0]

    def test_chunk_size_controls_the_number_of_chunks(self):
        solver = StreamingGröbnerSolver([x + y - 2, x**2 + y**2 - 2], [x, y], chunk_size=2)
        chunks = [c for c in solver.stream_basis() if c["type"] == "basis_chunk"]
        assert len(chunks) == 1
        assert len(chunks[0]["items"]) == 2

    def test_solutions_are_streamed_as_strings_for_wire_safety(self, solver):
        chunks = list(solver.stream_solutions())
        assert [c["type"] for c in chunks][-1] == "completion"
        first = chunks[0]["solution"]
        assert set(first) == {"x", "y"}
        assert all(isinstance(v, str) for v in first.values())
        assert chunks[0]["is_real"] is True

    def test_non_real_solutions_are_flagged_not_dropped(self):
        solver = StreamingGröbnerSolver([x**2 + y**2 + 1, x - y], [x, y])
        chunks = list(solver.stream_solutions())
        assert len(chunks) == 3
        assert all(c["is_real"] is False for c in chunks[:2])

    def test_state_tracks_the_computation(self, solver):
        list(solver.stream_basis())
        state = solver.get_state()
        assert isinstance(state, GröbnerBasisState)
        assert state.status == "completed"
        assert state.variables == [x, y]
        assert state.to_dict()["num_basis_polys"] == 2
        assert state.to_dict()["num_solutions"] == 0
        assert list(solver.stream_solutions())[-1]["total_solutions"] == 1
        assert solver.get_state().to_dict()["num_solutions"] == 1


class TestRealTime:
    def test_is_a_streaming_solver_with_the_same_chunking(self):
        rt = RealTimeGröbnerSolver([x**2 - 1, x + y], [x, y], chunk_size=2)
        assert isinstance(rt, StreamingGröbnerSolver)
        chunks = list(rt.stream_basis())
        assert chunks[-1]["type"] == "completion"

    def test_progress_callback_sees_every_chunk(self):
        seen = []
        rt = RealTimeGröbnerSolver([x**2 - 1, x + y], [x, y], chunk_size=1,
                                   progress_callback=seen.append)
        list(rt.stream_basis())
        assert [p["percentage"] for p in seen] == [50.0, 100.0]

    def test_callback_is_optional(self):
        rt = RealTimeGröbnerSolver([x - 1], [x])
        assert rt.progress_callback is None
        assert list(rt.stream_basis())[-1]["status"] == "completed"


class TestInfiniteFamilies:
    def test_one_free_variable_is_reported_as_a_family(self):
        G = sp.groebner([x + y - 2], x, y, order="lex")
        info = handle_infinite_solutions(list(G.polys), [x, y])
        assert info["has_infinite_solutions"] is True
        assert info["dimension"] == 1
        assert info["free_variables"] == ["y"]
        assert info["dependent_variables"] == ["x"]
        assert "1-dimensional" in info["parametric_description"]

    def test_finite_system_is_not_flagged(self):
        G = sp.groebner([x + y - 2, x - y], x, y, order="lex")
        info = handle_infinite_solutions(list(G.polys), [x, y])
        assert info["has_infinite_solutions"] is False
        assert info["dimension"] == 0
        assert info["free_variables"] == []

    def test_no_equations_means_everything_is_free(self):
        info = handle_infinite_solutions([], [x, y])
        assert info["has_infinite_solutions"] is True
        assert info["dimension"] == 2
        assert info["basis"] == []


class TestStateObject:
    def test_to_dict_reports_counters_not_objects(self):
        state = GröbnerBasisState([x**2 - 1, x + y], [x, y], order="lex")
        state.status = "completed"
        state.solutions = [{x: sp.Integer(1), y: sp.Integer(-1)}]
        payload = state.to_dict()
        assert payload["polynomials"] == ["x**2 - 1", "x + y"]
        assert payload["variables"] == ["x", "y"]
        assert payload["order"] == "lex"
        assert payload["num_solutions"] == 1
        assert payload["error"] is None

    def test_to_json_is_utf8_safe(self):
        state = GröbnerBasisState([x - 1], [x], order="grlex")
        state.error = "Gröbner basis failed"
        restored = json.loads(state.to_json())
        assert restored["error"] == "Gröbner basis failed"
        assert restored["order"] == "grlex"

    def test_initial_status(self):
        state = GröbnerBasisState([x - 1], [x])
        assert state.status == "initialized"
        assert state.basis is None
        assert state.to_dict()["num_basis_polys"] == 0
