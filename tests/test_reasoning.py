# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Symbolic A* search over an energy landscape, and the HSWS semantic scorer."""

import itertools

import pytest
import sympy as sp

from symbo.reasoning.a_star import (
    EnergyLandscape,
    SymbolicAStarPathfinder,
    SymbolicEnergyLandscape,
)
from symbo.reasoning.hsws import (
    Betaconcept,
    Concept,
    DictionarySemanticEngine,
    HSWS,
    RobustSemanticEngine,
    Subconcept,
)

x, y = sp.symbols("x y")
BOUNDS = {x: (-3.0, 3.0), y: (-3.0, 3.0)}


@pytest.fixture()
def landscape():
    return SymbolicEnergyLandscape((x - 1) ** 2 + (y + 2) ** 2, [x, y])


class TestSymbolicEnergyLandscape:
    def test_energy_accepts_symbol_dicts(self, landscape):
        assert landscape.energy({x: 1.0, y: -2.0}) == pytest.approx(0.0)
        assert landscape.energy({x: 0.0, y: 0.0}) == pytest.approx(5.0)

    def test_missing_variables_default_to_zero(self, landscape):
        assert landscape.energy({x: 1.0}) == pytest.approx(4.0)

    def test_gradient_matches_symbolic_differentiation(self, landscape):
        assert landscape.gradient({x: 1.0, y: 0.0}) == {x: 0.0, y: 4.0}
        assert landscape.influence_map({x: 1.0, y: 0.0}) == {x: 0.0, y: 4.0}

    def test_custom_landscape_subclasses_the_abc(self):
        class Quadratic(EnergyLandscape):
            def energy(self, state):
                return sum(v * v for v in state.values())

            def gradient(self, state):
                return {var: 2.0 * state.get(var, 0.0) for var in self.variables}

            def influence_map(self, state):
                return self.gradient(state)

        quad = Quadratic([x, y])
        with pytest.raises(TypeError):
            EnergyLandscape([x, y])  # the ABC is not instantiable
        finder = SymbolicAStarPathfinder(quad, [x, y], BOUNDS, step_size=0.5)
        path = finder.find_path({x: -1.0, y: 1.0}, {x: 0.0, y: 0.0})
        assert path[0] == {x: -1.0, y: 1.0}
        assert path[-1] == {x: 0.0, y: 0.0}


class TestPathfinder:
    def test_rejects_nonsense_options(self, landscape):
        with pytest.raises(ValueError, match="mode"):
            SymbolicAStarPathfinder(landscape, [x, y], BOUNDS, mode="sideways")
        with pytest.raises(ValueError, match="step_size"):
            SymbolicAStarPathfinder(landscape, [x, y], BOUNDS, step_size=0.0)

    def test_state_conversions_round_trip(self, landscape):
        finder = SymbolicAStarPathfinder(landscape, [x, y], BOUNDS)
        state = {x: 1.5, y: -2.25}
        assert finder.state_to_tuple(state) == (1.5, -2.25)
        assert finder.tuple_to_state((1.5, -2.25)) == state

    def test_heuristic_is_zero_at_the_goal_and_grows_with_distance(self, landscape):
        finder = SymbolicAStarPathfinder(landscape, [x, y], BOUNDS)
        goal = {x: 1.0, y: -2.0}
        assert finder.heuristic(goal, goal) == 0.0
        near = finder.heuristic({x: 1.0, y: -1.0}, goal)
        far = finder.heuristic({x: -3.0, y: 3.0}, goal)
        assert 0.0 < near < far

    def test_edge_cost_charges_distance_and_uphill_moves(self, landscape):
        finder = SymbolicAStarPathfinder(landscape, [x, y], BOUNDS)
        # downhill step: only the geometric cost is charged
        assert finder.edge_cost({x: 0.0, y: 0.0}, {x: 1.0, y: 0.0}) == pytest.approx(1.0)
        # uphill step: distance + the climb
        assert finder.edge_cost({x: 1.0, y: 0.0}, {x: 0.0, y: 0.0}) == pytest.approx(2.0)
        climbing = SymbolicAStarPathfinder(landscape, [x, y], BOUNDS, mode="maximize")
        assert climbing.edge_cost({x: 0.0, y: 0.0}, {x: 1.0, y: 0.0}) == pytest.approx(2.0)

    def test_neighbors_respect_the_step_size(self, landscape):
        finder = SymbolicAStarPathfinder(landscape, [x, y], BOUNDS, step_size=0.25)
        neighbours = finder.get_neighbors({x: 0.0, y: 0.0})
        assert len(neighbours) == 4
        assert all(abs(abs(n[x] - 0.0) + abs(n[y] - 0.0) - 0.25) < 1e-12
                   for n in neighbours)

    def test_neighbors_are_clipped_to_the_bounds(self, landscape):
        finder = SymbolicAStarPathfinder(landscape, [x, y], BOUNDS, step_size=0.5)
        for state in finder.get_neighbors({x: 3.0, y: 0.0}):
            assert state[x] <= 3.0
            assert -3.0 <= state[y] <= 3.0

    def test_minimising_path_descends_the_landscape(self, landscape):
        finder = SymbolicAStarPathfinder(landscape, [x, y], BOUNDS, step_size=0.25)
        path = finder.find_path({x: -2.0, y: 1.0}, {x: 1.0, y: -2.0})
        assert path[0] == {x: -2.0, y: 1.0}
        assert path[-1] == {x: 1.0, y: -2.0}
        profile = [landscape.energy(state) for state in path]
        assert profile[0] == pytest.approx(18.0)
        assert profile[-1] == pytest.approx(0.0)
        # a *symbolic* path need not be monotone, but it must not wander far up
        assert max(profile) <= profile[0]

    def test_maximising_path_climbs(self, landscape):
        finder = SymbolicAStarPathfinder(landscape, [x, y], BOUNDS, step_size=0.5,
                                         mode="maximize")
        path = finder.find_path({x: 1.0, y: -2.0}, {x: -3.0, y: 3.0})
        assert landscape.energy(path[-1]) > landscape.energy(path[0])

    def test_compute_path_cost_is_the_sum_of_edges(self, landscape):
        finder = SymbolicAStarPathfinder(landscape, [x, y], BOUNDS, step_size=0.25)
        path = finder.find_path({x: 0.0, y: 0.0}, {x: 0.5, y: 0.0})
        manual = sum(finder.edge_cost(a, b) for a, b in itertools.pairwise(path))
        assert float(finder.compute_path_cost(path)) == pytest.approx(manual)

    def test_analyze_path_reports_the_energy_profile(self, landscape):
        finder = SymbolicAStarPathfinder(landscape, [x, y], BOUNDS, step_size=1.0)
        path = finder.find_path({x: 0.0, y: 0.0}, {x: 1.0, y: -2.0})
        info = finder.analyze_path(path)
        assert info["length"] == len(path)
        assert info["energy_profile"][0] == pytest.approx(info["start_energy"])
        assert info["energy_profile"][-1] == pytest.approx(info["end_energy"])
        assert info["energy_change"] == pytest.approx(info["end_energy"] - info["start_energy"])
        assert info["cost"] == pytest.approx(float(finder.compute_path_cost(path)))

    def test_search_gives_up_when_the_goal_is_unreachable(self, landscape):
        finder = SymbolicAStarPathfinder(landscape, [x, y], {x: (0.0, 0.5), y: (0.0, 0.5)},
                                         step_size=0.25)
        with pytest.raises((ValueError, RuntimeError), match=r"path|iteration"):
            finder.find_path({x: 0.0, y: 0.0}, {x: 3.0, y: 3.0})


class TestSemanticEngine:
    def test_registration_and_scores(self):
        engine = DictionarySemanticEngine()
        engine.register_meaning("ai", "artificial intelligence")
        engine.register_synonym("ai", "machine intelligence")
        engine.register_antonym("ai", "natural stupidity")
        assert engine.get_meaning_score("AI", "artificial intelligence") == 1.0
        assert engine.get_synonym_score("ai", "machine intelligence") == 1.0
        assert engine.get_antonym_score("ai", "natural stupidity") == 1.0
        assert engine.get_meaning_score("ai", "cooking") == 0.0
        # synonyms and antonyms are registered symmetrically
        assert engine.get_synonym_score("machine intelligence", "ai") == 1.0
        # a term is always its own synonym
        assert engine.get_synonym_score("ai", "ai") == 1.0

    def test_fuzzy_matching_is_tuned_by_the_threshold(self):
        lenient = DictionarySemanticEngine(fuzziness_threshold=0.6)
        strict = DictionarySemanticEngine(fuzziness_threshold=0.99)
        query = "artificl inteligence is fun"
        targets = ["artificial intelligence"]
        assert lenient.match(query, targets)
        assert not strict.match(query, targets)

    def test_match_reports_confidence(self):
        engine = DictionarySemanticEngine()
        hits = engine.match("artificial intelligence is fun", ["artificial intelligence"])
        assert hits == [("artificial intelligence", 1.0)]
        near = engine.match("artificl inteligence", ["artificial intelligence"])
        assert near and near[0][1] < 1.0

    def test_tokenizer(self):
        engine = DictionarySemanticEngine()
        assert engine.tokenize("Hello,  World!") == ["hello", "world"]

    def test_legacy_alias_is_the_same_class(self):
        assert DictionarySemanticEngine is RobustSemanticEngine

    def test_unregistered_term_scores_zero(self):
        engine = DictionarySemanticEngine()
        assert engine.get_meaning_score("nothing", "here") == 0.0


@pytest.fixture()
def concept():
    engine = DictionarySemanticEngine()
    engine.register_meaning("AI", "artificial intelligence")
    engine.register_synonym("AI", "machine intelligence")
    engine.register_meaning("Machine Learning", "ml")
    engine.register_synonym("Machine Learning", "neural networks")
    sub = Subconcept("Machine Learning")
    sub.add_betaconcept(Betaconcept("Neural Networks"))
    root = Concept("AI")
    root.add_subconcept(sub)
    return engine, root


class TestHSWS:
    def test_result_schema(self, concept):
        engine, root = concept
        result = HSWS(engine).process(root, "artificial intelligence")
        assert set(result) == {"total_rt", "coordinates", "interpretation", "concept_name"}
        assert result["concept_name"] == "AI"
        assert set(result["coordinates"]) == {"x", "y", "z"}

    def test_a_matching_query_scores_higher_than_an_unrelated_one(self, concept):
        engine, root = concept
        hit = HSWS(engine).process(root, "artificial intelligence")
        miss = HSWS(engine).process(root, "a cooking recipe")
        assert hit["total_rt"] > miss["total_rt"]
        assert miss["total_rt"] == root.base_rt
        assert hit["interpretation"] == "Plausible Connection"
        assert miss["interpretation"] == "Weak / Indeterminate"

    def test_registered_meanings_are_matched_not_only_names(self, concept):
        """Regression: only the node name was matched, so register_meaning() had no effect."""
        engine, root = concept
        HSWS(engine).process(root, "ml")
        assert root.subconcepts[0].matched_meaning is True

    def test_antonyms_subtract(self, concept):
        engine, root = concept
        engine.register_antonym("AI", "natural stupidity")
        with_antonym = HSWS(engine).process(root, "artificial intelligence or natural stupidity")
        without = HSWS(engine).process(root, "artificial intelligence")
        assert with_antonym["total_rt"] == without["total_rt"] - 300.0

    def test_overlap_reinforcement_triggers(self, concept):
        engine, root = concept
        sub = root.subconcepts[0]
        result = HSWS(engine).process(root, "neural networks")
        assert sub.overlap_triggered is True
        assert sub.overlap_value == sub.OVERLAP_MULTIPLIER * sub.base_rt
        assert result["coordinates"]["z"] > 0.0

    def test_matches_are_reset_between_queries(self, concept):
        engine, root = concept
        HSWS(engine).process(root, "artificial intelligence")
        assert root.matched_meaning is True
        HSWS(engine).process(root, "a cooking recipe")
        assert root.matched_meaning is False
        assert root.matched_synonym is False

    def test_interpretation_thresholds(self, concept):
        engine, _root = concept
        hs = HSWS(engine)
        for rt, expected in [(3500.0, "Absolute Truth / High Certainty"),
                             (2000.0, "Strong Plausibility"),
                             (900.0, "Plausible Connection"),
                             (10.0, "Weak / Indeterminate"),
                             (-5.0, "Contradiction / Falsehood")]:
            assert hs._interpret_result(rt) == expected

    def test_empty_engine_still_returns_the_base_score(self):
        engine = DictionarySemanticEngine()
        result = HSWS(engine).process(Concept("lonely"), "anything")
        assert result["total_rt"] == 500.0

    def test_betaconcept_scoring(self):
        bc = Betaconcept("x", matched_meaning=True)
        assert bc.calculate_rt() == bc.base_rt + bc.MEANING_WEIGHT
        bc.matched_synonym = True
        assert bc.calculate_rt() == bc.base_rt + bc.MEANING_WEIGHT + bc.SYNONYM_WEIGHT
        bc.reset_matches()
        assert bc.calculate_rt() == 0.0

    def test_subconcept_scoring_without_engine(self):
        sub = Subconcept("s", matched_meaning=True)
        total, y_comp, z_comp = sub.calculate_rt(DictionarySemanticEngine())
        assert y_comp == sub.base_rt + sub.SCN_MEANING_WEIGHT
        assert z_comp == 0.0
        assert total == y_comp
