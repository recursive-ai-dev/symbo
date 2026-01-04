# Copyright 2025
# Recursive AI Devs
# Symbo-Logic Hierarchical Semantic Weighting System (HSWS) v1.0

"""
Symbo-Logic Hierarchical Semantic Weighting System (HSWS) v1.0
==============================================================

The HSWS is an advanced branching logic engine designed to quantify the semantic
alignment of a "Main Concept" against a user's query. It utilizes a weighted
hierarchical tree (Concept > Subconcept > Betaconcept) to calculate a
"Rotation Value" (Rt), mapping the result to a logical 3D sphere.
"""

import math
from typing import List, Dict, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# --- Semantic Engine Interface ---

class SemanticEngine(ABC):
    """
    Abstract base class for the implementation layer's semantic engine.
    Responsible for determining Meaning, Synonym, and Antonym relationships.
    """

    @abstractmethod
    def get_meaning_score(self, term1: str, term2: str) -> float:
        """Return a score (0.0 to 1.0) indicating if term2 is the meaning/definition of term1."""
        pass

    @abstractmethod
    def get_synonym_score(self, term1: str, term2: str) -> float:
        """Return a score (0.0 to 1.0) indicating if term2 is a synonym of term1."""
        pass

    @abstractmethod
    def get_antonym_score(self, term1: str, term2: str) -> float:
        """Return a score (0.0 to 1.0) indicating if term2 is an antonym of term1."""
        pass

    @abstractmethod
    def is_meaning(self, term: str, definition: str) -> bool:
        """Binary check for meaning relationship."""
        pass

    @abstractmethod
    def is_synonym(self, term: str, synonym: str) -> bool:
        """Binary check for synonym relationship."""
        pass

    @abstractmethod
    def is_antonym(self, term: str, antonym: str) -> bool:
        """Binary check for antonym relationship."""
        pass


class DictionarySemanticEngine(SemanticEngine):
    """
    A production-ready dictionary-based semantic engine.
    Allows explicit registration of semantic relationships.
    Useful for precise, domain-specific logic where 'fuzzy' matching is undesirable.
    """

    def __init__(self):
        # Maps term -> set of related terms
        self.meanings: Dict[str, set] = {}
        self.synonyms: Dict[str, set] = {}
        self.antonyms: Dict[str, set] = {}

    def register_meaning(self, term: str, definition: str):
        term = term.lower().strip()
        definition = definition.lower().strip()
        if term not in self.meanings:
            self.meanings[term] = set()
        self.meanings[term].add(definition)

    def register_synonym(self, term: str, synonym: str):
        term = term.lower().strip()
        synonym = synonym.lower().strip()
        if term not in self.synonyms:
            self.synonyms[term] = set()
        self.synonyms[term].add(synonym)
        # Synonyms are typically symmetric
        if synonym not in self.synonyms:
            self.synonyms[synonym] = set()
        self.synonyms[synonym].add(term)

    def register_antonym(self, term: str, antonym: str):
        term = term.lower().strip()
        antonym = antonym.lower().strip()
        if term not in self.antonyms:
            self.antonyms[term] = set()
        self.antonyms[term].add(antonym)
        # Antonyms are typically symmetric
        if antonym not in self.antonyms:
            self.antonyms[antonym] = set()
        self.antonyms[antonym].add(term)

    def get_meaning_score(self, term1: str, term2: str) -> float:
        return 1.0 if self.is_meaning(term1, term2) else 0.0

    def get_synonym_score(self, term1: str, term2: str) -> float:
        return 1.0 if self.is_synonym(term1, term2) else 0.0

    def get_antonym_score(self, term1: str, term2: str) -> float:
        return 1.0 if self.is_antonym(term1, term2) else 0.0

    def is_meaning(self, term: str, definition: str) -> bool:
        t = term.lower().strip()
        d = definition.lower().strip()
        return d in self.meanings.get(t, set())

    def is_synonym(self, term: str, synonym: str) -> bool:
        t = term.lower().strip()
        s = synonym.lower().strip()
        return s in self.synonyms.get(t, set())

    def is_antonym(self, term: str, antonym: str) -> bool:
        t = term.lower().strip()
        a = antonym.lower().strip()
        return a in self.antonyms.get(t, set())


# --- HSWS Components ---

@dataclass
class Betaconcept:
    """
    The Betaconcept (BCn) - The Granular Data
    Specific data points or keywords found in the input.
    Range: +35.000 Rt to -35.000 Rt
    """
    name: str
    base_rt: float = 35.0

    # Matches found in the query/input
    matched_meaning: bool = False
    matched_synonym: bool = False
    matched_antonym: bool = False

    # Weights
    MEANING_WEIGHT: float = 15.0
    SYNONYM_WEIGHT: float = 10.0
    ANTONYM_WEIGHT: float = -10.0 # Negative value for contradiction

    def calculate_rt(self) -> float:
        """
        Rt(BCn_i) = Meaning ± Synonym ± Antonym
        """
        rt = 0.0
        # If the concept ITSELF is present/matched as a base, we might start with base_rt
        # But the formula says Rt(BCn) = Meaning +/- ...
        # The base_rt seems to be a cap or a reference.
        # Let's follow the formula: "Rt(BCn_i) = Meaning +/- Synonym +/- Antonym"
        # However, the prompt says "BCnMeaning: +/- 15.000 Rt", etc.
        # It also lists "BCnBase = +35.000" in the example.
        # In the example: "BCn_1 ... BCn_Base = +35.000".
        # This suggests the Base Rt is the starting point if the concept is identified.

        rt += self.base_rt

        if self.matched_meaning:
            rt += self.MEANING_WEIGHT
        if self.matched_synonym:
            rt += self.SYNONYM_WEIGHT
        if self.matched_antonym:
            rt += self.ANTONYM_WEIGHT

        return rt


@dataclass
class Subconcept:
    """
    The Subconcept (SCn) - The Contextual Bridge
    Nuanced branches of the main concept.
    Range: +150.000 Rt to -150.000 Rt (Base)
    """
    name: str
    base_rt: float = 100.0 # Default from example (Superposition = 100)

    betaconcepts: List[Betaconcept] = field(default_factory=list)

    # Matches in query
    matched_meaning: bool = False
    matched_synonym: bool = False
    matched_antonym: bool = False

    # Overlap state
    overlap_triggered: bool = False
    overlap_value: float = 0.0

    # Constants
    OVERLAP_MULTIPLIER: float = 1.5

    # Weights for SCn matches (inferred from description of Components)
    SCN_MEANING_WEIGHT: float = 60.0
    SCN_SYNONYM_WEIGHT: float = 40.0
    SCN_ANTONYM_WEIGHT: float = -40.0

    def add_betaconcept(self, bcn: Betaconcept):
        self.betaconcepts.append(bcn)

    def calculate_rt(self, semantic_engine: SemanticEngine) -> Tuple[float, float, float]:
        """
        Rt(SCn_j) = SCn_Base + Ov_calculated + Sum(Rt(BCn)_related)

        Returns:
            (Total SCn Rt, Component Y contribution, Component Z contribution)
            Where Y = SCn_Base + Ov + (Components: Meaning/Synonym/Antonym)
            Where Z = Sum(Rt(BCn))
        """
        # 1. Calculate Overlap (Ov)
        self.overlap_triggered = False
        self.overlap_value = 0.0

        for bcn in self.betaconcepts:
            if isinstance(semantic_engine, DictionarySemanticEngine):
                bcn_synonyms = semantic_engine.synonyms.get(bcn.name.lower(), set())
                bcn_synonyms.add(bcn.name.lower()) # Include self

                scn_meanings = semantic_engine.meanings.get(self.name.lower(), set())

                # Intersection?
                if not bcn_synonyms.isdisjoint(scn_meanings):
                     self.overlap_triggered = True

                scn_synonyms = semantic_engine.synonyms.get(self.name.lower(), set())

                # Check if BCn name is in SCn Synonyms?
                if not bcn_synonyms.isdisjoint(scn_synonyms):
                    self.overlap_triggered = True

            else:
                # Fallback for generic engine: check direct synonym/meaning
                if semantic_engine.is_synonym(bcn.name, self.name) or \
                   semantic_engine.is_meaning(self.name, bcn.name):
                    self.overlap_triggered = True

            if self.overlap_triggered:
                break

        if self.overlap_triggered:
            self.overlap_value = self.OVERLAP_MULTIPLIER * self.base_rt

        # 2. Aggregation
        sum_rt_bcn = sum(bcn.calculate_rt() for bcn in self.betaconcepts)

        # Calculate Base + Components
        rt_y_component = self.base_rt + self.overlap_value

        if self.matched_meaning:
            rt_y_component += self.SCN_MEANING_WEIGHT
        if self.matched_synonym:
            rt_y_component += self.SCN_SYNONYM_WEIGHT
        if self.matched_antonym:
            rt_y_component += self.SCN_ANTONYM_WEIGHT

        rt_total = rt_y_component + sum_rt_bcn

        return rt_total, rt_y_component, sum_rt_bcn


@dataclass
class Concept:
    """
    The Concept (Cn) - The Macro Anchor
    The central idea being analyzed.
    Range: +1000.000 Rt to -1000.000 Rt
    """
    name: str
    base_rt: float = 500.0 # Initial default from example

    subconcepts: List[Subconcept] = field(default_factory=list)

    # Matches in query (Macro level)
    matched_meaning: bool = False
    matched_synonym: bool = False
    matched_antonym: bool = False

    CN_MEANING_WEIGHT: float = 400.0
    CN_SYNONYM_WEIGHT: float = 300.0
    CN_ANTONYM_WEIGHT: float = -300.0

    def add_subconcept(self, scn: Subconcept):
        self.subconcepts.append(scn)

    def calculate_rt(self, semantic_engine: SemanticEngine) -> Tuple[float, float, float, float]:
        """
        Rt(Total) = Rt(Cn) + Sum(Rt(SCn))

        Returns:
            (Total Rt, X, Y, Z) coordinates for the sphere.
        """
        # Rt(Cn)
        rt_cn = self.base_rt
        if self.matched_meaning:
            rt_cn += self.CN_MEANING_WEIGHT
        if self.matched_synonym:
            rt_cn += self.CN_SYNONYM_WEIGHT
        if self.matched_antonym:
            rt_cn += self.CN_ANTONYM_WEIGHT

        x = rt_cn
        y_total = 0.0
        z_total = 0.0

        rt_sub_sum = 0.0

        for scn in self.subconcepts:
            rt_scn, y_comp, z_comp = scn.calculate_rt(semantic_engine)
            rt_sub_sum += rt_scn
            y_total += y_comp
            z_total += z_comp

        total_rt = rt_cn + rt_sub_sum

        return total_rt, x, y_total, z_total


class HSWS:
    """
    Symbo-Logic Hierarchical Semantic Weighting System (HSWS) Engine.
    Orchestrates the calculation of semantic alignment.
    """

    def __init__(self, semantic_engine: SemanticEngine):
        self.semantic_engine = semantic_engine

    def process(self, concept: Concept, query: str) -> Dict[str, Any]:
        """
        Process a Main Concept against a user query using the HSWS logic.
        """
        query_lower = query.lower()

        # 1. Analyze Concept (Macro) matches - Simplistic token search
        if concept.name.lower() in query_lower:
             concept.matched_meaning = True

        # This naive implementation assumes exact substring match for "meaning".
        # For a full implementation, the SemanticEngine would tokenize and match.

        # We also need to traverse the tree to set matched flags on SCn and BCn
        # based on the query, if not already set.
        # This allows the HSWS to be more autonomous.

        for scn in concept.subconcepts:
            if scn.name.lower() in query_lower:
                scn.matched_meaning = True

            for bcn in scn.betaconcepts:
                if bcn.name.lower() in query_lower:
                    bcn.matched_meaning = True

                # Check known synonyms of BCn in query?
                # If we have synonyms registered in SemanticEngine, we could check them.
                if isinstance(self.semantic_engine, DictionarySemanticEngine):
                    syns = self.semantic_engine.synonyms.get(bcn.name.lower(), set())
                    for s in syns:
                        if s.lower() in query_lower:
                            bcn.matched_synonym = True

        # Rt Calculation
        total_rt, x, y, z = concept.calculate_rt(self.semantic_engine)

        # Result Interpretation
        interpretation = "Weak/Undefined"
        if total_rt > 3000.0:
            interpretation = "Absolute Truth"
        elif total_rt > 2000.0:
            interpretation = "Strong Plausibility"
        elif total_rt < 0:
            interpretation = "Contradiction/Falsehood"

        return {
            "total_rt": total_rt,
            "coordinates": {"x": x, "y": y, "z": z},
            "interpretation": interpretation,
            "concept_name": concept.name
        }
