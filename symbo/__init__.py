# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Symbo — Nano-scale Hybrid Generative Symbolic Engine
====================================================

A modular symbolic-numeric reasoning system.
"""

__version__ = "0.1.0"
__author__ = "Damien Davison & Michael Maillet & Sacha Davison"
__organization__ = "Recursive AI Devs"

# Core primitives and tensor
from .primitives import AtomicPrimitives, add, mul, diff
from .tensor import SymbolicTensor
from .reasoning.hsws import HSWS, Concept, Subconcept, Betaconcept, DictionarySemanticEngine, SemanticEngine

__all__ = [
    'AtomicPrimitives',
    'SymbolicTensor',
    'add',
    'mul',
    'diff',
    'HSWS',
    'Concept',
    'Subconcept',
    'Betaconcept',
    'DictionarySemanticEngine',
    'SemanticEngine'
]
