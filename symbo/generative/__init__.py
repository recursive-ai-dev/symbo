# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Generative symbolic cores: Taylor-manifold policy function construction."""

from .taylor import PolicyFunction, TaylorExpansion, generate_multivariate_taylor

__all__ = ["PolicyFunction", "TaylorExpansion", "generate_multivariate_taylor"]
