# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""High-speed serialization (MessagePack / Arrow) for Symbo objects."""

from .serialization import (
    ArrowTableBuilder,
    SerializationError,
    SymboSerializer,
)

__all__ = ["ArrowTableBuilder", "SerializationError", "SymboSerializer"]
