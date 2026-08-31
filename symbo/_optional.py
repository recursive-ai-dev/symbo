# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Optional dependency helpers
===========================

Symbo's core engine only needs :mod:`sympy` and :mod:`numpy` (plus
:mod:`networkx` for the analytics graph). Everything else accelerates or extends
a *specific* feature, and the import policy is deliberately split in two:

* the heavyweight interactive backends -- :mod:`torch`, :mod:`matplotlib`,
  :mod:`plotly`, :mod:`skopt`, :mod:`streamlit`, :mod:`kanren` -- are imported
  *inside* the function that needs them, so that ``import symbo`` stays fast and
  a plain install never touches them;
* the small serialisation codecs -- :mod:`msgpack`, :mod:`pyarrow`, :mod:`dill`
  -- are probed *once* at module scope by the modules that use them and kept as
  a module-level ``None`` when absent. Those modules then guard every entry point
  with ``if codec is None``, which is the price of a single uniform check per
  module rather than one per function.

Either way the failure mode is the same: a missing optional package produces one
clear, actionable error instead of an ``ImportError`` at ``import symbo``.

Examples
--------
>>> from symbo._optional import optional_module, require
>>> optional_module("definitely_not_installed") is None
True
>>> require("numpy")  # doctest: +SKIP
<module 'numpy' ...>
"""

from __future__ import annotations

import importlib
from typing import Any, Optional

#: Maps an importable module name to the pip extra that provides it.
EXTRA_FOR_PACKAGE: dict[str, str] = {
    "torch": "neuro",
    "matplotlib": "viz",
    "matplotlib.pyplot": "viz",
    "plotly": "viz",
    "plotly.graph_objects": "viz",
    "plotly.subplots": "viz",
    "skopt": "opt",
    "networkx": "kb",
    "kanren": "kb",
    "msgpack": "io",
    "pyarrow": "io",
    "dill": "io",
    "streamlit": "dashboard",
}


def optional_module(name: str) -> Optional[Any]:
    """
    Import ``name`` and return the module, or ``None`` if it is not installed.

    Never raises. Use :func:`require` when the caller genuinely needs the
    package to proceed.
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def is_available(name: str) -> bool:
    """Return ``True`` when the module ``name`` can be imported."""
    return optional_module(name) is not None


def require(name: str, feature: Optional[str] = None) -> Any:
    """
    Import ``name`` or raise :class:`MissingOptionalDependency`.

    Parameters
    ----------
    name:
        Dotted module name, e.g. ``"torch"`` or ``"matplotlib.pyplot"``.
    feature:
        Human readable name of the Symbo feature that needs the package. It is
        included in the error message so users know what they were trying to do.

    Returns
    -------
    module
    """
    module = optional_module(name)
    if module is None:
        raise MissingOptionalDependency(name, feature)
    return module


class MissingOptionalDependency(ImportError):
    """Raised when a lazily-imported optional package is needed but missing."""

    def __init__(self, name: str, feature: Optional[str] = None):
        extra = EXTRA_FOR_PACKAGE.get(name)
        install = f"pip install 'symbo[{extra}]'" if extra else f"pip install {name}"
        what = f" for {feature}()" if feature else ""
        super().__init__(
            f"The optional package '{name}' is required{what}. "
            f"Install it with: {install}"
        )
        # ``ImportError.__init__`` resets ``name``/``path``, so these have to be
        # assigned after the super().__init__ call.
        self.name = name
        self.feature = feature


__all__ = [
    "EXTRA_FOR_PACKAGE",
    "MissingOptionalDependency",
    "is_available",
    "optional_module",
    "require",
]
