# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Command line entry point for Symbo.

Usage
-----
    python -m symbo                 # run the full demo pipeline
    python -m symbo --repl          # pipeline, then an interactive REPL
    python -m symbo --check         # environment / optional-dependency report
    python -m symbo --demo rbc      # a single demo (rbc | kamke | bench)
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

import symbo


def _report_environment() -> int:
    """Print version, backend and optional-dependency status. Returns exit code."""
    from symbo._optional import EXTRA_FOR_PACKAGE, is_available

    print(f"symbo {symbo.__version__} ({symbo.__organization__})")
    print(f"python  {sys.version.split()[0]}")
    for pkg in ("sympy", "numpy", "networkx"):
        try:
            mod = __import__(pkg)
            print(f"{pkg:<12} {getattr(mod, '__version__', '?'):<10} required")
        except ImportError:  # pragma: no cover - only on a broken install
            print(f"{pkg:<12} {'missing':<10} REQUIRED - pip install 'symbo[{pkg}]'")
            return 1

    print("\noptional features:")
    for pkg in sorted(EXTRA_FOR_PACKAGE):
        top = pkg.split(".")[0]
        if top in ("sympy", "numpy", "networkx"):
            continue
        state = "available" if is_available(pkg) else "missing"
        print(f"  {pkg:<24} {state:<10} extra: {EXTRA_FOR_PACKAGE.get(pkg, '-')}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments and dispatch to the requested entry point."""
    parser = argparse.ArgumentParser(
        prog="symbo",
        description="Symbo - hybrid generative symbolic reasoning engine",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true",
                       help="print an environment / feature report and exit")
    group.add_argument("--repl", action="store_true",
                       help="run the demo pipeline, then open the interactive REPL")
    group.add_argument("--demo", choices=("rbc", "kamke", "bench", "pipeline"),
                       default="pipeline", help="run a single demo")
    group.add_argument("--dashboard", action="store_true",
                       help="launch the Streamlit dashboard (needs the dashboard extra)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="log at DEBUG level instead of WARNING")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from symbo import demos

    if args.check:
        return _report_environment()

    if args.dashboard:
        demos.streamlit_dashboard()
        return 0

    if args.demo == "rbc":
        demos.demo_rbc_perturbation()
    elif args.demo == "kamke":
        demos.demo_kamke_ade()
    elif args.demo == "bench":
        demos.benchmark_performance()
    else:
        trainer = demos.run_full_pipeline()
        if args.repl:
            demos.start_repl(trainer.nt, trainer)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
