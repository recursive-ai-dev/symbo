# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Executable documentation: every ``python`` block in the docs must run.

The README, MODULES.md and the military-grade guides carry the API surface most
users read first, so their examples are treated as tests. Each block is executed in a fresh interpreter from
the repository root with a timeout; a block that prints is fine, a block that
raises is a documentation bug.

Blocks can opt out with a fence language other than ``python`` (e.g. ``text``),
which is how the intentionally-non-runnable snippets are marked.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = [
    "README.md",
    "MODULES.md",
    "README_MILITARY_GRADE.md",
    "docs/MILITARY_GRADE_NANOTENSOR.md",
]

BLOCK_RE = re.compile(r"```(?:python|py3|py)\n(.*?)```", re.DOTALL)


def _blocks(document: str):
    text = (REPO_ROOT / document).read_text(encoding="utf-8")
    # strip the leading prompt used by doctest-style blocks, if any
    for match in BLOCK_RE.finditer(text):
        block = match.group(1)
        if block.lstrip().startswith(">>>"):
            continue
        yield block.strip()


def _collect():
    cases = []
    for document in DOCS:
        for index, block in enumerate(_blocks(document)):
            cases.append((document, index, block))
    return cases


CASES = _collect()


def test_documented_blocks_were_found():
    """Guards against the regex silently matching nothing."""
    assert len(CASES) >= 10, CASES
    assert any(document == "MODULES.md" for document, _, _ in CASES)
    assert any(document == "README.md" for document, _, _ in CASES)


@pytest.mark.slow
@pytest.mark.parametrize(
    "document,index,block",
    CASES,
    ids=[f"{document}:{index}" for document, index, _ in CASES],
)
def test_block_runs(document, index, block, tmp_path):
    """Run a documentation snippet exactly as a reader would copy-paste it."""
    first_line = block.splitlines()[0] if block else ""
    try:
        result = subprocess.run(
            [sys.executable, "-c", block],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            timeout=900,
            env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin",
                 "HOME": str(tmp_path), "MPLBACKEND": "Agg"},
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - CI guard
        pytest.fail(f"{document} block {index} timed out: {first_line!r} ({exc})")
    assert result.returncode == 0, (
        f"{document} block {index} failed\n--- code ---\n{block}\n"
        f"--- stderr ---\n{result.stderr[-3000:]}"
    )
