#!/bin/bash
# Build standalone desktop binaries for the Symbo engine and its demos.
#
# Symbo is a package (see pyproject.toml), so the frozen executables are built
# against the *installed* package rather than a loose symbo.py file.
set -euo pipefail

cd "$(dirname "$0")"

echo "Installing build requirements..."
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[viz,io]'
python3 -m pip install pyinstaller

build_binary () {
    local name="$1"; shift
    local entry="$1"; shift
    echo "Building ${name} from ${entry}..."
    python3 -m PyInstaller --onefile --clean --name "${name}" \
        --collect-submodules symbo \
        --exclude-module streamlit \
        --exclude-module plotly \
        --exclude-module torch \
        "$@" \
        "${entry}"
}

build_binary symbo_demo demo_military_grade.py
build_binary symbo_stress_test stress_test.py

# The CLI entry point (python -m symbo) is frozen as its own executable.
build_binary symbo-cli symbo/__main__.py --hidden-import symbo.demos

echo
echo "Build complete. Executables are located in the dist/ directory."
echo "Verify with:  ./dist/symbo-cli --check"
