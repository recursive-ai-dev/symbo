#!/bin/bash
# Builds standalone desktop binaries for the Symbo Engine and its demos.

echo "Installing build requirements..."
pip install pyinstaller

echo "Building Military Grade Demo Binary..."
pyinstaller --onefile --name symbo_demo \
    --add-data "symbo.py:." \
    --hidden-import "symbo" \
    --exclude-module streamlit \
    --exclude-module plotly \
    demo_military_grade.py

echo "Building Stress Test Binary..."
pyinstaller --onefile --name symbo_stress_test \
    --add-data "symbo.py:." \
    --hidden-import "symbo" \
    --exclude-module streamlit \
    --exclude-module plotly \
    stress_test.py

echo "Build complete. Executables are located in the dist/ directory."
