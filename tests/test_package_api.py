# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Packaging contract: public API surface, lazy optional dependencies, CLI."""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import symbo
import symbo.__main__
import symbo.demos
from symbo._optional import (
    EXTRA_FOR_PACKAGE,
    MissingOptionalDependency,
    is_available,
    optional_module,
    require,
)

SUBPACKAGES = [
    "symbo.analytics",
    "symbo.analytics.explain",
    "symbo.analytics.perturbation",
    "symbo.demos",
    "symbo.ecosystem",
    "symbo.generative",
    "symbo.generative.taylor",
    "symbo.io",
    "symbo.io.serialization",
    "symbo.nanotensor",
    "symbo.nano_tensor_enhanced",
    "symbo.primitives",
    "symbo.reasoning",
    "symbo.reasoning.a_star",
    "symbo.reasoning.hsws",
    "symbo.security",
    "symbo.solver",
    "symbo.solver.groebner",
    "symbo.tensor",
    "symbo.wasm_bindings",
]


class TestPublicAPI:
    @pytest.mark.parametrize("module", SUBPACKAGES)
    def test_submodule_imports(self, module):
        assert importlib.import_module(module) is not None

    def test_every_exported_name_resolves(self):
        for name in symbo.__all__:
            assert hasattr(symbo, name), f"symbo.{name} is exported but missing"

    @pytest.mark.parametrize("name", ["NanoTensor", "SymbolicTensor", "TaylorExpansion",
                                     "SecondOrderPerturbation", "WASMInterface", "safe_sympify"])
    def test_headline_names(self, name):
        assert name in symbo.__all__

    def test_all_has_no_duplicates(self):
        assert len(symbo.__all__) == len(set(symbo.__all__))

    def test_version_matches_the_distribution_metadata(self):
        from importlib.metadata import version

        assert symbo.__version__ == version("symbo")
        assert symbo.__version__.count(".") == 2

    def test_metadata(self):
        assert symbo.__author__
        assert symbo.__organization__

    def test_core_import_is_lazy_about_heavy_packages(self):
        """``import symbo`` must not pull in the heavyweight optional backends.

        The small serialisation codecs (msgpack/pyarrow/dill) *are* probed at
        module scope -- see :mod:`symbo._optional` -- so they are excluded here.
        """
        code = (
            "import symbo, sys, json;"
            "print(json.dumps({m: m in sys.modules for m in "
            "['torch', 'streamlit', 'skopt', 'plotly', 'matplotlib', 'kanren']}))"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             check=True).stdout
        loaded = json.loads(out)
        assert not any(loaded.values()), loaded

    @pytest.mark.parametrize("module", SUBPACKAGES)
    def test_subpackages_are_reachable_as_attributes(self, module):
        """``import symbo.tensor`` binds ``symbo.tensor`` for the whole session."""
        importlib.import_module(module)
        attribute = module[len("symbo."):].split(".")[0]
        assert hasattr(symbo, attribute)


class TestOptionalHelpers:
    def test_optional_module_never_raises(self):
        assert optional_module("definitely_not_installed_xyz") is None
        assert optional_module("numpy") is not None

    def test_is_available(self):
        assert is_available("sympy") is True
        assert is_available("definitely_not_installed_xyz") is False

    def test_require_returns_a_module(self):
        assert require("numpy").__name__ == "numpy"

    def test_require_names_the_extra_to_install(self):
        with pytest.raises(MissingOptionalDependency, match=r"pip install 'symbo\[neuro\]'"):
            require("torch", "SymbolicTrainer.torch_fit")

    def test_missing_dependency_without_a_feature(self):
        with pytest.raises(MissingOptionalDependency, match="pip install definitely_not"):
            require("definitely_not_installed_xyz")

    def test_missing_dependency_is_an_importerror(self):
        assert issubclass(MissingOptionalDependency, ImportError)
        with pytest.raises(ImportError):
            require("definitely_not_installed_xyz")

    def test_error_object_carries_context(self):
        try:
            require("definitely_not_installed_xyz", "feature")
        except MissingOptionalDependency as e:
            assert e.name == "definitely_not_installed_xyz"
            assert e.feature == "feature"

    def test_every_known_package_maps_to_a_declared_extra(self):
        extras = {"neuro", "viz", "io", "opt", "kb", "dashboard", "dev"}
        for package, extra in EXTRA_FOR_PACKAGE.items():
            assert package and extra in extras, package
        # the mapping is keyed by importable module name, not by distribution
        assert "torch" in EXTRA_FOR_PACKAGE
        assert EXTRA_FOR_PACKAGE["matplotlib.pyplot"] == "viz"


class TestDemos:
    def test_benchmark_returns_both_timings(self, capsys):
        sym_time, num_time = symbo.demos.benchmark_performance()
        assert sym_time > 0 and num_time > 0
        out = capsys.readouterr().out
        assert "Performance Benchmark" in out
        assert "Speed ratio" in out

    def test_kamke_demo_reports_a_solution(self, capsys):
        result = symbo.demos.demo_kamke_ade()
        out = capsys.readouterr().out
        assert result is not None
        assert "Kamke" in out or "ADE" in out or "ODE" in out

    @pytest.mark.slow
    def test_rbc_demo_fits_a_policy(self, capsys):
        """``plot=False`` keeps the demo free of matplotlib."""
        trainer = symbo.demos.demo_rbc_perturbation(plot=False)
        coeffs = trainer.fitted_coeffs
        assert coeffs, "the demo must fit at least one coefficient"
        assert {"g_k_a", "g_c_a", "g_k_k"} <= set(coeffs)
        out = capsys.readouterr().out
        assert "✅ converged" in out
        assert "Determined system: True" in out

        # the steady state is a fixed point of the policy, and a positive shock
        # raises next-period capital
        k_ss = 8.708784901731187
        at_ss = float(trainer.predict({"k": k_ss, "a": 0.0}).flat[0])
        assert at_ss == pytest.approx(k_ss, abs=1e-6)
        with_shock = float(trainer.predict({"k": k_ss, "a": 0.05}).flat[0])
        assert with_shock > at_ss
        assert np.isfinite(with_shock)

    @pytest.mark.slow
    def test_rbc_demo_first_order_is_leaner_than_second(self):
        second = symbo.demos.demo_rbc_perturbation(plot=False, order=2)
        first = symbo.demos.demo_rbc_perturbation(plot=False, order=1)
        assert len(first.fitted_coeffs) < len(second.fitted_coeffs)
        # both orders agree on the linear policy
        for name in ("g_k_a", "g_c_k"):
            assert first.fitted_coeffs[name] == pytest.approx(second.fitted_coeffs[name])

    @pytest.mark.slow
    def test_full_pipeline_runs_every_demo(self, capsys):
        trainer = symbo.demos.run_full_pipeline()
        out = capsys.readouterr().out
        assert "RBC Perturbation Demo" in out
        assert "Kamke ADE Demo" in out
        assert "Performance Benchmark" in out
        assert "All demos completed successfully" in out
        assert "facts stored: " in out
        assert trainer.fitted_coeffs

    def test_repl_evaluates_expressions_against_the_tensor(self):
        """The REPL reads stdin through an InteractiveConsole, so drive a process."""
        script = ("from symbo import NanoTensor, demos\n"
                  "demos.start_repl(NanoTensor((1,)))\n")
        proc = subprocess.run(
            [sys.executable, "-c", script, ],
            input="nt.data[0] = 1 + 1\nprint('REPL-RESULT', nt.data[0])\n",
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert "Symbolic AI REPL" in proc.stdout
        assert "REPL-RESULT 2" in proc.stdout

    def test_dashboard_requires_the_streamlit_extra(self):
        if is_available("streamlit"):
            pytest.skip("streamlit installed: launching the dashboard would block")
        with pytest.raises(MissingOptionalDependency, match=r"symbo\[dashboard\]"):
            symbo.demos.streamlit_dashboard()

    def test_dashboard_module_has_a_streamlit_entry_point(self):
        source = Path(symbo.demos.__file__).read_text()
        assert 'if __name__ == "__main__"' in source


class TestCLI:
    def test_check_reports_environment(self, capsys):
        assert symbo.__main__.main(["--check"]) == 0
        out = capsys.readouterr().out
        assert f"symbo {symbo.__version__}" in out
        assert "optional features:" in out

    def test_bench_demo(self, capsys):
        assert symbo.__main__.main(["--demo", "bench"]) == 0
        assert "Performance Benchmark" in capsys.readouterr().out

    def test_unknown_option_exits_with_usage(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            symbo.__main__.main(["--nope"])
        assert excinfo.value.code == 2
        assert "usage" in capsys.readouterr().err

    def test_check_runs_as_a_module(self):
        out = subprocess.run([sys.executable, "-m", "symbo", "--check"],
                             capture_output=True, text=True, check=True)
        assert symbo.__version__ in out.stdout

    def test_console_script_is_installed_and_runs(self):
        from importlib.metadata import distribution

        entry_points = distribution("symbo").entry_points
        scripts = [ep for ep in entry_points if ep.group == "console_scripts"]
        assert [ep.name for ep in scripts] == ["symbo"]
        assert scripts[0].value == "symbo.__main__:main"
