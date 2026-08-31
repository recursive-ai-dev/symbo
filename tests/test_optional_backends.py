# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Optional backends (matplotlib, plotly, torch, skopt).

Every test here runs in *both* configurations: when the extra is installed the
happy path is exercised, when it is not, the library must refuse with a
:class:`~symbo._optional.MissingOptionalDependency` that names the right extra.
"""

import numpy as np
import pytest
import sympy as sp

from symbo import HybridTrainer, MissingOptionalDependency, NanoTensor
from symbo._optional import is_available

x0, x1 = sp.symbols("x0 x1")


@pytest.fixture()
def surface_tensor():
    nt = NanoTensor((1,), max_order=2, base_vars=["x0", "x1"])
    nt.data[0] = x0**2 + x0 * x1 + sp.sin(x1)
    return nt


@pytest.fixture()
def linear_problem():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    y = 1.5 + 2.0 * X[:, 0] - 0.7 * X[:, 1]
    return X, y


class TestPlotBackends:
    def test_plot_contour_uses_matplotlib(self, surface_tensor):
        if not is_available("matplotlib"):
            with pytest.raises(MissingOptionalDependency, match=r"symbo\[viz\]"):
                surface_tensor.plot_contour("x0", "x1")
            return
        fig = surface_tensor.plot_contour("x0", "x1", n=20)
        assert hasattr(fig, "savefig")
        assert len(fig.axes) == 1

    def test_plot_contour_honours_an_supplied_axes(self, surface_tensor):
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        returned = surface_tensor.plot_contour("x0", "x1", n=15, ax=ax)
        assert returned is ax or returned is ax.figure
        plt.close("all")

    def test_plot_surface_uses_plotly(self, surface_tensor):
        if not is_available("plotly"):
            with pytest.raises(MissingOptionalDependency, match=r"symbo\[viz\]"):
                surface_tensor.plot_surface("x0", "x1")
            return
        figure = surface_tensor.plot_surface("x0", "x1", n=15)
        assert hasattr(figure, "to_json")

    def test_plot_grid_with_path_annotates_the_found_path(self, surface_tensor):
        if not is_available("matplotlib"):
            with pytest.raises(MissingOptionalDependency, match=r"symbo\[viz\]"):
                surface_tensor.plot_grid_with_path("x0", "x1", start=(0, 0), goal=(5, 7))
            return
        fig = surface_tensor.plot_grid_with_path("x0", "x1", start=(0, 0),
                                                 goal=(5, 7), n1=12, n2=12)
        assert hasattr(fig, "savefig")


class TestTorchPathway:
    def test_make_loader_without_torch_names_the_extra(self):
        if is_available("torch"):
            pytest.skip("torch is installed; the failure path cannot be exercised")
        with pytest.raises(MissingOptionalDependency, match=r"symbo\[neuro\]"):
            HybridTrainer.make_loader(np.zeros((4, 2)), np.zeros(4))

    def test_torch_fit_without_torch_names_the_extra(self):
        if is_available("torch"):
            pytest.skip("torch is installed; the failure path cannot be exercised")
        trainer = HybridTrainer(NanoTensor((1,), max_order=1, base_vars=["x0"]))
        with pytest.raises(MissingOptionalDependency, match=r"symbo\[neuro\]"):
            trainer.torch_fit([], epochs=1)

    @pytest.mark.torch
    def test_torch_fit_recovers_a_known_linear_policy(self, linear_problem):
        torch = pytest.importorskip("torch")
        X, y = linear_problem
        nt = NanoTensor((1,), max_order=1, base_vars=["x0", "x1"])
        nt.generate_taylor({"x0": 0.0, "x1": 0.0})
        trainer = HybridTrainer(nt)
        loader = HybridTrainer.make_loader(X, y, batch_size=16, shuffle=True)
        assert isinstance(loader, torch.utils.data.DataLoader)
        fitted = trainer.torch_fit(loader, epochs=400, lr=0.05)
        assert fitted["g_x0"] == pytest.approx(2.0, abs=0.05)
        assert fitted["g_x1"] == pytest.approx(-0.7, abs=0.05)
        assert fitted["g_0"] == pytest.approx(1.5, abs=0.05)

    @pytest.mark.torch
    def test_torch_fit_rejects_bad_tensors(self, linear_problem):
        pytest.importorskip("torch")
        X, y = linear_problem
        loader = HybridTrainer.make_loader(X, y, batch_size=8)

        wide = HybridTrainer(NanoTensor((2, 2), max_order=1, base_vars=["x0", "x1"]))
        with pytest.raises(ValueError, match="scalar tensor"):
            wide.torch_fit(loader, epochs=1)

        unfitted = NanoTensor((1,), max_order=1, base_vars=["x0", "x1"])
        empty = HybridTrainer(unfitted)
        with pytest.raises(ValueError, match="coefficient symbols"):
            empty.torch_fit(loader, epochs=1)

        nt = NanoTensor((1,), max_order=1, base_vars=["x0", "x1"])
        nt.generate_taylor({"x0": 0.0, "x1": 0.0})
        trainer = HybridTrainer(nt)
        with pytest.raises(ValueError, match="var_order"):
            trainer.torch_fit(loader, epochs=1, var_order=["x0", "nope"])


class TestRegressionSearch:
    def test_symbolic_regression_returns_a_fitted_tensor(self, linear_problem):
        X, y = linear_problem
        trainer = HybridTrainer(NanoTensor((1,), base_vars=["x0", "x1"]))
        best = trainer.symbolic_regression(X, y, max_deg=2, n_calls=6)
        assert isinstance(best, NanoTensor)
        assert best.shape == (1,)
        # the docstring promises a *fitted* tensor, not a bare ansatz
        assert best.fitted_coeffs
        predictions = HybridTrainer(best).predict_batch(X)
        assert float(np.max(np.abs(predictions - y))) < 1e-6

    def test_symbolic_regression_validates_its_input(self, linear_problem):
        X, y = linear_problem
        trainer = HybridTrainer(NanoTensor((1,)))
        with pytest.raises(ValueError, match="2-D"):
            trainer.symbolic_regression(X[:, 0], y)

    def test_predict_batch_checks_the_feature_count(self, linear_problem):
        X, _y = linear_problem
        trainer = HybridTrainer(NanoTensor((1,), base_vars=["x0"]))
        with pytest.raises(ValueError, match="feature column"):
            trainer.predict_batch(X)

    def test_fallback_grid_search_is_used_without_skopt(self, linear_problem, monkeypatch):
        """No scikit-optimize: the deterministic sweep over orders must still run."""
        import symbo.nanotensor as nt_mod

        X, y = linear_problem
        trainer = HybridTrainer(NanoTensor((1,), base_vars=["x0", "x1"]))
        monkeypatch.setattr(nt_mod, "optional_module", lambda name: None)
        best = trainer.symbolic_regression(X, y, max_deg=1)
        assert float(np.max(np.abs(HybridTrainer(best).predict_batch(X) - y))) < 1e-6


class TestKnowledgeBaseBackends:
    def test_queries_work_with_and_without_kanren(self):
        from symbo import KnowledgeBase

        kb = KnowledgeBase()
        kb.add_fact("e", "p", 1.5)
        assert kb.query("e", "p") == [1.5]
        assert kb.query("e") == [("e", "p", 1.5)]
        assert kb.query("missing", "p") == []
