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
Executable demos, benchmarks and the interactive REPL
======================================================

These entry points used to live at the bottom of the single-file prototype
``symbo.py``. They keep working, but now as an explicit module so that the core
engine (``symbo.nanotensor``) carries no plotting or dashboard dependencies.

Available entry points
----------------------
``demo_rbc_perturbation()``
    Second-order perturbation of an RBC-style model, printing the policy
    coefficients and (when matplotlib is installed) a contour plot.
``demo_kamke_ade()``
    Algebraic differential equation example using curve parametrization.
``benchmark_performance()``
    Symbolic vs numeric differentiation timing.
``run_full_pipeline()``
    Everything above, plus a knowledge-base round trip.
``start_repl(nt, trainer)``
    Python REPL preloaded with the fitted tensor and trainer.
``streamlit_dashboard()``
    Optional Streamlit UI (requires the ``dashboard`` extra).

Running them
------------
    python -m symbo                  # full pipeline, non-interactive
    python -m symbo --repl           # full pipeline, then open the REPL
    python -m symbo.demos --pipeline # same as ``python -m symbo``
"""

from __future__ import annotations

import logging
import time
from code import InteractiveConsole
from typing import Dict, Optional

import numpy as np
import sympy as sp

from symbo.nanotensor import KnowledgeBase, NanoTensor, SymbolicTrainer

logger = logging.getLogger("symbo.demos")


def demo_rbc_perturbation(ss_guess: Optional[Dict[str, float]] = None,
                          plot: bool = True,
                          order: int = 2):
    """
    Full RBC model demo: a second-order perturbation solution of a real model.

    The economy is the textbook RBC model with log-linear technology

        y_t = exp(a_t) k_t^alpha + (1 - delta) k_t - c_t,     a_{t+1} = rho a_t

    and CRRA-free log utility, i.e. the Euler equation

        c_t^(-1) = beta c_{t+1}^(-1) (alpha exp(a_{t+1}) k_{t+1}^(alpha-1) + 1 - delta).

    Everything is solved by :class:`~symbo.analytics.perturbation.SecondOrderPerturbation`
    -- the verified engine in this package -- and the resulting policy is installed
    in a `NanoTensor` so it can be evaluated, plotted and queried like any other
    fitted tensor.

    Parameters
    ----------
    ss_guess : dict, optional
        Initial guess for the steady state, keyed by ``'k'``, ``'c'``, ``'a'`` (or
        by the symbols themselves). Omit it to let the solver solve exactly.
    plot : bool
        Draw the policy over ``(k, a)`` when matplotlib is available.
    order : {1, 2}
        Perturbation order.

    Returns
    -------
    SymbolicTrainer
        Trainer around a scalar tensor holding the fitted ``k_next`` policy; its
        ``fitted_coeffs`` are the ``g_*`` derivatives of both orders.
    """
    import sympy as sp

    from symbo.analytics.perturbation import SecondOrderPerturbation

    print("\n=== RBC Perturbation Demo ===")

    alpha, beta, delta, rho = sp.symbols('alpha beta delta rho')
    k, c, a = sp.symbols('k c a')
    kp, cp, ap = sp.symbols('k_next c_next a_next')

    params = {alpha: 0.36, beta: 0.99, delta: 0.08, rho: 0.9}
    equations = [
        c ** (-1) - beta * cp ** (-1) * (alpha * sp.exp(ap) * kp ** (alpha - 1) + 1 - delta),
        sp.exp(a) * k ** alpha + (1 - delta) * k - c - kp,
        ap - rho * a,
    ]

    solver = SecondOrderPerturbation(equations, [k], [c], [a], params,
                                     shock_persistence={a: rho})
    guess = None
    if ss_guess:
        by_name = {str(sym): sym for sym in (k, c, a)}
        guess = {by_name[str(name)]: float(value) for name, value in ss_guess.items()
                 if str(name) in by_name} or None

    start = time.time()
    steady_state = solver.compute_steady_state(guess)
    solution = solver.solve(order=order, verify_at=1e-3)
    elapsed = time.time() - start

    print("Steady state: " + ", ".join(f"{sym} = {steady_state[sym]:.6f}"
                                      for sym in (k, c, a)))
    print(f"Perturbation (order {order}): "
          f"{'✅ converged' if solution.converged else '❌ not converged'} ({elapsed:.2f}s)")
    diagnostics = solution.diagnostics
    print(f"Determined system: {diagnostics['determined']} "
          f"({diagnostics['n_effective_equations']} equations for "
          f"{diagnostics['n_endogenous']} endogenous variables)")
    residual = diagnostics.get('residual_h0.001')
    if residual is not None:
        print(f"Equation residual at h=1e-3: {residual:.3e}")

    print("Fitted coefficients:")
    for name, value in sorted(solution.coefficients.items()):
        print(f"  {name} = {value:.6f}")

    # Install the capital policy in a NanoTensor so the rest of the library
    # (predict / plot / serialise) works on the demo's result. The policy maps
    # *levels* to next-period levels, exactly as `evaluate_policy` does.
    nt = NanoTensor((1,), max_order=order, base_vars=['k', 'a'])
    nt.data.flat[0] = solution.policy_expression('k')
    trainer = SymbolicTrainer(nt)
    trainer.fitted_coeffs.update({name: float(value)
                                 for name, value in solution.coefficients.items()})

    test_state = {'k': 9.0, 'a': 0.05}
    prediction = trainer.predict(test_state)
    print(f"\nPolicy at {test_state}: k' = {float(prediction.flat[0]):.6f}")

    if plot:
        try:
            import matplotlib.pyplot as plt

            k_ss = float(steady_state[k])
            nt.plot_contour('k', 'a', levels=12,
                           range1=(k_ss - 1.0, k_ss + 1.0),
                           range2=(-0.1, 0.1), n=40)
            plt.show()      # a demo should actually open a window when it can
        except ImportError as exc:      # MissingOptionalDependency is an ImportError
            logger.info("skipping policy contour: %s", exc)

    return trainer


def demo_kamke_ade():
    """
    Curve parametrization, then a genuine ADE parametrization.

    1. The folium of Descartes ``x³ + y³ − 3xy = 0`` is singular at the origin,
       so :meth:`NanoTensor.parametrize_curve` returns the textbook rational
       map ``(3t/(1+t³), 3t²/(1+t³))``.
    2. The Kamke ADE ``(y')² + 3 y' − 2 y − 3 x = 0`` is *not* a plane curve
       in ``(x, y)``. Setting ``y' = t`` and differentiating the resulting
       relation eliminates ``x`` and yields a true parametric pair
       ``(x(t), y(t))`` with no leftover derivative.
    """
    print("\n=== Kamke ADE Demo ===")

    x, y, t = sp.symbols('x y t')
    nt = NanoTensor((1,))
    folium = x**3 + y**3 - 3 * x * y
    x_param, y_param = nt.parametrize_curve(folium, t=t)
    print(f"Folium parametrization: (x(t), y(t)) = ({x_param}, {y_param})")

    # ADE: y'^2 + 3 y' - 2 y - 3 x = 0. Set y' = t:
    #   y = (t**2 + 3*t - 3*x)/2
    # Differentiate w.r.t. the parameter, using dy/dx = t:
    #   t x' = (2t + 3 - 3 x')/2  =>  x'(2t + 3) = 2t + 3
    # so x' = 1 (t != -3/2) and x = t + C.
    C = sp.Symbol('C')
    x_ade = t + C
    y_ade = sp.simplify((t**2 + 3 * t - 3 * x_ade) / 2)
    print(f"ADE parametrization: (x(t), y(t)) = ({x_ade}, {y_ade})")
    print("  (parameter t is y'; C is an integration constant)")

    return x_param, y_param

def benchmark_performance():
    """
    Time symbolic vs numeric differentiation. **Not a like-for-like comparison.**

    Times ``NanoTensor.diff_cached`` on a 100x100 tensor of default (mostly
    zero) expressions against ``numpy.gradient`` on random floats: an exact
    symbolic derivative versus a finite difference, on different data. The
    ratio is hardware- and expression-dependent (hundreds-to-thousands x on
    a typical laptop) and is printed as a measurement, not a product claim.
    """
    print("\n=== Performance Benchmark ===")

    # Symbolic method
    nt_sym = NanoTensor((100, 100), max_order=2)
    start = time.time()
    _ = nt_sym.diff_cached('k', 1)
    sym_time = time.time() - start

    # Numeric method (numpy)
    nt_num = np.random.randn(100, 100)
    start = time.time()
    _ = np.gradient(nt_num)
    num_time = time.time() - start

    print(f"Symbolic diff: {sym_time:.4f}s")
    print(f"Numeric diff: {num_time:.4f}s")
    print(f"Speed ratio: {sym_time/num_time:.2f}x "
          "(symbolic is slower; this is exact diff vs finite difference, not autodiff)")

    return sym_time, num_time

def start_repl(nt: NanoTensor, trainer: Optional[SymbolicTrainer] = None):
    """
    Launch an interactive symbolic REPL.

    Provides an `InteractiveConsole` preloaded with key symbols:

    - nt      : NanoTensor instance
    - trainer : associated SymbolicTrainer
    - sp      : SymPy module
    - solve   : SymPy solve
    - symbols : SymPy symbols
    - plot    : convenience alias to `nt.plot_contour`

    Intended for exploratory work and quick experiments.
    """

    locals_dict = {
        'nt': nt, 'trainer': trainer, 'sp': sp, 'solve': sp.solve,
        'symbols': sp.symbols, 'plot': nt.plot_contour
    }
    con = InteractiveConsole(locals_dict)
    banner = """
    Symbolic AI REPL
    ================
    Commands:
    - nt.eval_numeric({'k':1.1})
    - trainer.fit(data, 'symbolic')
    - solve(R, k)
    - nt.plot_contour('k', 'eps')
    """
    print(banner)
    con.interact()

def streamlit_dashboard():
    """
    Launch a Streamlit dashboard for interactive exploration.

    Provides:

    - sliders for the two arguments of the capital policy (``k``, ``a``),
    - a button that runs the RBC perturbation demo (cached: the solve costs
      several seconds and Streamlit reruns the script on every widget change),
    - the policy function ``k'`` plotted against capital, with the steady state
      marked,
    - and a JSON view of the fitted ``g_*`` coefficients plus the residual check.

    To run, execute::

        python -m symbo --dashboard

    or, from an installed environment, ``streamlit run symbo/demos.py`` with
    this function exposed as the main entry point.
    """
    from symbo._optional import require

    st = require("streamlit", "streamlit_dashboard")
    plt = require("matplotlib.pyplot", "streamlit_dashboard")

    st.title("Symbo — RBC policy functions")
    st.caption("Second-order perturbation of a real RBC model, solved symbolically.")

    order = st.sidebar.select_slider("Perturbation order", options=[1, 2], value=2)

    @st.cache_resource(show_spinner="Computing the perturbation solution...")
    def _solve(perturbation_order: int):
        return demo_rbc_perturbation(plot=False, order=perturbation_order)

    trainer = _solve(int(order))
    tensor = trainer.nt
    k_ss = 8.708784901731187  # steady-state capital of the demo's parameterisation

    k = st.sidebar.slider("Capital k (levels)", 7.0, 10.5, float(k_ss), 0.01)
    a = st.sidebar.slider("Log productivity shock a", -0.10, 0.10, 0.0, 0.005)

    policy = float(tensor.eval_numeric({'k': k, 'a': a}).flat[0])
    st.subheader("Policy evaluation")
    st.metric(label="k' (capital next period)", value=f"{policy:.5f}",
              delta=f"{policy - k_ss:+.5f} vs steady state")

    st.subheader("Policy function")
    ks = np.linspace(7.0, 10.5, 60)
    curve = [float(tensor.eval_numeric({'k': float(kk), 'a': a}).flat[0]) for kk in ks]
    fig, ax = plt.subplots()
    ax.plot(ks, curve, label=f"k' at a = {a:+.3f}")
    ax.plot(ks, ks, linestyle="--", linewidth=1, label="k' = k (no transition)")
    ax.axvline(k_ss, color="grey", linewidth=1, label="steady state")
    ax.set_xlabel("k today")
    ax.set_ylabel("k next period")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

    with st.expander("Fitted coefficients and accuracy"):
        st.json(trainer.fitted_coeffs)
        st.write(
            "Residuals are measured by substituting the policy back into the "
            "model equations at a small shock scale; see "
            "`SecondOrderPerturbation.solve(verify_at=...)`."
        )

def run_full_pipeline():
    """
    Execute the full Symbo demo pipeline.

    The pipeline includes:

    1. RBC perturbation demo,
    2. Kamke ADE demo,
    3. performance benchmark,
    4. population of a KnowledgeBase with fitted RBC policy coefficients.

    Returns
    -------
    SymbolicTrainer
        Trainer instance from the RBC perturbation demo (for further use).
    """
    print("Starting NanoTensor Symbolic AI Pipeline...")

    # Demo 1: RBC perturbation
    trainer = demo_rbc_perturbation()

    # Demo 2: Kamke ODE
    demo_kamke_ade()

    # Demo 3: Benchmarks
    benchmark_performance()

    # Knowledge base demo
    kb = KnowledgeBase()
    for name, value in trainer.fitted_coeffs.items():
        kb.add_fact('rbc_policy', name, value)

    print("\nKnowledge Base Query:")
    if trainer.fitted_coeffs:
        first = sorted(trainer.fitted_coeffs)[0]
        print(f"  rbc_policy / {first} -> {kb.query('rbc_policy', first)}")
    print(f"  facts stored: {len(kb.query('rbc_policy'))}")

    print("\n✅ All demos completed successfully!")
    return trainer


if __name__ == "__main__":  # pragma: no cover - Streamlit entry point
    # ``streamlit run symbo/demos.py`` executes this module as __main__.
    streamlit_dashboard()
