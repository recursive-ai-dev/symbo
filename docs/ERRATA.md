# Errata for the `docs/` PDFs

The four customer-facing PDFs in this directory predate the 0.2.x API and
several correctness fixes. Treat them as historical. The living contract is
`README.md`, `MODULES.md`, and the docstrings.

| Claim in the PDFs | Correction |
|---|---|
| Second-order perturbation "reproduced closed-form coefficient values found in traditional macroeconomic solvers" | The original implementation set `c_{t+1} = c_t` (a no-op substitution). The engine now solves the quadratic first-order system with a Blanchard–Kahn filter and checks residuals on the *raw* equations. See `symbo.analytics.perturbation`. |
| Production `y = α A k^α` in the RBC demo | Typo. Output is `A k^α`. Steady-state consumption is `c* ≈ 1.483`, not `0.088`. |
| A* finds "the optimal path" with a raw Manhattan heuristic over a value landscape | Admissible only when cell costs are non-negative and the heuristic is scaled by `min(Z)`. Negative landscapes use SPFA/Bellman-Ford. |
| `groebner_solve` "finds **every** possible exact solution" | Real solutions of a *zero-dimensional* system. Inconsistent and positive-dimensional ideals raise. |
| Kamke ADE demo "successfully generated parametric equations" | The old output still contained `yp`. The demo now parametrizes the folium and derives a true `(x(t), y(t))` for the ADE. |
| "compiled to WebAssembly" / "portable browser-side inference" | JSON/msgpack-serializable *interfaces*. No `.wasm` artifact is produced. |
| "symbolic differentiation is approximately ten times slower than NumPy" | The bundled benchmark compares exact symbolic diff to `np.gradient` on different data and measures hundreds-to-thousands× here. |
| "318 classical algorithms" | Unverifiable. See `algorithm-corpus.md`. |
| `from symbo.symbo import NanoTensor` | `from symbo import NanoTensor`. |

`The-Symbo-Owner's-Manual.pdf` §5.1 / Blueprint 6.1 presented
`SymbolicTrainer.fit(data, method='perturbation')` as the entry point for a
full second-order analysis. That path used the untested `full_perturbation`
helper. `fit(method='perturbation')` now accepts a complete model tuple
`(equations, state_vars, control_vars, shock_vars, params)` and routes it to
`SecondOrderPerturbation`. The single-residual form is deprecated and returns
`False` if the helper logs a solver failure.
