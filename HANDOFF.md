# Handoff — Symbo 0.2.0

Date: 2026-08-27 · Branch: `arena/01a04127-symbo` (base `main@2fe4800`)
Scope: take the repository to a production-quality, handoff-ready state — package it,
fix what was broken or half-finished, test it, document it, and wire up CI.

Everything below is *done and verified* in this branch except the items listed under
"Known limitations", which are stated as limitations rather than as finished work.

---

## 1. State in one screen

| | |
|---|---|
| Package | `symbo` 0.2.0, installable (`pyproject.toml`, setuptools), console script `symbo` |
| Import surface | `import symbo` → 50 re-exported names; `symbo.__version__` matches distribution metadata |
| Tests | `python -m pytest tests -q` → **511 passed, 3 skipped** (~2.5 min) |
| Doctests | `python -m pytest --doctest-modules symbo -q` → **26 passed, 1 skipped** |
| Docs as tests | `tests/test_docs.py` executes every ```python``` block in README.md (1) and MODULES.md (10) in a subprocess; 12 tests including the guard that both files are covered |
| Coverage | `--cov=symbo` → **83 %** overall (`nanotensor.py` 67 %, `ecosystem.py` 89 %, `tensor.py` 84 %, `demos.py` 69 % — the remainder is viz and torch branches) |
| Lint | `ruff check` (E,F,W,B,C4,SIM,RUF, line-length 110) + `pyflakes` clean over `symbo`, `tests`, and the two root scripts |
| CI | `.ci/ci.yml` (staged; `git mv` it to `.github/workflows/ci.yml`): lint · core-only matrix (py3.10, py3.13) · extras · torch (allowed failure) · wheel build + clean-venv smoke test |
| Optional deps | core needs only `sympy`, `numpy`, `networkx`; 7 extras (`io`, `viz`, `neuro`, `opt`, `kb`, `dashboard`, `all`) |

## 2. What the structure became

The 3.6k-line root module `symbo.py` **is now `symbo/nanotensor.py`** (moved with
`git mv`, so `git log --follow` keeps its history). The half-built `symbo/` package
(an empty `__init__`, a `symbo/tensor.py` that `symbo.py` imported) became the real
package:

```
symbo/
  __init__.py          public API + __version__ + lazy sub-package re-exports
  __main__.py          --check | --repl | --demo rbc|kamke|bench|pipeline | --dashboard
  _optional.py         optional_module/is_available/require + MissingOptionalDependency
  nanotensor.py        engine core (NanoTensor, trainers, KB, WASM helpers, persistence)
  nano_tensor_enhanced.py  standalone hardened copy (vendable into an agent runtime)
  tensor.py primitives.py security.py demos.py ecosystem.py wasm_bindings.py
  analytics/{perturbation,explain}.py   generative/taylor.py   solver/groebner.py
  reasoning/{a_star,hsws}.py            io/serialization.py
```

Design decisions a reviewer should know about:

* **Lazy optional imports.** `import symbo` never touches torch/matplotlib/plotly/skopt/
  streamlit/kanren (asserted by `test_core_import_is_lazy_about_heavy_packages`, which
  runs a *subprocess* so the assertion is about a fresh interpreter). The small
  serialisation codecs (msgpack/pyarrow/dill) are probed once at module scope and kept
  as `None` when absent, because every guard in those modules is written as
  `if codec is None`; the trade-off is documented in `symbo/_optional.py`.
* **Errors from optional paths are actionable.** `MissingOptionalDependency` names the
  extra to install (`pip install 'symbo[neuro]'`) and the feature that needed it.
* **`PerturbationSolution` is deliberately not a dataclass**; `coefficients` is a
  read-only merged view of `first_order` + `second_order`, while
  `policy_functions`/`risk_corrections` are keyed by *string* name (documented in
  MODULES.md because the asymmetry is inherited from the solver's internals).
* **Caches are validated against content, not identity-of-address.** Two modules had
  caches keyed on `id(expr)` / on the evaluation point alone; both are fixed (see §3).
* **`MilitaryGradeNanoTensor` stays a standalone copy** (its purpose is being
  vendorable as one file), but the relationship to `NanoTensor` is now spelled out in
  the module docstring and README_MILITARY_GRADE.md, because the duplication is
  otherwise confusing.

## 3. Bugs found and fixed

Not a style pass — each item below was a wrong answer, a crash, or a hang, and each is
covered by a test.

**Sandbox escape (security).** `safe_sympify` validated a call *name* allowlist, but
`lambdify`/`autowrap`/`dotprint`/`test`/`init_printing` were reachable, and nested
`sympify` re-entered the evaluator. Those code-generation entry points are now
denylisted and the denylist is applied to nested calls; `tests/test_security.py` (42
tests) includes the file-write probe that used to succeed.

**`import symbo` was impossible / needed the world.** `symbo/__init__.py` did not
export `NanoTensor`, so the README's first example raised `ImportError`; and the
monolithic `symbo.py` imported `torch`, `kanren`, `matplotlib` and `plotly` at module
scope (next to unguarded `heapq`-era imports), so a user with only the documented
dependencies could not import the package at all. Both fixed: the package re-exports
the public API, and every non-core backend now goes through `symbo/_optional.py`.

**Perturbation solver (the flagship module).**
* A model with fewer equations than endogenous variables silently produced
  minimum-norm coefficients → `determined` / `n_effective_equations` are now exposed as
  solver properties (`solver.determined`) *and* in `solution.diagnostics`, so an
  under-specified model is detectable before it is trusted.
* `variance=1.0` (documented in the docstring!) crashed with
  `'float' object has no attribute 'get'`; `variance` now accepts a scalar, a
  `{shock: value}` dict, or `None`, and rejects unknown shock names.
* `verify_at=(1e-3, 1e-2)` was wrapped as `[tuple]` and the resulting `TypeError` was
  swallowed by a `logger.debug`, so the diagnostics silently vanished → validation now
  accepts scalar/sequence/dict, rejects an empty sequence eagerly, and a failed
  diagnostic is a `logger.warning`.
* `parameters` keys could be given as strings; they are normalised to symbols.
* `shock_persistence` values may now name a symbol from `parameters`.

**Steady state hung on realistic models, so `demo_rbc_perturbation()` hung too.**
`NanoTensor.compute_steady_state` always tried `sympy.solve` first; for an RBC Euler
residual with fractional powers that never returns (measured: >10 minutes, stack parked
in `heuristicgcd`) — the previous "10-second timeout" in the CLI entry point had been
measured on a *simpler* model, so the flagship demo was never shipped working. It now
attempts the exact solve only for small polynomial systems (`_is_small_polynomial_system`), takes
a `method={'auto','symbolic','numeric'}` argument, and solves genuine *systems* with
`nsolve(eqs, vars, guess)` instead of the previous "solve equation 0 once per variable"
heuristic (which was a silent wrong-answer machine and, once the hang was removed, an
outright `TypeError`). `demo_rbc_perturbation` went from **hangs → 11.7 s** with
residual 3.4e-10, and
`tests/test_package_api.py::test_compute_steady_state_never_hangs_on_transcendental_systems`
pins the guarantee: a transcendental system either returns or raises, it never spins.

**`demo_rbc_perturbation` produced a degenerate policy.** It reported
"✅ Success" while fitting a single coefficient (`g_sig_sig = 0`) and then crashed in
`predict` with three unfitted symbols. It is now built on the verified
`SecondOrderPerturbation` engine, prints the steady state, determinacy check and
residual, and installs the *levels* policy in a `NanoTensor` so
`trainer.predict({'k': 9.0, 'a': 0.05})` → 9.342649 (checked against the independent
`evaluate_policy` path in `tests/test_perturbation.py`). `streamlit_dashboard` was
rewritten around it (cached solve, level-space sliders, steady state marked).

**`SymbolicTensor` (`tensor.py`) didn't implement its README API.** `create`, `add`,
`sub`, `mul`, `div`, `outer`, `__matmul__`, `contract` were missing or wrong; shape
mismatches broadcast instead of failing; `eval_numeric` half-evaluated. Now: shape
validation on construction and on every binary op, `contract` validated against
`np.einsum`, strict `eval_numeric` (missing symbol ⇒ `KeyError`), plus
`from_nested`/`to_nested` and a docstring note that `diff/subs/simplify` return **new**
tensors (unlike `NanoTensor`).

**`SymbolicTrainer`/`HybridTrainer`.** `fit` ignored its `method`, `symbolic_regression`
never fitted the order it selected (contradicting its own docstring — it now returns a
*fitted* tensor), `predict_batch`/`make_loader`/`torch_fit` had unvalidated inputs
(clear `ValueError`s now), `deriv_tree(**kwargs)` was unusable, `save_brain`/`load_brain`
were not atomic and stdlib `pickle` raised `PicklingError` on lambdify closures
(`__getstate__`/`_TRANSIENT_STATE`), `compute_grid` used `meshgrid`'s `xy` default
(collapsing start/goal into one node → degenerate A* paths), plotting helpers returned
`None`, and Taylor generation was capped at order 2.

**Gröbner (`solver/groebner.py`).** `groebner_solve` consumed a generator
(`list(G.polys)` now), `StreamingGröbnerSolver` reported chunked progress that could
regress, and the "infinite solutions" helper was not reachable from the documented API.
`handle_infinite_solutions`, `RealTimeGröbnerSolver.progress_callback` and the
`GröbnerBasisState.to_dict/to_json` shape are pinned by `tests/test_groebner.py`
(20 tests).

**Serialization (`io/serialization.py`) was one-way.** There were `serialize_*` methods
with no inverse for policy functions and Gröbner states, formats fell through to JSON
silently, and a corrupt msgpack payload leaked `msgpack.ExtraData`. Now every format has
a matching `deserialize_*` with `_pack`/`_unpack`/`_expect` helpers, an explicit
unsupported-format error, wrapped decode errors, a legacy wire-format reader for
`PolicyFunction`, and `verify_round_trip_serialization()` returns all-True (also a CI
step).

**Ecosystem mocks (`ecosystem.py`).** `MockChrono.chrono_propagate` had its arguments
in a different order from the ABC and `compute_lyapunov_exponents` was
dimension-shape-wrong; `MockTopo` returned placeholder homology (a naive 8-connected
sign-change cell graph, which invents loops — it is now a marching-segments crossing
graph, so nested circles give `b1 == 2`); `MockMorpho` did not exist at all (now:
17 transformations with required-parameter validation, `generate_variants`,
`learn_transformation`). `EcosystemBridge` validates providers with `isinstance` at
construction and its `NotImplementedError`s name the interface *and* the bundled mock.

**Reasoning.** `SymbolicAStarPathfinder` silently treated any `mode` other than
`'minimize'` as `'maximize'` (a typo flipped the objective) and its `edge_cost`
*rewarded* a maximiser for descending; both fixed, plus `step_size <= 0` rejected.
`hsws.py`'s `_match_recursive` matched only a node's *name*, so `register_meaning()`
had no effect at all (meanings were looked up only for one engine class, behind a
`RobustSemanticEngine` check that was dead code because
`DictionarySemanticEngine` **is** `RobustSemanticEngine`); registered definitions are
now consulted for every engine and the dead isinstance branches are gone. Effect:
`process(Concept("AI"), "artificial intelligence")` → 900.0 "Plausible Connection"
instead of the uninformative 500.0.

**Explainability.** `DerivativeNode.node_id` was `f"{type}_{id(expr)}"` — recycled CPython
addresses merged unrelated nodes, producing self-loops and a ranking where every
variable tied; identity is now content-based, `direct_sensitivities()` was added for a
path-independent `|df/dv|` ranking, and `export_graphviz` returns the path it wrote.

**`nano_tensor_enhanced.py`.** `eval_numeric` bound only symbols in a generated Taylor
expansion, so hand-assigned entries (`nt.data[0] = x**2 + y`, the documented usage in
both READMEs) returned *symbolic junk*; symbol-keyed points returned `0.0`; the eval
cache ignored mutation of `data` (stale hits returned stale numbers); `symvars` was
cached and went stale for the same reason; caches grew without bound
(`max_cache_size` was ignored); `current_goal` was reported but never set; the shape
was unvalidated; and the timing-anomaly detector warned on every cache hit (0.0001 s
"anomaly"). All fixed — the eval cache is now validated against the exact expression
objects it was built from (`_data_snapshot`), which is cheap and exact because SymPy
expressions are immutable.

**Root scripts.** `demo_military_grade.py`, `stress_test.py` and
`build_desktop_binaries.sh` loaded a `symbo.py` that no longer exists (the first two
crashed on `import`, the shell script added `--add-data "symbo.py:."`). They now use the
installed package; both scripts run end-to-end, are lint-clean, and CI executes them.

## 4. New tests

Rewritten onto package imports: `test_primitives.py` (3 → 35 tests),
`test_new_features.py`, `test_military_grade_nanotensor.py`. Added:

| file | tests | covers |
|---|---|---|
| `tests/conftest.py` | — | shared RBC model fixture |
| `test_nanotensor_core.py` | 59 | engine core, trainers, KB, persistence, grids |
| `test_package_api.py` | 72 | API surface, lazy imports, extras, demos, CLI |
| `test_security.py` | 42 | `safe_sympify` allow/deny lists, length limits, injection |
| `test_nano_tensor_enhanced.py` | 37 | hardened standalone class |
| `test_ecosystem.py` | 36 | four providers + bridge |
| `test_reasoning.py` | 31 | A* and HSWS |
| `test_tensor.py` | 29 | `SymbolicTensor` algebra |
| `test_perturbation.py` | 26 | solver verified by residual scaling |
| `test_wasm_bindings.py` | 25 | browser boundary |
| `test_serialization.py` | 20 | wire formats, legacy payloads |
| `test_groebner.py` | 20 | batch/streaming/real-time |
| `test_explain.py` | 20 | derivative trees |
| `test_taylor.py` | 17 | expansions, policy functions, WASM payload |
| `test_optional_backends.py` | 13 | both "extra present" and "extra missing" paths |
| `test_docs.py` | 12 | README/MODULES examples actually run |

`tests/test_docs.py` is the one to trust for "is the documentation real": it executes
each block in a fresh interpreter from a temp directory. Writing it found three doc bugs
(wrong `create` call signature, a missing `shock_persistence`, a missing argument) — and
the perturbation block in MODULES.md is the worked example I'd show a new user.

## 5. Verification recipe

```bash
pip install -e ".[dev]"                      # or ".[viz,io,opt,kb,dashboard,dev]"
python -m pytest tests -q                    # 511 passed, 3 skipped, no warnings
python -m pytest tests -q -m "not slow"      # the same, minus ~2 min of solves
python -m pytest --doctest-modules symbo -q  # 26 passed
python -m ruff check symbo tests demo_military_grade.py stress_test.py
python -m pyflakes symbo tests demo_military_grade.py stress_test.py
python -m symbo --check                      # feature report for this interpreter
python -m symbo --demo bench                 # seconds
python -m symbo --demo rbc                   # ~12 s, prints residual 3.4e-10
python -c "from symbo.io.serialization import verify_round_trip_serialization as v; print(v())"
python demo_military_grade.py && python stress_test.py
```

Numbers to compare against (Python 3.11, core + msgpack/pyarrow/dill/networkx, no
matplotlib/torch): pytest 511 passed / 3 skipped (514 collected); doctests 26 passed
(+1 skipped); coverage 83 %; `verify_round_trip_serialization()` all True.

Build note: `pyproject.toml` declares the PEP 639 SPDX form (`license =
"Apache-2.0"`), so building from source needs **setuptools >= 77** (pinned in
`[build-system]`). Plain `pip install .` / `pip install -e .` get it through build
isolation; `--no-build-isolation` on an older system setuptools fails at config
validation, which is expected, not a broken package.

## 6. Known limitations

1. **torch is untested here.** It cannot be installed in this sandbox (TLS-blocked
   index, no CPU wheel), so `HybridTrainer.torch_fit`/`make_loader` are covered only by
   their *missing-dependency* behaviour plus `tests/test_optional_backends.py`, which
   runs the happy path when torch is importable. CI has a dedicated
   `pip install torch --index-url .../whl/cpu` job, `continue-on-error: true`, selected
   with `-m torch`. Consider making it blocking once it has been seen green.
2. **matplotlib/plotly paths are only smoke-covered.** In this environment they are
   absent, so the plot tests skip; the CI `extras` job installs them but does not assert
   on figure contents beyond type. No image-comparison tests exist.
3. **`kanren` is not exercised** (not installed here); `KnowledgeBase` runs on its
   pure-python fallback. The relational-query path is untested.
4. **Python 3.10 and 3.13 are declared but only run in CI** — local verification was
   3.11. `requires-python = ">=3.10"` is unverified for 3.10 beyond `ruff target-version`.
5. **`full_perturbation` in `nanotensor.py` remains fragile** (that is the routine that
   hid the steady-state hang). The user-facing demos no longer use it, and it has no
   tests. It should either be re-implemented on top of
   `symbo.analytics.perturbation.SecondOrderPerturbation` or deprecated; deleting it is
   an API decision for the maintainers.
6. **`MockFortArch` is a mock**: `encrypt_expression` is `str(expr).encode()`. It
   exercises the interface, it is not security. Same for the homomorphic step
   (simplify-on-plaintext).
7. **The WASM story is a Python-side contract, not a `.wasm` build.** `wasm_bindings.py`
   gives JSON/msgpack-compatible functions and a browser test payload, and the payload
   round-trips, but no Pyodide/WASI artifact is produced by this repo and
   `build_desktop_binaries.sh` produces PyInstaller executables, not a browser bundle.
8. **No static typing gate.** Annotations are thorough, but there is no mypy/pyright
   config, and `pyproject.toml` ships no `py.typed`.
9. **Docs are markdown + PDFs**; nothing builds a site, and the four PDFs in `docs/`
   are historical (they predate several of the API corrections above).
10. **CI is written but not switched on.** `.ci/ci.yml` is a complete five-job workflow,
    and it is in that directory rather than `.github/workflows/` only because GitHub
    refuses to let a GitHub App token create files under `.github/workflows/` without
    the `workflows` permission (the push is rejected with *"refusing to allow a GitHub
    App to create or update workflow"*). A maintainer enables it with
    `mkdir -p .github/workflows && git mv .ci/ci.yml .github/workflows/ci.yml`; until
    then the merge is unverified by machines, so run the recipe in §5.
11. **Version/packaging details for release:** `CITATION.cff` was updated to 0.2.0 (it
    said 1.0.0 while the package said 0.1.x/0.2.0); there is no CHANGELOG, no git tag and
    no PyPI publication — a `v0.2.0` tag plus `python -m build` + `twine upload` is all
    that stands between this branch and a release (the CI `package` job already builds and
    checks the distributions).

## 7. Suggested next steps, in order of value

1. Land this branch; tag `v0.2.0`; publish to PyPI (the `package` CI job verifies the wheel).
2. Make the torch job blocking, then add `skopt` to the same job so
   `symbolic_regression`'s GP path is covered.
3. Decide `full_perturbation`: reimplement or `DeprecationWarning` + removal plan (limitation 5).
4. Fold `MilitaryGradeNanoTensor` into `NanoTensor` as an opt-in hardening layer, or
   generate one from the other, to end the duplicated agency code (limitation §2).
5. Add `py.typed` + a `mypy` job (start in non-strict mode on `symbo/tensor.py`,
   `symbo/security.py`, `symbo/io/`, which are the modules users extend).
6. Replace the two remaining `logger.debug`-level "computation failed" messages in
   `nanotensor.py` with warnings, following the pattern now used in the perturbation
   solver, so silent degradation is impossible anywhere in the package.
7. Build a docs site from `docs/*.md` + the docstrings (mkdocs) and move the PDFs under
   `docs/history/`, marked as pre-0.2 references.

## 8. Commit map

Eleven commits, in dependency order — `git log --oneline main..HEAD` is the summary and
each message states the bug it fixes.

| commit | what is in it |
|---|---|
| `build(packaging): make symbo an installable package with a real entry point` | `symbo.py` → `symbo/nanotensor.py`, package re-exports, `_optional.py`, `demos.py`, `__main__.py` CLI, `pyproject.toml` (extras + pytest + ruff config, `setuptools>=77`), `requirements.txt`, `.gitignore` |
| `fix(core): close the safe_sympify sandbox escape and implement the tensor API` | `security.py`, `tensor.py`, `generative/taylor.py`, `primitives.py` |
| `fix(analytics): make the perturbation solver honest and the solvers robust` | `analytics/perturbation.py`, `analytics/explain.py`, `solver/groebner.py` |
| `fix(io): stop silently mis-decoding payloads, and decode Taylor keys` | `io/serialization.py`, `wasm_bindings.py` |
| `fix(ecosystem): replace the stubbed mocks with real implementations` | `ecosystem.py` (`MockTopo`, `MockMorpho`, real Euler propagation, provider checks) |
| `fix(reasoning): make A* cost honest and HSWS engine-agnostic` | `reasoning/a_star.py`, `reasoning/hsws.py` |
| `fix(enhanced): stop the standalone tensor from caching and evaluating wrongly` | `nano_tensor_enhanced.py` |
| `fix(core): make the timing-anomaly warnings usable` | the two `warnings.warn` sites in `nanotensor.py` / `nano_tensor_enhanced.py` |
| `test(suite): cover every public module, the docs, and both dependency modes` | `tests/` (3 → 18 files, 23 → 514 tests) |
| `ci: gate lint, a core-only install, extras, torch and a built wheel` | `.ci/ci.yml`, `build_desktop_binaries.sh`, `demo_military_grade.py`, `stress_test.py` |
| `docs: ...` (this commit) | `README.md`, `MODULES.md`, `README_MILITARY_GRADE.md`, `IMPLEMENTATION_SUMMARY.md`, `CITATION.cff`, `HANDOFF.md` |

Every engine-level fix (steady state, grid indexing, trainers, knowledge base,
plotting, persistence) is in the first commit, because they live in
`symbo/nanotensor.py`; that one is the big one to review — `git show --stat` on it
reports the rename with only 53 % similarity, i.e. most of the file changed.
