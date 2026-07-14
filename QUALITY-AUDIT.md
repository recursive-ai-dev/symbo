# Symbo Code Quality Audit

## 1. Resilience: Missing Caching Invalidation on Structural Changes
**Files and Lines Affected:** `symbo.py` lines 803, 812, 1383, 1394
**The Problem:** `_compile_func` and `diff_cached` rely on internal caches. When `self.data` is overwritten by differentiation, substitution, or simplification, some caches are cleared, but `_compile_func` (decorated with `@lru_cache`) cannot be cleared directly without calling `.cache_clear()`. Also, `simplify` drops `_symvars_cache` and `_diff_cache` but misses `_lambdify_cache` completely.
**Why it matters:** If `simplify()` or `diff()` modifies the structure, `eval_numeric` might still use stale lambda functions from `_lambdify_cache` (which is not cleared by `simplify`) or `_compile_func` could return cached lambdas for old expressions.
**Proposed Fix:** Add `self._compile_func.cache_clear()` and `self._lambdify_cache.clear()` in any structural mutation methods, notably `diff()` (where `new_nt` is created but maybe not relevant if it's a new instance, however `simplify` mutates `self.data` in place and MUST clear `self._lambdify_cache` and `self._compile_func.cache_clear()`).
**Overlaps:** None

## 2. Correctness Risk: Division by Zero in Anomaly Check
**Files and Lines Affected:** `symbo.py` lines 382-384
**The Problem:** The variance calculation `variance = stats["M2"] / (stats["n"] - 1)` lacks protection if `stats["n"]` is 1.
**Why it matters:** Even though `stats["n"] < 10` is checked just above it, if the threshold is lowered, or if there is ever a condition where `n == 1`, this will raise a `ZeroDivisionError`, crashing the evaluation.
**Proposed Fix:** Add a check `if stats["n"] <= 1: return False` before computing variance.
**Overlaps:** None

## 3. Redundant Work: O(n^2) Cache Invalidation in `subs`
**Files and Lines Affected:** `symbo.py` lines 873-884
**The Problem:** The `_is_coeff_key` helper function within `subs` checks if a key is in `self.coeff_vars`, doing list lookups. The `any(_is_coeff_key(k) for k in sub_dict.keys())` call has complexity O(K * C) where K is number of substitution keys and C is number of coefficients.
**Why it matters:** `coeff_vars` is a list. For large tensors or taylor expansions with many coefficients, this linear scan inside a loop will significantly slow down substitution checks.
**Proposed Fix:** Convert `self.coeff_vars` to a `set` (or keep a shadow set) for O(1) membership testing, or at least compute `coeff_set = set(self.coeff_vars)` once before the loop in `subs`.
**Overlaps:** None

## 4. Real Bug: Mutable Default Argument in `start_repl`
**Files and Lines Affected:** `symbo.py` lines 2092, 1961
**The Problem:** In functions like `demo_rbc_perturbation(ss_guess: Dict[str, float] = None)`, mutable defaults aren't used, but there's a risk. The main bug is in `full_perturbation` where `var_order: List[str] = ['k', 'a', 'eps', 'sig']` is a mutable default.
**Why it matters:** If the caller mutates `var_order`, the default for all future calls to `full_perturbation` is permanently modified.
**Proposed Fix:** Change to `var_order: List[str] = None` and initialize `if var_order is None: var_order = ['k', 'a', 'eps', 'sig']`.
**Overlaps:** None

## 5. Resilience: Unsafe `eval_numeric` Return Reshape
**Files and Lines Affected:** `symbo.py` line 967
**The Problem:** `np.array(result_flat).reshape(self.shape)` assumes that `result_flat` can be reshaped perfectly to `self.shape`. If lambdified functions return scalars instead of arrays (when `point` scalars are passed), `np.array` will be a 1D array of length `len(self.data.flat)`. But if the lambdified function returns arrays (e.g. from vectorized inputs), `result_flat` will be a list of arrays, and `np.array(result_flat)` will have shape `(len(funcs), batch_size)`, which `reshape(self.shape)` will fail on.
**Why it matters:** It will crash with a ValueError when evaluating over a grid (batching) if `shape` doesn't account for the batch dimension.
**Proposed Fix:** Support batching by checking the shape of the result, or explicitly ensure it handles single-point evaluation by wrapping scalar return values correctly. For single point, `reshape(self.shape)` is fine, but it breaks `predict_batch`.
**Overlaps:** None

## 6. Dead Code / Redundancy: `_compile_func` Unused
**Files and Lines Affected:** `symbo.py` line 782-785
**The Problem:** `_compile_func` is defined and uses `@lru_cache`, but it is never called anywhere in the `symbo.py` file. Instead, `eval_numeric` uses `self._lambdify_cache` (line 959).
**Why it matters:** Dead code adds maintenance burden and confusing overlapping logic with `_lambdify_cache`.
**Proposed Fix:** Remove `_compile_func` entirely.
**Overlaps:** Overlaps with Item 1, but resolving this by deleting it fixes the LRU cache issue.

## 7. Real Bug: Implicit Variable Ordering in NN Evaluation
**Files and Lines Affected:** `symbo.py` lines 1889-1895
**The Problem:** `vars_syms = list(nt.base_vars) + list(nt.coeff_vars)`. `args` passed to `forward` are expected to match `nt.base_vars`. If the model is called with a tensor containing a different variable order, it silently produces garbage gradients.
**Why it matters:** Neural network training will fail to converge silently due to mismatched input features.
**Proposed Fix:** Ensure the `DataLoader` yields columns strictly ordered by `nt.base_vars`, or pass variable names into `SymModule` to explicitly select the correct columns from the input batch.
**Overlaps:** None

## 8. Correctness Risk: `simplify` Recovery Fails
**Files and Lines Affected:** `symbo.py` lines 808-821
**The Problem:** In `diff()`, if the differentiation fails, the code attempts to recover by calling `self.simplify()`, which modifies `self.data` in place.
**Why it matters:** If `diff()` was called on a `NanoTensor`, modifying `self.data` in place is a side effect that the caller did not ask for. It silently mutates the state of the object during an operation that is expected to return a new object (a new `NanoTensor` is created and returned).
**Proposed Fix:** In the recovery block, do `new_data = np.vectorize(sp.simplify)(self.data)` and operate on `new_data`, without calling `self.simplify()` which modifies the instance.
**Overlaps:** None

## 9. Real Bug: Silent Failure in Optimization
**Files and Lines Affected:** `symbo.py` lines 433-434, 476-479
**The Problem:** `optimize_storage` catches `Exception` and silently sets caches to `None`. `_attempt_self_optimization` catches `Exception` and passes.
**Why it matters:** If an error occurs (e.g. `MemoryError` or `RecursionError` from SymPy CSE), it is silently swallowed. The object degrades to unoptimized state without any logs, making debugging impossible if optimization has a systematic failure.
**Proposed Fix:** Log a warning using the `warnings` module when CSE optimization fails instead of a completely silent `pass`.
**Overlaps:** None

## 10. Redundant Work: Repetitive Sorting in Caching
**Files and Lines Affected:** `symbo.py` line 938
**The Problem:** `key = tuple(sorted([v.name for v in vars_in_point])) + tuple(sorted(point.keys()))` sorts the list of variable names on every single evaluation.
**Why it matters:** `eval_numeric` is typically called thousands of times (e.g. in `plot_surface` or `plot_grid_with_path`). String sorting on every call adds unnecessary overhead.
**Proposed Fix:** Since `vars_in_point` is derived directly from `point.keys()` intersecting with `self.symvars`, the cache key can just be a `frozenset(point.keys())` assuming `point` dictates the variables.
**Overlaps:** None
