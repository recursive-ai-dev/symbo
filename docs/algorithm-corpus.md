# Algorithm corpus

Several documents historically opened with the claim that Symbo was built by
deconstructing **318 classical algorithms**. That figure is a rhetorical
framing from early drafts. This repository does **not** ship a bibliography or
a mapping from any numbered algorithm onto a primitive, so the claim is not
falsifiable as written.

What *is* implemented, and tested, is the compact set of operations on
`symbo.primitives.AtomicPrimitives`:

| Category | Operations |
|---|---|
| Algebraic | `symbolic_add`, `symbolic_mul`, `symbolic_pow`, `symbolic_div` |
| Differential | `symbolic_diff`, `gradient`, `hessian`, `jacobian` |
| Tensor | `tensor_contraction`, `outer_product`, `tensor_trace`, `symbolic_tensor_product` |
| Polynomial | `polynomial_expand`, `polynomial_factor`, `polynomial_collect`, `polynomial_degree`, `polynomial_coeffs` |
| Matrix | `matrix_det`, `matrix_inv`, `matrix_eigenvals`, `matrix_eigenvects` |
| Calculus | `symbolic_integrate` |
| Evaluation | `substitute`, `evaluate_numeric`, `simplify`, `trigsimp`, `ratsimp`, `cancel` |

Higher-level engines (Gröbner bases, second-order perturbation, A* on a grid,
Taylor generation) recombine these primitives; they are not a hidden catalogue
of 318 named algorithms. If a numbered corpus is published later, this file is
the place to list it.
