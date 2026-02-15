# Copyright 2025 Achronus
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
MixFlow-MG: Mixed-mode differentiation for efficient meta-gradient computation.

Implements the forward-over-reverse reparameterization from:

    Kemaev et al., "Scalable Meta-Learning via Mixed-Mode Differentiation",
    ICML 2025. (Section 3.1, Algorithm 2, Appendix A.4)

The key insight: standard bilevel optimization computes inner gradients
inside the outer differentiation graph, creating reverse-over-reverse
differentiation. This stores all intermediate activations from each
inner backward pass during the outer backward pass, consuming O(T * |A|)
memory where T is the number of inner steps and |A| is the activation size.

MixFlow-MG reparameterizes the inner loop so that inner gradients are
computed via `fwdrev_grad` — a function with a custom VJP that computes
Hessian-vector products (HVPs) using forward-over-reverse mode. This
avoids storing inner backward pass activations, reducing dynamic memory
by up to 10x and wall-clock time by up to 25%.
"""

from functools import partial
from typing import Any, Callable, Tuple

import jax


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def fwdrev_grad(
    fn: Callable,
    *args: Any,
) -> Any:
    """
    Compute the gradient of `fn` with a custom VJP rule that uses
    forward-over-reverse mode for Hessian-vector products.

    In the forward pass, this is identical to `jax.grad(fn)(*args)`.
    In the backward pass (when this function is itself differentiated
    by an outer `jax.grad`), the HVPs are computed using `jax.jvp`
    over `jax.grad` (forward-over-reverse) instead of the default
    `jax.grad` over `jax.grad` (reverse-over-reverse).

    This avoids storing intermediate activations from the inner backward
    pass, which is the primary source of memory savings.

    Parameters
    ----------
    fn : Callable
        A scalar-valued function to differentiate. Must return either
        a scalar or `(scalar, aux)` tuple
    *args : Any
        Arguments to `fn` (typically model parameters and inputs)

    Returns
    -------
    grad : Any
        Gradient of `fn` w.r.t. its first argument, same structure
        as `args[0]`

    Notes
    -----
    Based on Algorithm 2 and Appendix A.4 of Kemaev et al. (2025).
    The `nondiff_argnums=(0,)` ensures `fn` is not traced by JAX
    but passed through as a static Python callable.
    """
    return jax.grad(fn)(*args)


def _fwdrev_grad_fwd(
    fn: Callable,
    *args: Any,
) -> Tuple[Any, Any]:
    """
    Forward rule for `fwdrev_grad`.

    Computes the gradient (primal output) and saves the arguments
    as residuals for use in the backward pass. The gradient itself
    is NOT saved as a residual — it will be recomputed in the backward
    pass via `jax.jvp`.

    Parameters
    ----------
    fn : Callable
        The function being differentiated
    *args : Any
        Arguments to `fn`

    Returns
    -------
    grad : Any
        The gradient (primal output)
    residuals : Tuple
        The arguments, saved for the backward pass
    """
    grad = jax.grad(fn)(*args)
    return grad, args


def _fwdrev_grad_bwd(
    fn: Callable,
    residuals: Any,
    cotangent: Any,
) -> Tuple[Any, ...]:
    """
    Backward rule for `fwdrev_grad` using forward-over-reverse mode.

    When the outer `jax.grad` calls back through `fwdrev_grad`, it
    needs to compute a vector-Hessian product (VHP): `v @ H` where
    `v` is the cotangent and `H = ∂²L/∂θ²`.

    By the symmetry of the Hessian (Schwarz's theorem), this equals
    `(H @ v)ᵀ = (HVP)ᵀ`. The HVP `H @ v` can be computed efficiently
    using `jax.jvp` over `jax.grad` (forward-over-reverse), which
    avoids storing the activations from the inner backward pass.

    Parameters
    ----------
    fn : Callable
        The function being differentiated
    residuals : Tuple
        The saved arguments from the forward pass
    cotangent : Any
        The cotangent vector from the outer backward pass (same
        structure as the gradient output)

    Returns
    -------
    grads : Tuple[Any, ...]
        Gradients w.r.t. each argument of `fwdrev_grad`, computed
        via forward-over-reverse HVP

    Notes
    -----
    The `jax.jvp(jax.grad(fn), primals, tangents)` call computes:
        - primals: `jax.grad(fn)(*args)` (recomputed, not stored)
        - tangents: `J @ tangents` where J is the Jacobian of grad(fn)

    For a scalar loss, the Jacobian of `grad(fn)` is exactly the Hessian,
    so `tangents` gives us the HVP we need.
    """
    args = residuals

    # Forward-over-reverse: jvp of grad(fn) computes HVP
    # primals_out = grad(fn)(*args)       [recomputed]
    # tangents_out = H @ cotangent        [the HVP we need]
    _, hvp = jax.jvp(jax.grad(fn), args, cotangent)
    return hvp


fwdrev_grad.defvjp(_fwdrev_grad_fwd, _fwdrev_grad_bwd)


def fwdrev_value_and_grad(
    fn: Callable,
    has_aux: bool = False,
) -> Callable:
    """
    Drop-in replacement for `jax.value_and_grad` that uses
    forward-over-reverse differentiation for the inner gradient.

    When used inside a function that is itself differentiated by
    `jax.grad` (the outer meta-gradient), this produces the same
    mathematical result as `jax.value_and_grad` but with substantially
    lower memory consumption.

    Parameters
    ----------
    fn : Callable
        A scalar-valued function, optionally returning auxiliary data
    has_aux : bool
        If `True`, `fn` returns `(scalar, aux)` and the gradient
        is only w.r.t. the scalar. Default is `False`

    Returns
    -------
    wrapped : Callable
        A function with the same signature as `fn` that returns
        `((value, aux), grad)` if `has_aux` else `(value, grad)`
    """
    if has_aux:

        def _value_fn(*args):
            """Scalar-only version for grad computation."""
            val, _ = fn(*args)
            return val

        def wrapped(*args):  # type: ignore
            value, aux = fn(*args)
            grad = fwdrev_grad(_value_fn, *args)
            return (value, aux), grad

    else:

        def wrapped(*args):
            value = fn(*args)
            grad = fwdrev_grad(fn, *args)
            return value, grad

    return wrapped
