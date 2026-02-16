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

References
----------
.. [1] Kemaev et al., "Scalable Meta-Learning via Mixed-Mode
   Differentiation", Proceedings of the 42nd ICML, PMLR 267, 2025.
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
    Compute the gradient of `fn` w.r.t. its first argument, with a
    custom VJP rule that uses forward-over-reverse mode for HVPs.

    In the forward pass, this is identical to `jax.grad(fn)(*args)`.
    In the backward pass (when this function is itself differentiated
    by an outer `jax.grad`), the HVPs are computed using `jax.jvp`
    over `jax.grad` (forward-over-reverse) instead of the default
    `jax.grad` over `jax.grad` (reverse-over-reverse).

    This avoids storing intermediate activations from the inner backward pass, which is the primary source of memory savings.

    Important: `fn` is a non-diff argument (static Python callable).
    Any values that the outer gradient needs to differentiate through
    must be passed as explicit `*args`, not closed over by `fn`.

    Parameters
    ----------
    fn : Callable
        A scalar-valued function to differentiate. Must return a scalar. All differentiable state must be passed via `*args`.
    *args : Any
        Arguments to `fn`. The gradient is computed w.r.t. the first
        argument only, but ALL args participate in the outer backward
        pass (HVP computation).

    Returns
    -------
    grad : Any
        Gradient of `fn` w.r.t. `args[0]`
    """
    return jax.grad(fn)(*args)


def _fwdrev_grad_fwd(
    fn: Callable,
    *args: Any,
) -> Tuple[Any, Any]:
    """
    Forward rule for `fwdrev_grad`.

    Computes the gradient (primal output) and saves the args
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
    Backward rule using forward-over-reverse HVP.

    Computes `jax.jvp(jax.grad(fn), primals, tangents)` which gives
    the Hessian-vector product needed by the outer backward pass.

    The cotangent tuple has the same structure as `*args`. The JVP
    propagates tangents through all args, so the outer gradient
    correctly flows through both the params (first arg) and any
    additional differentiable state (remaining args).

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
    """
    args = residuals
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

    Important: unlike `jax.value_and_grad`, the loss function `fn`
    must receive ALL differentiable state as explicit arguments. Values
    that the outer `jax.grad` needs to differentiate through cannot
    be closed over — they must be passed as arguments to `fn`.

    The gradient is computed w.r.t. the **first** argument only (like
    `jax.value_and_grad` with default `argnums=0`). Additional
    arguments are passed through to support outer-gradient flow.

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
