# Copyright 2026 Achronus
# Copyright 2022 Garena Online Private Limited
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
# The `Adan` optimizer is adapted from the official implementation at
# https://github.com/sail-sg/Adan (Apache-2.0), with modifications.


import math
from typing import Any, Callable, Dict, Iterable, List, Tuple

import torch
from torch.optim import Optimizer

ParamsT = (
    Iterable[torch.Tensor]
    | Iterable[dict[str, Any]]
    | Iterable[tuple[str, torch.Tensor]]
)


class Adan(Optimizer):
    """
    Adan (Adaptive Nesterov Momentum Algorithm) optimizer from the paper:
    [Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models](https://arxiv.org/abs/2208.06677).

    Parameters
    ----------
    params : ParamsT
        Iterable of parameters to optimize or
        dicts defining parameter groups.
    lr : float (optional)
        Learning rate. Default is `1e-3`
    betas : Tuple[float, float, float] (optional)
        coefficients used for first- and second-order moments.
        Default is `(0.98, 0.92, 0.99)`
    eps : float (optional)
        Term added to the denominator to improve
        numerical stability. Default is `1e-8`
    weight_decay : float (optional)
        Decoupled weight decay (L2 penalty). Default is `0`
    max_grad_norm : float (optional)
        Value used to clip global grad norm. Default is `0.0` (no clip)
    no_prox : bool (optional)
        How to perform the decoupled weight decay. Default is `False`
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Tuple[float, float, float] = (0.98, 0.92, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        max_grad_norm: float = 0.0,
        no_prox: bool = False,
    ):
        for name, value in (("lr", lr), ("eps", eps), ("max_grad_norm", max_grad_norm)):
            if value < 0.0:
                raise ValueError(f"'{name}' must be non-negative, got '{value}'")

        for i, beta in enumerate(betas):
            if not 0.0 <= beta < 1.0:
                raise ValueError(f"'betas[{i}]' must be in '[0.0, 1.0)', got '{beta}'")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            no_prox=no_prox,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Restore the optimizer from a pickled `state`.

        Ensures parameter groups saved without the `no_prox` option
        receive its default value.

        Parameters
        ----------
        state : Dict[str, Any]
            Unpickled optimizer state.
        """

        super(Adan, self).__setstate__(state)

        for group in self.param_groups:
            group.setdefault("no_prox", False)

    @torch.no_grad()
    def restart_opt(self) -> None:
        """
        Reset the optimizer to a freshly constructed state.

        Zeroes the moment buffers (`exp_avg`, `exp_avg_sq`, `exp_avg_diff`)
        and resets the step counter for every parameter group.
        """

        for group in self.param_groups:
            group["step"] = 0
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]

                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    # Exponential moving average of gradient difference
                    state["exp_avg_diff"] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore
        """
        Perform a single optimization step to update the parameters.

        Parameters
        ----------
        closure : Callable[[], float] (optional)
            A closure that reevaluates the model and
            returns the loss. Optional for most optimizers.

        Returns
        -------
        loss : float | None
            The loss returned by `closure`, or `None` when
            no closure is given.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.defaults["max_grad_norm"] > 0:
            device = self.param_groups[0]["params"][0].device
            global_grad_norm = torch.zeros(1, device=device)

            max_grad_norm = torch.tensor(self.defaults["max_grad_norm"], device=device)
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        grad = p.grad
                        global_grad_norm.add_(grad.pow(2).sum())

            global_grad_norm = torch.sqrt(global_grad_norm)

            clip_global_grad_norm = torch.clamp(
                max_grad_norm / (global_grad_norm + self.defaults["eps"]),
                max=1.0,
            ).item()
        else:
            clip_global_grad_norm = 1.0

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            exp_avg_diffs = []
            neg_pre_grads = []

            beta1, beta2, beta3 = group["betas"]
            # assume same step across group now to simplify things
            # per parameter step can be easily support
            # by making it tensor, or pass list into kernel
            if "step" in group:
                group["step"] += 1
            else:
                group["step"] = 1

            bias_correction1 = 1.0 - beta1 ** group["step"]
            bias_correction2 = 1.0 - beta2 ** group["step"]
            bias_correction3 = 1.0 - beta3 ** group["step"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["exp_avg_diff"] = torch.zeros_like(p)

                if "neg_pre_grad" not in state or group["step"] == 1:
                    state["neg_pre_grad"] = p.grad.clone().mul_(-clip_global_grad_norm)

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                exp_avg_diffs.append(state["exp_avg_diff"])
                neg_pre_grads.append(state["neg_pre_grad"])

            if not params_with_grad:
                continue

            kwargs = dict(
                params=params_with_grad,
                grads=grads,
                exp_avgs=exp_avgs,
                exp_avg_sqs=exp_avg_sqs,
                exp_avg_diffs=exp_avg_diffs,
                neg_pre_grads=neg_pre_grads,
                beta1=beta1,
                beta2=beta2,
                beta3=beta3,
                bias_correction1=bias_correction1,
                bias_correction2=bias_correction2,
                bias_correction3_sqrt=math.sqrt(bias_correction3),
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                no_prox=group["no_prox"],
                clip_global_grad_norm=clip_global_grad_norm,
            )

            _multi_tensor_adan(**kwargs)

        return loss


def _multi_tensor_adan(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    exp_avg_sqs: List[torch.Tensor],
    exp_avg_diffs: List[torch.Tensor],
    neg_pre_grads: List[torch.Tensor],
    *,
    beta1: float,
    beta2: float,
    beta3: float,
    bias_correction1: float,
    bias_correction2: float,
    bias_correction3_sqrt: float,
    lr: float,
    weight_decay: float,
    eps: float,
    no_prox: bool,
    clip_global_grad_norm: float,
) -> None:
    """
    Apply the Adan update to a list of parameters using `torch._foreach_*` ops.

    All buffers are updated in place. `neg_pre_grads` doubles as scratch
    space during the update and holds the negated `grads` on exit, ready
    for the next step.

    Parameters
    ----------
    params : List[torch.Tensor]
        Parameters to update.
    grads : List[torch.Tensor]
        Gradients for each parameter.
    exp_avgs : List[torch.Tensor]
        First moment buffers (`m_t`).
    exp_avg_sqs : List[torch.Tensor]
        Second moment buffers (`n_t`).
    exp_avg_diffs : List[torch.Tensor]
        Gradient difference moment buffers (`v_t`).
    neg_pre_grads : List[torch.Tensor]
        Negated gradients from the previous step.
    beta1 : float
        Decay rate for the first moment.
    beta2 : float
        Decay rate for the gradient difference moment.
    beta3 : float
        Decay rate for the second moment.
    bias_correction1 : float
        Bias correction term for `beta1`.
    bias_correction2 : float
        Bias correction term for `beta2`.
    bias_correction3_sqrt : float
        Square root of the bias correction term for `beta3`.
    lr : float
        Learning rate.
    weight_decay : float
        Decoupled weight decay coefficient.
    eps : float
        Numerical stability constant.
    no_prox : bool
        When `True`, applies weight decay multiplicatively before the
        update instead of using the proximal form.
    clip_global_grad_norm : float
        Global gradient clipping scale applied to `grads`.
    """

    if len(params) == 0:
        return

    torch._foreach_mul_(grads, clip_global_grad_norm)

    # for memory saving, we use `neg_pre_grads`
    # to get some temp variable in a inplace way
    torch._foreach_add_(neg_pre_grads, grads)

    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)  # m_t

    torch._foreach_mul_(exp_avg_diffs, beta2)
    torch._foreach_add_(exp_avg_diffs, neg_pre_grads, alpha=1 - beta2)  # diff_t

    torch._foreach_mul_(neg_pre_grads, beta2)
    torch._foreach_add_(neg_pre_grads, grads)
    torch._foreach_mul_(exp_avg_sqs, beta3)
    torch._foreach_addcmul_(
        exp_avg_sqs, neg_pre_grads, neg_pre_grads, value=1 - beta3
    )  # n_t

    denom = torch._foreach_sqrt(exp_avg_sqs)
    torch._foreach_div_(denom, bias_correction3_sqrt)
    torch._foreach_add_(denom, eps)

    step_size_diff = lr * beta2 / bias_correction2
    step_size = lr / bias_correction1

    if no_prox:
        torch._foreach_mul_(params, 1 - lr * weight_decay)
        torch._foreach_addcdiv_(params, exp_avgs, denom, value=-step_size)
        torch._foreach_addcdiv_(params, exp_avg_diffs, denom, value=-step_size_diff)
    else:
        torch._foreach_addcdiv_(params, exp_avgs, denom, value=-step_size)
        torch._foreach_addcdiv_(params, exp_avg_diffs, denom, value=-step_size_diff)
        torch._foreach_div_(params, 1 + lr * weight_decay)

    torch._foreach_zero_(neg_pre_grads)
    torch._foreach_add_(neg_pre_grads, grads, alpha=-1.0)
