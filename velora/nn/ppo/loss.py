# Copyright 2026 Achronus
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


from typing import Tuple

import torch


class PPOLoss:
    """
    A container for the PPO loss terms.

    Computes the clipped surrogate policy loss, the (optionally
    clipped) value loss, and the entropy bonus as individual terms,
    combining them into the total training loss via `total`.

    Parameters
    ----------
    clip_coef : float
        The surrogate clipping coefficient
    clip_vloss : bool
        Whether to use a clipped loss for the value function
    ent_coef : float
        The entropy coefficient
    vf_coef : float
        The value function coefficient
    """

    def __init__(
        self,
        clip_coef: float,
        clip_vloss: bool,
        ent_coef: float,
        vf_coef: float,
    ) -> None:
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef

    def policy(
        self,
        advantages: torch.Tensor,
        ratio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes a mini-batches policy gradient loss using the clipped
        surrogate objective.

        Parameters
        ----------
        advantages : torch.Tensor
            Current mini-batch of advantages
        ratio : torch.Tensor
            The probability ratios between the updated and rollout
            policies, `pi_new(a|s) / pi_old(a|s)`, computed as
            `(new_log_probs - old_log_probs).exp()`

        Returns
        -------
        pg_loss : torch.Tensor
            Policy gradient loss
        """
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio,
            1 - self.clip_coef,
            1 + self.clip_coef,
        )
        return torch.max(pg_loss1, pg_loss2).mean()

    def value(
        self,
        new_values: torch.Tensor,
        returns: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes a mini-batches value loss.

        Parameters
        ----------
        new_values : torch.Tensor
            Mini-batch of next values
        returns : torch.Tensor
            Mini-batch of returns
        values : torch.Tensor
            Mini-batch of current values

        Returns
        -------
        v_loss : torch.Tensor
            Value loss
        """
        new_values = new_values.view(-1)

        if self.clip_vloss:
            v_loss_unclipped = (new_values - returns) ** 2
            v_clipped = values + torch.clamp(
                new_values - values,
                -self.clip_coef,
                self.clip_coef,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            return 0.5 * v_loss_max.mean()

        return 0.5 * ((new_values - returns) ** 2).mean()

    def entropy(self, entropy: torch.Tensor) -> torch.Tensor:
        """
        Computes a mini-batches entropy bonus.

        Parameters
        ----------
        entropy : torch.Tensor
            The policy distribution's entropy, summed over the action
            dimensions `(minibatch_size,)`

        Returns
        -------
        entropy_loss : torch.Tensor
            Mean entropy of the mini-batch
        """
        return entropy.mean()

    def total(
        self,
        policy_loss: torch.Tensor,
        value_loss: torch.Tensor,
        entropy_loss: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combines the computed loss terms into the total training loss:
        `policy - ent_coef * entropy + value * vf_coef`.

        Parameters
        ----------
        policy_loss : torch.Tensor
            The policy gradient loss from `policy`
        value_loss : torch.Tensor
            The value loss from `value`
        entropy_loss : torch.Tensor
            The entropy bonus from `entropy`

        Returns
        -------
        loss : torch.Tensor
            Total training loss
        """
        return policy_loss - self.ent_coef * entropy_loss + value_loss * self.vf_coef

    @torch.no_grad()
    def diagnostics(
        self,
        log_ratio: torch.Tensor,
        ratio: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes gradient-free diagnostics of the clipped surrogate
        objective for a mini-batch.

        Parameters
        ----------
        log_ratio : torch.Tensor
            The log probability ratios between the updated and rollout
            policies, `new_log_probs - old_log_probs`
            `(minibatch_size,)`
        ratio : torch.Tensor
            The probability ratios, `log_ratio.exp()`
            `(minibatch_size,)`

        Returns
        -------
        old_approx_kl : torch.Tensor
            The naive KL divergence estimate, `mean(-log_ratio)`
        approx_kl : torch.Tensor
            The lower-variance KL divergence estimate,
            `mean((ratio - 1) - log_ratio)`, from
            [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html)
        clip_frac : torch.Tensor
            The fraction of the mini-batch clipped by the surrogate
            objective, where `|ratio - 1| > clip_coef`
        """
        old_approx_kl = (-log_ratio).mean()
        approx_kl = ((ratio - 1) - log_ratio).mean()
        clip_frac = ((ratio - 1.0).abs() > self.clip_coef).float().mean()
        return old_approx_kl, approx_kl, clip_frac
