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

from dataclasses import dataclass


@dataclass
class PPOConfig:
    """
    A configuration for the PPO algorithm's hyperparameters.

    Parameters
    ----------
    total_timesteps : int (optional)
        The total number of environment timesteps to train for.
        Default is `1_000_000`
    lr : float (optional)
        The learning rate of the optimizer. Default is `3e-4`
    num_steps : int (optional)
        The number of steps to run in each environment per policy rollout.
        Default is `2048`
    anneal_lr : bool (optional)
        Whether to anneal the learning rate for the policy and value
        networks. Default is `True`
    gamma : float (optional)
        The discount factor. Default is `0.99`
    gae_lambda : float (optional)
        The lambda for the Generalized Advantage Estimation (GAE).
        Default is `0.95`
    num_minibatches : int (optional)
        The number of mini-batches per update. Default is `32`
    update_epochs : int (optional)
        The number of epochs (K) to update the policy. Default is `10`
    norm_adv : bool (optional)
        Whether to normalize the advantages. Default is `True`
    clip_coef : float (optional)
        The surrogate clipping coefficient. Default is `0.2`
    clip_vloss : bool (optional)
        Whether to use a clipped loss for the value function, as per the
        paper. Default is `True`
    ent_coef : float (optional)
        The entropy coefficient. Default is `0.0`
    vf_coef : float (optional)
        The value function coefficient. Default is `0.5`
    max_grad_norm : float (optional)
        The maximum norm for gradient clipping. Default is `0.5`
    target_kl : float (optional)
        The target KL divergence threshold. Default is `None`
    """

    total_timesteps: int = 1_000_000
    lr: float = 3e-4
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None


@dataclass
class RPOConfig(PPOConfig):
    """
    A configuration for the Robust Policy Optimization (RPO) algorithm's
    hyperparameters. Extends `PPOConfig`.

    Parameters
    ----------
    total_timesteps : int (optional)
        The total number of environment timesteps to train for.
        Default is `1_000_000`
    lr : float (optional)
        The learning rate of the optimizer. Default is `3e-4`
    num_steps : int (optional)
        The number of steps to run in each environment per policy rollout.
        Default is `2048`
    anneal_lr : bool (optional)
        Whether to anneal the learning rate for the policy and value
        networks. Default is `True`
    gamma : float (optional)
        The discount factor. Default is `0.99`
    gae_lambda : float (optional)
        The lambda for the Generalized Advantage Estimation (GAE).
        Default is `0.95`
    num_minibatches : int (optional)
        The number of mini-batches per update. Default is `32`
    update_epochs : int (optional)
        The number of epochs (K) to update the policy. Default is `10`
    norm_adv : bool (optional)
        Whether to normalize the advantages. Default is `True`
    clip_coef : float (optional)
        The surrogate clipping coefficient. Default is `0.2`
    clip_vloss : bool (optional)
        Whether to use a clipped loss for the value function, as per the
        paper. Default is `True`
    ent_coef : float (optional)
        The entropy coefficient. Default is `0.0`
    vf_coef : float (optional)
        The value function coefficient. Default is `0.5`
    max_grad_norm : float (optional)
        The maximum norm for gradient clipping. Default is `0.5`
    target_kl : float (optional)
        The target KL divergence threshold. Default is `None`
    rpo_alpha : float (optional)
        The perturbation bound for Robust Policy Optimization (RPO).
        During policy updates, noise `z ~ Uniform(-rpo_alpha, rpo_alpha)`
        is added to the action mean before recomputing log probabilities,
        keeping the policy distribution wide to prevent premature
        entropy collapse. Higher values increase robustness at the cost
        of slower convergence. Default is `0.01`
    """

    rpo_alpha: float = 0.01
