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


import torch

from velora.nn.ppo.config import RPOConfig
from velora.nn.ppo.sampler import RPOActionSampler
from velora.nn.ppo.standard import PPO


class RPO(PPO):
    """
    A Robust Policy Optimization (RPO) trainer for continuous action
    spaces, based on [RPO](https://arxiv.org/abs/2212.07536).

    Extends `PPO` by perturbing the action mean with noise
    `z ~ Uniform(-rpo_alpha, rpo_alpha)` when re-evaluating actions
    during policy updates, keeping the policy distribution wide to
    prevent premature entropy collapse.

    Parameters
    ----------
    env_id : str
        The Gymnasium environment ID (e.g., `HalfCheetah-v4`)
    config : RPOConfig
        The algorithm's hyperparameter configuration
    num_envs : int (optional)
        The number of parallel environments. Default is `1`
    seed : int (optional)
        Random number generator seed. Default is `28`
    device : torch.device (optional)
        Device to load tensors onto. When `None`, sets device to CUDA
        or CPU automatically. Default is `None`
    project_name : str (optional)
        The project name for wandb metric logging. Default is `velora`
    exp_name : str (optional)
        The experiment name, shared by all seed runs of the same
        algorithm variant. Used for run grouping and benchmark tooling.
        Default is `rpo`
    """

    config: RPOConfig

    def __init__(
        self,
        env_id: str,
        config: RPOConfig,
        *,
        num_envs: int = 1,
        seed: int = 28,
        device: torch.device | None = None,
        project_name: str = "velora",
        exp_name: str = "rpo",
    ) -> None:
        super().__init__(
            env_id,
            num_envs=num_envs,
            seed=seed,
            config=config,
            device=device,
            project_name=project_name,
            exp_name=exp_name,
        )

        # Override defaults
        self.sampler = RPOActionSampler(self.config.rpo_alpha)
