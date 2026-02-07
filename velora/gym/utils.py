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

import gymnasium as gym


def get_n_actions(env_name: str) -> int:
    """
    Gets the number of actions for a Gymnasium environment.

    Parameters
    ----------
    env_name : str
        Name of the Gymnasium environment.

    Returns
    -------
    n_actions : int
        Number of actions in the environment.

    Raises
    ------
    invalid_env : ValueError
        Unsupported environment if it does not have a `Discrete` action space.
    """
    env = gym.make(env_name)

    if isinstance(env.action_space, gym.spaces.Discrete):
        return env.action_space.n.item()

    raise ValueError(
        f"Unsupported environment. Must have a 'Discrete' action space. Got: {type(env.action_space)}"
    )
