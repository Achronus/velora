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

from typing import List

from velora.utils.format import number_to_short

NAME_STR = """
__     __   _
\\ \\   / /__| | ___  _ __ __ _
 \\ \\ / / _ \\ |/ _ \\| '__/ _` |
  \\ V /  __/ | (_) | | | (_| |
   \\_/ \\___|_|\\___/|_|  \\__,_|
"""


def training_info(
    agent: "RLModuleAgent",
    n_episodes: int,
    batch_size: int,
    window_size: int,
    warmup_steps: int,
    callbacks: List["TrainCallback"],
) -> None:
    """
    Display's starting information to the console for a training run.

    Parameters:
        agent (Any): the agent being trained
        n_episodes (int): maximum number of training episodes
        batch_size (int): sampling batch size
        window_size (int): moving average window size
        warmup_steps (int): number of buffer warmup steps
        callbacks (List[TrainCallback]): applied training callbacks
    """
    output = NAME_STR.strip()
    params_str = f"{agent.active_params:,}/{agent.total_params:,}"

    if agent.active_params > 10_000:
        active, total = (
            number_to_short(agent.active_params),
            number_to_short(agent.total_params),
        )
        params_str += f" ({active}/{total})"

    cb_str = "\n\nActive Callbacks:"
    cb_str += "\n---------------------------------\n"
    cb_str += "\n".join(cb.info().lstrip() for cb in callbacks)
    cb_str += "\n---------------------------------\n"

    output += cb_str if callbacks else "\n\n"
    output += f"Training '{agent.__class__.__name__}' agent on '{agent.env.spec.id}' for '{number_to_short(n_episodes)}' episodes.\n"
    output += f"Using '{agent.buffer.__class__.__name__}' with 'capacity={number_to_short(agent.buffer.capacity)}'.\n"
    output += f"Sampling episodes with '{batch_size=}'.\n"
    output += f"Warming buffer with '{number_to_short(warmup_steps)}' samples before training starts.\n"
    output += f"Generating random values with a seed of '{agent.seed}'.\n"
    output += f"Running computations on device '{agent.device}'.\n"
    output += f"Moving averages computed based on 'window_size={number_to_short(window_size)}'.\n"
    output += f"Using networks with '{params_str}' active parameters.\n"
    output += "---------------------------------"

    print(output)
