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


from abc import ABC, abstractmethod
from typing import ClassVar

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg


class MjlabEnvSpec(ABC):
    """
    A base class for `mjlab` environment specifications.

    Each specification owns a single task and builds the `mjlab` config
    that describes it. `config` performs the assembly, drawing on
    smaller methods that each contribute one part of it.

    Only `_scene` is required, as a task cannot exist without entities
    to populate. It's recommend to use the same convention for other
    parts of the `ManagerBasedRlEnvCfg`, so methods such as `_observations`,
    `_actions`,`_rewards`, `_terminations`, `_events`, etc. for readability.
    This is left at the discretion of the developer.

    Variants of a task subclass this specification and override only the
    parts that differ.

    Attributes
    ----------
    name : str
        The environment ID the specification is registered under (e.g.,
        `dm_control/acrobot-swingup-v0`)
    """

    name: ClassVar[str]

    @abstractmethod
    def config(self) -> ManagerBasedRlEnvCfg:
        """
        Builds the environment config.

        Assembles a new config on every call, leaving callers free to
        mutate it.

        Returns
        -------
        cfg : ManagerBasedRlEnvCfg
            The assembled environment config
        """
        ...

    @abstractmethod
    def _scene(self) -> SceneCfg:
        """
        Builds the scene the task takes place in.

        Returns
        -------
        cfg : SceneCfg
            The task's entities, terrain, and sensors, along with the
            number of parallel environments and their spacing
        """
        ...
