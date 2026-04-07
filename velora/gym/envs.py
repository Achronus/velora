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

from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Callable, Dict, Iterator, List, Self, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium.vector import VectorEnv

from velora.gym.error import MissingPackageError
from velora.gym.make import (
    make_atari_env,
    make_box2d_env,
    make_dmc_env,
    make_mujoco_env,
)

MakeFn = Callable[..., VectorEnv]


@dataclass(frozen=True)
class EnvSpec:
    """
    Specification for a single environment in the scheduler.

    Parameters
    ----------
    name : str
        Gymnasium environment ID (e.g., `"ALE/Pong-v5"`)
    make_fn : MakeFn
        Factory function `(env_name, num_vec_envs) -> VectorEnv`
    category : str
        Environment category for scheduling constraints
    """

    name: str
    make_fn: MakeFn
    category: str


@dataclass
class EnvGroup:
    """
    Base environment group dataclass.

    Attributes
    ----------
    prefix : str
        The prefix for environment names (e.g., "ALE" for Atari)
    category : str
        Category name of the group
    envs : List[str]
        List of environment names in this group
    """

    prefix: str = ""
    category: str = ""
    version: str = "v5"
    required_packages: List[str] = field(default_factory=list)
    envs: List[str] = field(default_factory=list)

    @property
    def n_envs(self) -> int:
        """Number of environments in this group."""
        return len(self.envs)

    @property
    def make_fn(self) -> MakeFn:
        """Factory function for creating environments in this group."""
        raise NotImplementedError("Subclasses must implement make_fn")

    def get_name(self, env: str, version: str | None = None) -> str:
        """
        Get the full environment name with prefix and version.

        Parameters
        ----------
        env : str
            The base environment name
        version : str (optional)
            The version suffix. Default is `None` (uses group's version)

        Returns
        -------
        name : str
            Full environment name (e.g., "ALE/Breakout-v5")
        """
        raise NotImplementedError("Subclasses must implement get_name")

    def all_names(self, version: str | None = None) -> List[str]:
        """
        Get all full environment names in this group.

        Parameters
        ----------
        version : str (optional)
            The version suffix. Default is `None` (uses group's version)

        Returns
        -------
        names : List[str]
            List of full environment names
        """
        return [self.get_name(env, version) for env in self.envs]

    def __contains__(self, env: str) -> bool:
        """Check if an environment is in this group."""
        return env in self.envs

    def __getitem__(self, key: Union[int, slice]) -> "EnvGroup":
        """
        Slice the environment group to get a subset.

        Parameters
        ----------
        key : int | slice
            Index or slice for selecting environments

        Returns
        -------
        group : EnvGroup
            New EnvGroup with the selected environments
        """
        if isinstance(key, int):
            selected = [self.envs[key]]
        else:
            selected = self.envs[key]

        return self.__class__(prefix=self.prefix, envs=selected)

    def __iter__(self) -> Iterator[Tuple[str, MakeFn]]:
        """Iterate over (env_name, make_fn) tuples."""
        for env in self.envs:
            yield self.get_name(env), self.make_fn

    def __len__(self) -> int:
        """Return number of environments."""
        return len(self.envs)

    def check(self) -> Dict[str, bool]:
        """
        Check if required packages are installed.

        Returns
        -------
        status : Dict[str, bool]
            Mapping of package names to installation status
        """
        return {pkg: find_spec(pkg) is not None for pkg in self.required_packages}

    def is_available(self) -> bool:
        """
        Check if all required packages are installed.

        Returns
        -------
        available : bool
            True if all required packages are installed
        """
        return all(self.check().values())

    def as_specs(self) -> List[EnvSpec]:
        """
        Convert all environments in this group to scheduler specs.

        Returns
        -------
        specs : List[EnvSpec]
            One spec per environment in this group
        """
        return [
            EnvSpec(
                name=env_name,
                make_fn=make_fn,
                category=self.category,
            )
            for env_name, make_fn in self
        ]

    def dump(self) -> Dict[str, object]:
        """
        Serialize this group to a JSON-compatible dictionary.

        Captures the class name and module path so the exact subclass
        can be reconstructed via `load` regardless of where the
        class lives in the package hierarchy.

        Returns
        -------
        data : Dict[str, object]
            Serialized representation containing keys -
            `[class, module, prefix, category, version, required_packages, envs]`
        """
        return {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "prefix": self.prefix,
            "category": self.category,
            "version": self.version,
            "required_packages": self.required_packages,
            "envs": self.envs,
        }

    @classmethod
    def load(cls, data: Dict[str, object]) -> Self:
        """
        Reconstruct an `EnvGroup` subclass from a serialized dictionary.

        Uses the saved `module` and `class` keys to dynamically import
        and instantiate the correct subclass, so the exact environment
        group is restored even if its module path changes between versions.

        Parameters
        ----------
        data : Dict[str, object]
            Dictionary produced by `dump`

        Returns
        -------
        group : EnvGroup
            Reconstructed environment group instance

        Raises
        ------
        import_error : ImportError
            If the saved module cannot be imported
        module_error : AttributeError
            If the saved class is not found in the module
        """
        import importlib

        module = importlib.import_module(str(data["module"]))
        group_cls = getattr(module, data["class"])  # type: ignore

        return group_cls(
            prefix=data["prefix"],
            category=data["category"],
            version=data["version"],
            required_packages=data["required_packages"],
            envs=data["envs"],
        )


@dataclass
class AtariEnvs(EnvGroup):
    """
    [Atari Learning Environment](https://ale.farama.org/) (ALE).

    57 classic Atari 2600 games.
    """

    prefix: str = "ALE"
    category: str = "Atari"
    version: str = "v5"
    required_packages: List[str] = field(
        default_factory=lambda: ["gymnasium", "ale_py"]
    )
    envs: List[str] = field(
        default_factory=lambda: [
            "Alien",
            "Amidar",
            "Assault",
            "Asterix",
            "Asteroids",
            "Atlantis",
            "BankHeist",
            "BattleZone",
            "BeamRider",
            "Berzerk",
            "Bowling",
            "Boxing",
            "Breakout",
            "Centipede",
            "ChopperCommand",
            "CrazyClimber",
            "Defender",
            "DemonAttack",
            "DoubleDunk",
            "Enduro",
            "FishingDerby",
            "Freeway",
            "Frostbite",
            "Gopher",
            "Gravitar",
            "Hero",
            "IceHockey",
            "Jamesbond",
            "Kangaroo",
            "Krull",
            "KungFuMaster",
            "MontezumaRevenge",
            "MsPacman",
            "NameThisGame",
            "Phoenix",
            "Pitfall",
            "Pong",
            "PrivateEye",
            "Qbert",
            "Riverraid",
            "RoadRunner",
            "Robotank",
            "Seaquest",
            "Skiing",
            "Solaris",
            "SpaceInvaders",
            "StarGunner",
            "Surround",
            "Tennis",
            "TimePilot",
            "Tutankham",
            "UpNDown",
            "Venture",
            "VideoPinball",
            "WizardOfWor",
            "YarsRevenge",
            "Zaxxon",
        ]
    )

    @property
    def make_fn(self) -> MakeFn:
        """Factory function for creating Atari environments."""
        return make_atari_env

    def get_name(self, env: str, version: str | None = None) -> str:
        """Get full name: ALE/{env}-v5"""
        ver = version if version is not None else self.version
        return f"{self.prefix}/{env}-{ver}"


@dataclass
class MuJoCoEnvs(EnvGroup):
    """
    [MuJoCo](https://mujoco.org/) continuous control environments.

    13 standard continuous control benchmarks via Gymnasium v5.
    """

    prefix: str = ""
    category: str = "MuJoCo"
    version: str = "v5"
    required_packages: List[str] = field(
        default_factory=lambda: ["gymnasium", "mujoco"]
    )
    envs: List[str] = field(
        default_factory=lambda: [
            "Ant",
            "HalfCheetah",
            "Hopper",
            "Humanoid",
            "HumanoidStandup",
            "InvertedDoublePendulum",
            "InvertedPendulum",
            "Pusher",
            "Reacher",
            "Striker",
            "Swimmer",
            "Thrower",
            "Walker2d",
        ]
    )

    @property
    def make_fn(self) -> MakeFn:
        """Factory function for creating MuJoCo environments."""
        return make_mujoco_env

    def get_name(self, env: str, version: str | None = None) -> str:
        """Get full name: {env}-v5"""
        ver = version if version is not None else self.version
        return f"{env}-{ver}"


@dataclass
class DMCEnvs(EnvGroup):
    """
    [DeepMind Control Suite](https://github.com/google-deepmind/dm_control)
    environments via [shimmy](https://shimmy.farama.org/).

    21 continuous control environments with diverse dynamics and reward structures.
    """

    prefix: str = "dm_control"
    category: str = "DMC"
    version: str = "v0"
    required_packages: List[str] = field(
        default_factory=lambda: ["gymnasium", "shimmy", "dm_control"]
    )
    envs: List[str] = field(
        default_factory=lambda: [
            "acrobot-swingup",
            "ball_in_cup-catch",
            "cartpole-balance",
            "cartpole-swingup",
            "cheetah-run",
            "finger-spin",
            "finger-turn_easy",
            "finger-turn_hard",
            "fish-swim",
            "fish-upright",
            "hopper-hop",
            "hopper-stand",
            "humanoid-run",
            "humanoid-stand",
            "humanoid-walk",
            "pendulum-swingup",
            "point_mass-easy",
            "reacher-easy",
            "reacher-hard",
            "walker-stand",
            "walker-walk",
        ]
    )

    @property
    def make_fn(self) -> MakeFn:
        """Factory function for creating DMC environments."""
        return make_dmc_env

    def get_name(self, env: str, version: str | None = None) -> str:
        """Get full name: dm_control/{domain}-{task}-v0"""
        ver = version if version is not None else self.version
        return f"{self.prefix}/{env}-{ver}"


@dataclass
class Box2DEnvs(EnvGroup):
    """
    [Box2D](https://box2d.org/) and Gymnasium classic control environments
    with continuous action spaces.

    6 environments spanning procedural terrain, thrust control, and
    classic control problems.
    """

    prefix: str = ""
    category: str = "Box2D"
    version: str = ""
    required_packages: List[str] = field(
        default_factory=lambda: ["gymnasium", "box2d-py"]
    )
    envs: List[str] = field(
        default_factory=lambda: [
            "BipedalWalker-v3",
            "BipedalWalkerHardcore-v3",
            "CarRacing-v3",
            "LunarLanderContinuous-v3",
            "MountainCarContinuous-v0",
            "Pendulum-v1",
        ]
    )

    @property
    def make_fn(self) -> MakeFn:
        """Factory function for creating Box2D environments."""
        return make_box2d_env

    def get_name(self, env: str, version: str | None = None) -> str:
        """Get full name (version already included in env name)."""
        return env


class EnvSet:
    """
    A collection of environment groups for training across multiple suites.

    Combines multiple `EnvGroup` instances into a single iterable that yields
    `(env_name, make_fn)` tuples. Supports slicing and combining groups.

    Parameters
    ----------
    *groups : EnvGroup
        Variable number of environment groups to combine

    Examples
    --------
    >>> env_set = EnvSet(ATARI[:10], PROCGEN)
    >>> for env_name, make_fn in env_set:
    ...     envs = make_fn(env_name, num_envs=8)
    """

    def __init__(self, *groups: EnvGroup) -> None:
        self._groups: List[EnvGroup] = list(groups)

    @property
    def n_envs(self) -> int:
        """Total number of environments across all groups."""
        return sum(g.n_envs for g in self._groups)

    @property
    def groups(self) -> List[EnvGroup]:
        """List of environment groups in this set."""
        return self._groups

    def all_names(self, version: str | None = None) -> List[str]:
        """
        Get all full environment names across all groups.

        Parameters
        ----------
        version : str (optional)
            The version suffix. Default is `None`

        Returns
        -------
        names : List[str]
            List of full environment names
        """
        names = []
        for group in self._groups:
            names.extend(group.all_names(version))

        return names

    def unique_names(self, version: str | None = None) -> List[str]:
        """
        Get all unique environment names across all groups.

        Parameters
        ----------
        version : str (optional)
            The version suffix. Default is `None`

        Returns
        -------
        names : List[str]
            List of unique environment names
        """
        return list(set(self.all_names(version)))

    def as_list(self) -> List[Tuple[str, MakeFn]]:
        """
        Convert the environment set to a list of `(env_name, make_fn)` tuples.

        Returns
        -------
        env_specs : List[Tuple[str, MakeFn]]
            List of environment specifications
        """
        return list(self)

    def env_categories(self) -> Dict[str, int]:
        """
        Get a dictionary of environment categories and their counts.

        Returns
        -------
        categories : Dict[str, int]
            Mapping of category names to environment counts
        """
        counts: Dict[str, int] = {}
        for g in self._groups:
            counts[g.category] = counts.get(g.category, 0) + g.n_envs

        return counts

    def max_action_count(self, batch_size: int) -> int:
        """
        Probe each unique environment to determine the maximum action count.

        For discrete spaces (`gym.spaces.Discrete`), returns the max number of
        actions. For continuous spaces (`gym.spaces.Box`), returns the max
        action dimensionality.

        Parameters
        ----------
        batch_size : int
            Number of vectorized environments to create per probe

        Returns
        -------
        max_actions : int
            Maximum action count or dimensionality across all environments
        """
        max_actions = 0

        for env_name, make_fn in self.as_list():
            env = make_fn(env_name, batch_size)
            action_space = env.single_action_space

            if isinstance(action_space, gym.spaces.Discrete):
                max_actions = max(max_actions, action_space.n.item())
            elif isinstance(action_space, gym.spaces.Box):
                max_actions = max(max_actions, int(np.prod(action_space.shape)))

            env.close()

        return max_actions

    def __iter__(self) -> Iterator[Tuple[str, MakeFn]]:
        """Iterate over (env_name, make_fn) tuples from all groups."""
        for group in self._groups:
            yield from group

    def __len__(self) -> int:
        """Total number of environments."""
        return self.n_envs

    def __add__(self, other: Self) -> Self:
        """Combine two EnvSets."""
        return type(self)(*self._groups, *other._groups)

    def verify_packages(self) -> None:
        """
        Verify all required packages are installed for every environment group.

        Raises
        ------
        error : MissingPackageError
            If any group has missing required packages
        """
        missing = {}
        for group in self._groups:
            status = group.check()
            not_installed = [pkg for pkg, ok in status.items() if not ok]
            if not_installed:
                missing[group.category] = not_installed

        if missing:
            lines = [f"  {cat}: {', '.join(pkgs)}" for cat, pkgs in missing.items()]
            raise MissingPackageError(
                "Missing required packages for environment groups:\n" + "\n".join(lines)
            )

    def __repr__(self) -> str:
        group_info = ", ".join(
            f"{g.__class__.__name__}({g.n_envs})" for g in self._groups
        )
        return f"EnvSet({group_info}, total={self.n_envs})"

    def as_specs(self) -> List[EnvSpec]:
        """
        Convert all environments across all groups to scheduler specs.

        Returns
        -------
        specs : List[EnvSpec]
            One spec per environment across all groups
        """
        specs = []
        for group in self._groups:
            specs.extend(group.as_specs())

        return specs


# Pre-instantiated environment groups for convenience
ATARI_BASE = AtariEnvs(
    envs=[
        "Assault",
        "Atlantis",
        "Boxing",
        "Breakout",
        "CrazyClimber",
        "DemonAttack",
        "Gopher",
        "Kangaroo",
        "Krull",
        "NameThisGame",
        "RoadRunner",
        "Robotank",
        "StarGunner",
        "VideoPinball",
    ],
)
ATARI_EASY = AtariEnvs(
    envs=[
        "BeamRider",
        "Enduro",
        "FishingDerby",
        "Freeway",
        "Hero",
        "IceHockey",
        "Jamesbond",
        "KungFuMaster",
        "Phoenix",
        "Pong",
        "Qbert",
        "SpaceInvaders",
        "Tennis",
        "TimePilot",
        "Tutankham",
        "UpNDown",
    ],
)
ATARI_MEDIUM = AtariEnvs(
    envs=[
        "Alien",
        "Amidar",
        "Asterix",
        "BankHeist",
        "BattleZone",
        "Centipede",
        "ChopperCommand",
        "Defender",
        "Riverraid",
        "Seaquest",
        "Venture",
        "WizardOfWor",
        "Zaxxon",
    ],
)
ATARI_HARD = AtariEnvs(
    envs=[
        "Asteroids",
        "Berzerk",
        "Bowling",
        "DoubleDunk",
        "Frostbite",
        "Gravitar",
        "MontezumaRevenge",
        "MsPacman",
        "Pitfall",
        "PrivateEye",
        "Skiing",
        "Solaris",
        "Surround",
        "YarsRevenge",
    ],
)
ATARI_57 = AtariEnvs()

MUJOCO_LOCOMOTION = MuJoCoEnvs(
    envs=[
        "Ant",
        "HalfCheetah",
        "Hopper",
        "Humanoid",
        "HumanoidStandup",
        "Swimmer",
        "Walker2d",
    ],
)
MUJOCO_MANIPULATION = MuJoCoEnvs(
    envs=[
        "Pusher",
        "Reacher",
        "Striker",
        "Thrower",
    ],
)
MUJOCO_BALANCE = MuJoCoEnvs(
    envs=[
        "InvertedDoublePendulum",
        "InvertedPendulum",
    ],
)
MUJOCO_13 = MuJoCoEnvs()

DMC_SIMPLE = DMCEnvs(
    envs=[
        "acrobot-swingup",
        "cartpole-balance",
        "cartpole-swingup",
        "pendulum-swingup",
        "point_mass-easy",
    ],
)
DMC_MANIPULATION = DMCEnvs(
    envs=[
        "ball_in_cup-catch",
        "finger-spin",
        "finger-turn_easy",
        "finger-turn_hard",
        "reacher-easy",
        "reacher-hard",
    ],
)
DMC_LOCOMOTION = DMCEnvs(
    envs=[
        "cheetah-run",
        "fish-swim",
        "fish-upright",
        "hopper-hop",
        "hopper-stand",
        "walker-stand",
        "walker-walk",
    ],
)
DMC_COMPLEX = DMCEnvs(
    envs=[
        "humanoid-run",
        "humanoid-stand",
        "humanoid-walk",
    ],
)
DMC_21 = DMCEnvs()
BOX2D_6 = Box2DEnvs()

CONTINUOUS_40 = EnvSet(MUJOCO_13, DMC_21, BOX2D_6)
