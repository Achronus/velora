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
from typing import Callable, Dict, Iterator, List, Tuple, Union

from velora.gym.make import make_atari_env, make_dmlab_env, make_procgen_env
from velora.gym.wrappers import JaxConversion

MakeFn = Callable[..., JaxConversion]


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
class ProcgenEnvs(EnvGroup):
    """
    [Procgen Benchmark](https://github.com/openai/procgen).

    16 procedurally generated game environments.
    """

    prefix: str = "procgen:procgen"
    category: str = "Procgen"
    version: str = "v0"
    required_packages: List[str] = field(
        default_factory=lambda: ["gymnasium", "procgen"]
    )
    envs: List[str] = field(
        default_factory=lambda: [
            "bigfish",
            "bossfight",
            "caveflyer",
            "chaser",
            "climber",
            "coinrun",
            "dodgeball",
            "fruitbot",
            "heist",
            "jumper",
            "leaper",
            "maze",
            "miner",
            "ninja",
            "plunder",
            "starpilot",
        ]
    )

    @property
    def make_fn(self) -> MakeFn:
        """Factory function for creating Procgen environments."""
        return make_procgen_env

    def get_name(self, env: str, version: str | None = None) -> str:
        """Get full name: procgen:procgen-{env}-v0"""
        ver = version if version is not None else self.version
        return f"{self.prefix}-{env}-{ver}"


@dataclass
class DMLabEnvs(EnvGroup):
    """
    [DeepMind Lab](https://github.com/google-deepmind/lab) DMLab-30 benchmark.

    30 3D navigation and puzzle-solving environments.
    """

    prefix: str = "DMLab"
    category: str = "DMLab"
    version: str = ""
    required_packages: List[str] = field(
        default_factory=lambda: ["shimmy", "deepmind_lab"]
    )
    envs: List[str] = field(
        default_factory=lambda: [
            # Rooms (5 levels)
            "rooms_collect_good_objects_train",
            "rooms_exploit_deferred_effects_train",
            "rooms_select_nonmatching_object",
            "rooms_watermaze",
            "rooms_keys_doors_puzzle",
            # Language (4 levels)
            "language_select_described_object",
            "language_select_located_object",
            "language_execute_random_task",
            "language_answer_quantitative_question",
            # LaserTag (4 levels)
            "lasertag_one_opponent_small",
            "lasertag_three_opponents_small",
            "lasertag_one_opponent_large",
            "lasertag_three_opponents_large",
            # NatLab (3 levels)
            "natlab_fixed_large_map",
            "natlab_varying_map_regrowth",
            "natlab_varying_map_randomized",
            # SkyMaze (2 levels)
            "skymaze_irreversible_path_hard",
            "skymaze_irreversible_path_varied",
            # PsychLab (4 levels)
            "psychlab_arbitrary_visuomotor_mapping",
            "psychlab_continuous_recognition",
            "psychlab_sequential_comparison",
            "psychlab_visual_search",
            # Explore (8 levels)
            "explore_object_locations_small",
            "explore_object_locations_large",
            "explore_obstructed_goals_small",
            "explore_obstructed_goals_large",
            "explore_goal_locations_small",
            "explore_goal_locations_large",
            "explore_object_rewards_few",
            "explore_object_rewards_many",
        ]
    )

    @property
    def make_fn(self) -> MakeFn:
        """Factory function for creating DeepMind Lab environments."""
        return make_dmlab_env

    def get_name(self, env: str, version: str | None = None) -> str:
        """Get full name: just the level name for DMLab."""
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
        return {g.category: g.n_envs for g in self._groups}

    def __iter__(self) -> Iterator[Tuple[str, MakeFn]]:
        """Iterate over (env_name, make_fn) tuples from all groups."""
        for group in self._groups:
            yield from group

    def __len__(self) -> int:
        """Total number of environments."""
        return self.n_envs

    def __add__(self, other: "EnvSet") -> "EnvSet":
        """Combine two EnvSets."""
        return EnvSet(*self._groups, *other._groups)

    def __repr__(self) -> str:
        group_info = ", ".join(
            f"{g.__class__.__name__}({g.n_envs})" for g in self._groups
        )
        return f"EnvSet({group_info}, total={self.n_envs})"


# Pre-instantiated environment groups for convenience
ATARI = AtariEnvs()
PROCGEN = ProcgenEnvs()
DMLAB = DMLabEnvs()

DISCRETE_103 = EnvSet(ATARI, PROCGEN, DMLAB)
