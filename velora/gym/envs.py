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

from gymnasium.vector import VectorEnv

from velora.gym.error import MissingPackageError
from velora.gym.make import make_atari_env

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
ATARI = AtariEnvs()
