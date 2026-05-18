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

from collections import defaultdict
from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Callable, Dict, Iterator, List, Self, Union

import gymnasium as gym
import numpy as np
from gymnasium.vector import VectorEnv

from velora.gym.error import MissingPackageError
from velora.gym.make import make_box2d_env, make_dmc_env, make_mujoco_env

MakeFn = Callable[..., VectorEnv]


@dataclass(frozen=True)
class EnvSpec:
    """
    Specification for a single environment — the unit of registration.

    Mirrors `envrax.EnvSpec` in shape so `velora.gym.make(name)` is
    call-site compatible with `envrax.make(name)`. Unlike envrax, which
    stores a `JaxEnv` class, this spec stores a Gymnasium factory
    (`make_fn`) returning a `gymnasium.vector.VectorEnv`.

    Parameters
    ----------
    name : str
        Canonical environment ID (e.g. `"Ant-v5"`,
        `"dm_control/cartpole-balance-v0"`).
    make_fn : MakeFn
        Factory function `(name, num_envs, **kwargs) -> VectorEnv`.
    suite : str
        Suite category tag (e.g. `"MuJoCo"`). Populated by
        `register_suite` from the parent `EnvSuite.category`.
    """

    name: str
    make_fn: MakeFn
    suite: str = ""


@dataclass
class EnvSuite:
    """
    A named, versioned collection of environments from one Gymnasium suite.

    Mirrors `envrax.EnvSuite` shape. Subclasses pin `prefix`, `category`,
    `version`, `required_packages`, override `make_fn` to return their
    factory, and override `get_name` to produce canonical IDs.

    Parameters
    ----------
    prefix : str
        Namespace prefix for environment names (e.g. `"dm_control"`).
    category : str
        Human-readable category label (e.g. `"MuJoCo"`).
    version : str
        Version suffix applied by `get_name` (e.g. `"v5"`). Default is `"v0"`.
    required_packages : List[str]
        Python packages that must be importable for this suite to work.
    envs : List[str]
        Short environment names. The canonical `EnvSpec.name` is produced
        by `get_name`.
    """

    prefix: str = ""
    category: str = ""
    version: str = "v0"
    required_packages: List[str] = field(default_factory=list)
    envs: List[str] = field(default_factory=list)

    @property
    def n_envs(self) -> int:
        """Number of environments in this suite."""
        return len(self.envs)

    @property
    def make_fn(self) -> MakeFn:
        """Factory function for creating environments in this suite."""
        raise NotImplementedError("Subclasses must implement make_fn")

    @property
    def specs(self) -> List[EnvSpec]:
        """
        `EnvSpec` instances for every environment in this suite.

        Mirrors `envrax.EnvSuite.specs`. Each spec's `name` is the
        canonical ID produced by `get_name`.

        Returns
        -------
        specs : List[EnvSpec]
            One spec per environment, all sharing this suite's `make_fn`.
        """
        return [
            EnvSpec(name=self.get_name(e), make_fn=self.make_fn, suite=self.category)
            for e in self.envs
        ]

    def get_name(self, env: str, version: str | None = None) -> str:
        """
        Return the canonical ID for a single environment.

        Parameters
        ----------
        env : str
            Short environment name.
        version : str (optional)
            Override the suite's default version suffix.

        Returns
        -------
        name : str
            Canonical environment ID (e.g. `"Ant-v5"`).
        """
        raise NotImplementedError("Subclasses must implement get_name")

    def all_names(self, version: str | None = None) -> List[str]:
        """
        Canonical IDs for every environment in this suite.

        Parameters
        ----------
        version : str (optional)
            Override the suite's default version suffix.

        Returns
        -------
        names : List[str]
            One canonical ID per environment.
        """
        return [self.get_name(e, version) for e in self.envs]

    def __contains__(self, env: str) -> bool:
        return env in self.envs

    def __getitem__(self, key: Union[int, slice]) -> "EnvSuite":
        """
        Return a new suite containing only the selected environment(s).

        Parameters
        ----------
        key : int | slice
            Index or slice into `self.envs`.

        Returns
        -------
        suite : EnvSuite
            Same subclass as `self`, with the subset of envs.
        """
        if isinstance(key, int):
            selected = [self.envs[key]]
        else:
            selected = self.envs[key]

        return self.__class__(
            prefix=self.prefix,
            category=self.category,
            version=self.version,
            required_packages=self.required_packages,
            envs=selected,
        )

    def __iter__(self) -> Iterator[str]:
        """Yield canonical ID strings."""
        for env in self.envs:
            yield self.get_name(env)

    def __len__(self) -> int:
        return len(self.envs)

    def check(self) -> Dict[str, bool]:
        """
        Whether each required package is importable.

        Returns
        -------
        status : Dict[str, bool]
            Mapping of package name to installation status.
        """
        return {pkg: find_spec(pkg) is not None for pkg in self.required_packages}

    def is_available(self) -> bool:
        """Whether all required packages are installed."""
        return all(self.check().values())

    def dump(self) -> Dict[str, object]:
        """
        Serialize this suite to a JSON-compatible dictionary.

        Captures the class name and module path so the exact subclass
        can be reconstructed via `load` regardless of where the class
        lives in the package hierarchy.

        Returns
        -------
        data : Dict[str, object]
            Serialized representation containing keys -
            `[class, module, prefix, category, version, required_packages, envs]`.
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
        Reconstruct an `EnvSuite` subclass from a serialized dictionary.

        Parameters
        ----------
        data : Dict[str, object]
            Dictionary produced by `dump`.

        Returns
        -------
        suite : EnvSuite
            Reconstructed environment suite instance.
        """
        import importlib

        module = importlib.import_module(str(data["module"]))
        suite_cls = getattr(module, data["class"])  # type: ignore

        return suite_cls(
            prefix=data["prefix"],
            category=data["category"],
            version=data["version"],
            required_packages=data["required_packages"],
            envs=data["envs"],
        )


@dataclass
class MuJoCoEnvs(EnvSuite):
    """
    [MuJoCo](https://mujoco.org/) continuous control environments.

    11 standard continuous control benchmarks via Gymnasium v5.
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
            "Swimmer",
            "Walker2d",
        ]
    )

    @property
    def make_fn(self) -> MakeFn:
        return make_mujoco_env

    def get_name(self, env: str, version: str | None = None) -> str:
        ver = version if version is not None else self.version
        return f"{env}-{ver}"


@dataclass
class DMCEnvs(EnvSuite):
    """
    [DeepMind Control Suite](https://github.com/google-deepmind/dm_control)
    environments.

    25 continuous control environments with diverse dynamics and reward
    structures, accessed via `dm_control.suite` directly (bypasses
    `dm_control.locomotion` and its labmaze dependency).
    """

    prefix: str = "dm_control"
    category: str = "DMC"
    version: str = "v0"
    required_packages: List[str] = field(
        default_factory=lambda: ["gymnasium", "dm_control"]
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
            "point_mass-hard",
            "reacher-easy",
            "reacher-hard",
            "swimmer-swimmer6",
            "swimmer-swimmer15",
            "walker-run",
            "walker-stand",
            "walker-walk",
        ]
    )

    @property
    def make_fn(self) -> MakeFn:
        return make_dmc_env

    def get_name(self, env: str, version: str | None = None) -> str:
        ver = version if version is not None else self.version
        return f"{self.prefix}/{env}-{ver}"


@dataclass
class Box2DEnvs(EnvSuite):
    """
    [Box2D](https://box2d.org/) and Gymnasium classic control environments
    with continuous action spaces.

    5 environments spanning procedural terrain, thrust control, and classic
    control problems. Version suffix is baked into each env name so
    `get_name` is the identity function.
    """

    prefix: str = ""
    category: str = "Box2D"
    version: str = ""
    required_packages: List[str] = field(default_factory=lambda: ["gymnasium", "Box2D"])
    envs: List[str] = field(
        default_factory=lambda: [
            "BipedalWalker-v3",
            "BipedalWalkerHardcore-v3",
            "LunarLanderContinuous-v3",
            "MountainCarContinuous-v0",
            "Pendulum-v1",
        ]
    )

    @property
    def make_fn(self) -> MakeFn:
        return make_box2d_env

    def get_name(self, env: str, version: str | None = None) -> str:
        return env


class _RegisteredSuite(EnvSuite):
    """Suite whose envs already hold canonical IDs — used by `EnvSet.from_names`.

    Resolves `specs` directly from the registry so each env can carry
    its own `make_fn` even when grouped under one suite.
    """

    def get_name(self, env: str, version: str | None = None) -> str:
        return env

    @property
    def make_fn(self) -> MakeFn:
        raise NotImplementedError(
            "_RegisteredSuite has no single make_fn; use spec.make_fn per env."
        )

    @property
    def specs(self) -> List[EnvSpec]:
        return [get_spec(e) for e in self.envs]


class EnvSet:
    """
    An ordered collection of `EnvSuite` instances.

    Mirrors `envrax.EnvSet`. Yields canonical ID strings when iterated
    and supports merging two sets with `+`.

    Parameters
    ----------
    *suites : EnvSuite
        Variable number of environment suites to combine.

    Examples
    --------
    >>> env_set = EnvSet(MUJOCO_11, BOX2D_5)
    >>> for name in env_set:
    ...     env = make(name, num_envs=8)
    """

    def __init__(self, *suites: EnvSuite) -> None:
        self._suites: List[EnvSuite] = list(suites)

    @property
    def n_envs(self) -> int:
        """Total number of environments across all suites."""
        return sum(s.n_envs for s in self._suites)

    @property
    def suites(self) -> List[EnvSuite]:
        """List of environment suites in this set."""
        return self._suites

    def all_names(self, version: str | None = None) -> List[str]:
        """
        Canonical IDs for every environment across all suites.

        Parameters
        ----------
        version : str (optional)
            Override the default version suffix for all suites.

        Returns
        -------
        names : List[str]
            One canonical ID per environment.
        """
        names: List[str] = []
        for suite in self._suites:
            names.extend(suite.all_names(version))
        return names

    def unique_names(self, version: str | None = None) -> List[str]:
        """
        Unique canonical IDs across all suites.

        Parameters
        ----------
        version : str (optional)
            Override the default version suffix.

        Returns
        -------
        names : List[str]
            List of unique canonical IDs.
        """
        return list(set(self.all_names(version)))

    def as_specs(self) -> List[EnvSpec]:
        """
        Flat list of every `EnvSpec` across all suites.

        Returns
        -------
        specs : List[EnvSpec]
            One spec per environment in registration order.
        """
        specs: List[EnvSpec] = []
        for suite in self._suites:
            specs.extend(suite.specs)
        return specs

    def as_list(self) -> List[tuple]:
        """
        Flat list of `(canonical_name, make_fn)` tuples across all suites.

        Returns
        -------
        env_list : List[Tuple[str, MakeFn]]
            One `(name, make_fn)` per environment in registration order.
        """
        return [(spec.name, spec.make_fn) for spec in self.as_specs()]

    @property
    def groups(self) -> List[EnvSuite]:
        """Alias for `suites` retained for downstream compatibility."""
        return self._suites

    def env_categories(self) -> Dict[str, int]:
        """
        Mapping of category name to environment count.

        Returns
        -------
        categories : Dict[str, int]
            One entry per distinct `EnvSuite.category` across this set.
        """
        counts: Dict[str, int] = {}
        for s in self._suites:
            counts[s.category] = counts.get(s.category, 0) + s.n_envs
        return counts

    def max_action_count(self, batch_size: int) -> int:
        """
        Probe each environment to determine the maximum action count.

        For `gym.spaces.Discrete`, returns the max number of actions.
        For `gym.spaces.Box`, returns the max action dimensionality.

        Parameters
        ----------
        batch_size : int
            Number of vectorized environments to create per probe.

        Returns
        -------
        max_actions : int
            Maximum action count or dimensionality across all environments.
        """
        max_actions = 0

        for spec in self.as_specs():
            env = spec.make_fn(spec.name, batch_size)
            action_space = env.single_action_space

            if isinstance(action_space, gym.spaces.Discrete):
                max_actions = max(max_actions, action_space.n.item())
            elif isinstance(action_space, gym.spaces.Box):
                max_actions = max(max_actions, int(np.prod(action_space.shape)))

            env.close()

        return max_actions

    def max_obs_dim(self, batch_size: int) -> int:
        """
        Probe each environment to determine the maximum observation
        dimensionality.

        For 1D `gym.spaces.Box` observations, returns the max obs dim.
        For 3D image observations, returns `0` (assumed homogeneous via
        preprocessing).

        Parameters
        ----------
        batch_size : int
            Number of vectorized environments to create per probe.

        Returns
        -------
        max_obs : int
            Maximum observation dimensionality across all environments,
            or `0` if all observations are images.
        """
        max_obs = 0

        for spec in self.as_specs():
            env = spec.make_fn(spec.name, batch_size)
            obs_space = env.single_observation_space

            if isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 1:
                max_obs = max(max_obs, obs_space.shape[0])

            env.close()

        return max_obs

    def __iter__(self) -> Iterator[str]:
        """Yield canonical ID strings from all suites in order."""
        for suite in self._suites:
            yield from suite

    def __len__(self) -> int:
        return self.n_envs

    def __add__(self, other: Self) -> Self:
        return type(self)(*self._suites, *other._suites)

    @classmethod
    def from_names(cls, names: List[str]) -> Self:
        """
        Build an `EnvSet` from a list of registered canonical IDs.

        Names are looked up via the registry and grouped by their suite
        category tag (`EnvSpec.suite`). Used to reconstruct an `EnvSet`
        from persisted state without needing the original suite class
        hierarchy.

        Parameters
        ----------
        names : List[str]
            Registered canonical env IDs.

        Returns
        -------
        env_set : EnvSet
            One `_RegisteredSuite` per distinct category, holding matching specs.
        """
        by_cat: Dict[str, List[str]] = defaultdict(list)
        for name in names:
            spec = get_spec(name)
            by_cat[spec.suite].append(spec.name)

        suites = [
            _RegisteredSuite(category=category, envs=env_names)
            for category, env_names in by_cat.items()
        ]
        return cls(*suites)

    def verify_packages(self) -> None:
        """
        Verify all required packages are installed for every suite.

        Raises
        ------
        error : MissingPackageError
            If any suite has missing required packages.
        """
        missing: Dict[str, List[str]] = {}
        for suite in self._suites:
            status = suite.check()
            not_installed = [pkg for pkg, ok in status.items() if not ok]
            if not_installed:
                missing[suite.category] = not_installed

        if missing:
            lines = [f"  {cat}: {', '.join(pkgs)}" for cat, pkgs in missing.items()]
            raise MissingPackageError(
                "Missing required packages for environment suites:\n" + "\n".join(lines)
            )

    def __repr__(self) -> str:
        suite_info = ", ".join(
            f"{s.__class__.__name__}({s.n_envs})" for s in self._suites
        )
        return f"EnvSet({suite_info}, total={self.n_envs})"


_REGISTRY: Dict[str, EnvSpec] = {}


def register(name: str, make_fn: MakeFn, *, suite: str = "") -> None:
    """
    Register a single environment in the `velora.gym` registry.

    Mirrors `envrax.register`. After registration, the env can be
    instantiated via `velora.gym.make(name)`.

    Parameters
    ----------
    name : str
        Canonical environment ID (e.g. `"Ant-v5"`).
    make_fn : MakeFn
        Factory `(name, num_envs, **kwargs) -> VectorEnv`.
    suite : str (optional)
        Suite category tag for introspection.

    Raises
    ------
    env_exists : ValueError
        If `name` is already registered.
    """
    if name in _REGISTRY:
        raise ValueError(f"Env '{name}' is already registered")
    _REGISTRY[name] = EnvSpec(name=name, make_fn=make_fn, suite=suite)


def register_suite(suite: EnvSuite, *, version: str | None = None) -> None:
    """
    Register every environment in an `EnvSuite` in one shot.

    Mirrors `envrax.register_suite`. Skips environments whose canonical
    ID is already registered (idempotent).

    Parameters
    ----------
    suite : EnvSuite
        Suite whose environments should be registered.
    version : str (optional)
        Override the suite's default version when computing canonical IDs.
    """
    for env in suite.envs:
        canonical = suite.get_name(env, version=version)
        if canonical in _REGISTRY:
            continue
        _REGISTRY[canonical] = EnvSpec(
            name=canonical,
            make_fn=suite.make_fn,
            suite=suite.category,
        )


def registered_names() -> List[str]:
    """Sorted list of every canonical ID currently in the registry."""
    return sorted(_REGISTRY.keys())


def registry() -> Dict[str, EnvSpec]:
    """Shallow copy of the registry mapping (canonical ID → `EnvSpec`)."""
    return dict(_REGISTRY)


def get_spec(name: str) -> EnvSpec:
    """
    Return the full `EnvSpec` for a registered environment.

    Mirrors `envrax.get_spec`.

    Parameters
    ----------
    name : str
        Registered canonical environment ID.

    Returns
    -------
    spec : EnvSpec
        Registered specification.

    Raises
    ------
    unknown_env : ValueError
        If `name` is not registered.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Env '{name}' is not registered")
    return _REGISTRY[name]


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
    ],
)
MUJOCO_BALANCE = MuJoCoEnvs(
    envs=[
        "InvertedDoublePendulum",
        "InvertedPendulum",
    ],
)
MUJOCO_11 = MuJoCoEnvs()

DMC_SIMPLE = DMCEnvs(
    envs=[
        "acrobot-swingup",
        "cartpole-balance",
        "cartpole-swingup",
        "pendulum-swingup",
        "point_mass-easy",
        "point_mass-hard",
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
        "swimmer-swimmer6",
        "swimmer-swimmer15",
        "walker-run",
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
DMC_25 = DMCEnvs()
BOX2D_5 = Box2DEnvs()

CONTINUOUS_41 = EnvSet(MUJOCO_11, DMC_25, BOX2D_5)
