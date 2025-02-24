import re
from dataclasses import dataclass
from typing import Dict, List, Literal

import gymnasium as gym


@dataclass
class EnvResult:
    """
    Container for environment search results.

    Parameters:
        name (str): the name of the environment
        type (str): the environment type, either `discrete` or `continuous`
    """

    name: str
    type: str


class SearchHelper:
    """
    A helper class containing methods for creating the `EnvSearch` cache.
    """

    @staticmethod
    def get_env_type(env_name: str) -> Literal["discrete", "continuous"] | None:
        """
        Determines if an environment has a discrete or continuous action space.

        Parameters:
            env_name (str): name of the environment

        Returns:
            env_type (str | None): one of three values -

            - `discrete` for Discrete action space.
            - `continuous` for Box action space.
            - `None`, otherwise.
        """
        try:
            env = gym.make(env_name)

            if isinstance(env.action_space, gym.spaces.Discrete):
                return "discrete"
            elif isinstance(env.action_space, gym.spaces.Box):
                return "continuous"
            return None

        except Exception as e:
            raise e
        finally:
            if "env" in locals():
                env.close()

    @staticmethod
    def get_latest_env_names() -> List[str]:
        """
        Builds a list of the latest [Gymnasium](https://gymnasium.farama.org/) environment names.

        Returns:
            names (List[str]): a list of names for all latest env versions.
        """
        env_dict: Dict[str, int] = {}

        for env_name in gym.envs.registry.keys():
            # Skip envs with paths (JAX-based) or old Gym versions
            if "/" in env_name or env_name.startswith("GymV"):
                continue

            match = re.match(r"(.+)-v(\d+)$", env_name)

            if match:
                base_name, version = match.groups()
                version = int(version)

            # Keep only the latest version
            if base_name not in env_dict or version > env_dict[base_name]:
                env_dict[base_name] = version

        return [f"{name}-v{version}" for name, version in env_dict.items()]


class EnvSearch:
    """
    A utility class for searching for [Gymnasium](https://gymnasium.farama.org/)
    environments.
    """

    _result_cache: List[EnvResult] = []
    _helper = SearchHelper()

    @classmethod
    def _build_name_cache(cls) -> None:
        """
        Builds the environment name cache storing the latest
        [Gymnasium](https://gymnasium.farama.org/) environment names.
        """
        if cls._result_cache:
            return

        names = cls._helper.get_latest_env_names()
        print(f"Caching {len(names)} environments...", end="")
        results = []

        for env in names:
            env_type = cls._helper.get_env_type(env)
            results.append(EnvResult(name=env, type=env_type))

        cls._result_cache = results
        print("Complete.")

    @classmethod
    def find(cls, query: str) -> List[EnvResult]:
        """
        Find a [Gymnasium](https://gymnasium.farama.org/) environment that contains `query`.

        Parameters:
            query (str): partial or complete name of an environment
                (e.g., `Lunar` or `Pendulum`)

        Returns:
            result (List[EnvResult]): a list of environment results matching the query.
        """
        cls._build_name_cache()

        matches = [env for env in cls._result_cache if query in env.name]
        print(f"{len(matches)} environments found.")
        return matches

    @classmethod
    def discrete(cls) -> List[EnvResult]:
        """
        Get all available discrete [Gymnasium](https://gymnasium.farama.org/) environments.

        Returns:
            names (List[EnvResult]): a list of available discrete environments.
        """
        cls._build_name_cache()

        matches = [env for env in cls._result_cache if env.type == "discrete"]
        print(f"{len(matches)} environments found.")
        return matches

    @classmethod
    def continuous(cls) -> List[EnvResult]:
        """
        Get all available continuous [Gymnasium](https://gymnasium.farama.org/) environments.

        Returns:
            names (List[EnvResult]): a list of available continuous environments.
        """
        cls._build_name_cache()

        matches = [env for env in cls._result_cache if env.type == "continuous"]
        print(f"{len(matches)} environments found.")
        return matches
