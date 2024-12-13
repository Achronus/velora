from gymnasium import spaces
from pydantic import ConfigDict, validate_call
import numpy as np


@validate_call(
    config=ConfigDict(arbitrary_types_allowed=True),
    validate_return=True,
)
def get_obs_shape(obs_space: spaces.Space) -> tuple[int, ...]:
    """
    Gets the shape of the observation space (useful for buffers).

    Args:
        obs_space (gymnasium.spaces.Space): A [Gymnasium](https://gymnasium.farama.org/api/spaces/) environments observation space
    """
    if not isinstance(obs_space, (spaces.Discrete, spaces.Box)):
        raise NotImplementedError(f"'{obs_space}' observation space is not supported")

    if isinstance(obs_space, spaces.Box):
        return obs_space.shape

    return (obs_space.n,)


@validate_call(
    config=ConfigDict(arbitrary_types_allowed=True),
    validate_return=True,
)
def get_action_size(action_space: spaces.Space) -> int:
    """
    Gets the action space size.

    Args:
        action_space (gymnasium.spaces.Space): A [Gymnasium](https://gymnasium.farama.org/api/spaces/) environments action space
    """
    if not isinstance(action_space, (spaces.Discrete, spaces.Box)):
        raise NotImplementedError(f"'{action_space}' action space is not supported")

    if isinstance(action_space, spaces.Box):
        return np.prod(action_space.shape).item()

    return action_space.n


@validate_call(
    config=ConfigDict(arbitrary_types_allowed=True),
    validate_return=True,
)
def is_continuous(space: spaces.Space) -> bool:
    """Returns True if a space is continuous. Otherwise, False."""
    if isinstance(space, spaces.Box):
        return True

    return False
