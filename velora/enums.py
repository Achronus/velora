from enum import StrEnum


class RenderMode(StrEnum):
    """Render modes for the [gymnasium.Env.render()](https://gymnasium.farama.org/api/env/#gymnasium.Env.render) method."""

    HUMAN = "human"
    RGB_ARRAY = "rgb_array"
    ANSI = "ANSI"
    RGB_ARRAY_LIST = "rgb_array_list"
    ANSI_LIST = "ansi_list"
