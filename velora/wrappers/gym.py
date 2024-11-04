from pydantic import BaseModel, ConfigDict
from typing import Any

ValidWrappers = []


class GymWrapperModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class AtariPreprocessing(GymWrapperModel):
    noop_max: int = 30
    frame_skip: int = 4
    screen_size: int | tuple[int, int] = 84
    terminal_on_life_loss: bool = False
    grayscale_obs: bool = True
    grayscale_newaxis: bool = False
    scale_obs: bool = False


class Autoreset(GymWrapperModel):
    pass


class ClipAction(GymWrapperModel):
    pass


class ClipReward(GymWrapperModel):
    lower_bound: float
    upper_bound: float


class DelayObservation(GymWrapperModel):
    delay: int


class DtypeObservation(GymWrapperModel):
    dtype: str


class FilterObservation(GymWrapperModel):
    keys: list[str]


class FlattenObservation(GymWrapperModel):
    pass


class FrameStack(GymWrapperModel):
    num_stack: int
    lz4_compress: bool = False


class GrayScaleObservation(GymWrapperModel):
    keep_dim: bool = False


class NormalizeObservation(GymWrapperModel):
    epsilon: float = 1e-8


class NormalizeReward(GymWrapperModel):
    gamma: float = 0.99
    epsilon: float = 1e-8


class RecordEpisodeStatistics(GymWrapperModel):
    pass


class ResizeObservation(GymWrapperModel):
    shape: tuple[int, int]


class RescaleAction(GymWrapperModel):
    min_action: float = -1.0
    max_action: float = 1.0


class RescaleObservation(GymWrapperModel):
    min_obs: float
    max_obs: float


class RewardToCost(GymWrapperModel):
    pass


class ScaleObservation(GymWrapperModel):
    scale: float = 1.0


class TimeAwareObservation(GymWrapperModel):
    pass


class TransformObservation(GymWrapperModel):
    f: Any


class TransformReward(GymWrapperModel):
    f: Any


class VectorListInfo(GymWrapperModel):
    pass
