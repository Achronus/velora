import mujorax  # noqa: F401 — registers envs in envrax's registry

from velora import base, cli, compute, disco, lnn, nn, tracking, utils
from velora.disco import (
    DiscoAgent,
    DiscoAgentSettings,
    DiscoValueSettings,
    EMASettings,
    LossCostSettings,
    PolicyAgentSettings,
    RuleTrainer,
    RuleTrainerSettings,
)
from velora.tracking import RunSettings

__all__ = [
    "RuleTrainer",
    "RuleTrainerSettings",
    "DiscoAgent",
    "PolicyAgentSettings",
    "DiscoAgentSettings",
    "DiscoValueSettings",
    "EMASettings",
    "LossCostSettings",
    "RunSettings",
]
