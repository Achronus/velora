from typing import Literal

from velora.envs.mjlab import MjlabEnvSpec
from velora.envs.dmc_suite.paths import XMLS_DIR
from velora.envs.dmc_suite.specs.acrobot import AcrobotSwingUp, AcrobotSwingUpSparse
from velora.envs.dmc_suite.specs.humanoid import (
    HumanoidRun,
    HumanoidStand,
    HumanoidWalk,
)

DMControlEnvNames = Literal[
    "dm_control/acrobot-swingup-v0",
    "dm_control/acrobot-swingup_sparse-v0",
    # "dm_control/ball_in_cup-catch-v0",
    # "dm_control/cartpole-balance-v0",
    # "dm_control/cartpole-balance_sparse-v0",
    # "dm_control/cartpole-swingup-v0",
    # "dm_control/cartpole-swingup_sparse-v0",
    # "dm_control/cheetah-run-v0",
    # "dm_control/finger-spin-v0",
    # "dm_control/finger-turn_easy-v0",
    # "dm_control/finger-turn_hard-v0",
    # "dm_control/fish-swim-v0",
    # "dm_control/hopper-hop-v0",
    # "dm_control/hopper-stand-v0",
    "dm_control/humanoid-run-v0",
    "dm_control/humanoid-stand-v0",
    "dm_control/humanoid-walk-v0",
    # "dm_control/pendulum-swingup-v0",
    # "dm_control/point_mass-easy-v0",
    # "dm_control/reacher-easy-v0",
    # "dm_control/reacher-hard-v0",
    # "dm_control/swimmer-swimmer6-v0",
    # "dm_control/walker-run-v0",
    # "dm_control/walker-stand-v0",
    # "dm_control/walker-walk-v0",
]

DMC_SPECS: dict[DMControlEnvNames, MjlabEnvSpec] = {
    "dm_control/acrobot-swingup-v0": AcrobotSwingUp(),
    "dm_control/acrobot-swingup_sparse-v0": AcrobotSwingUpSparse(),
    "dm_control/humanoid-run-v0": HumanoidRun(),
    "dm_control/humanoid-stand-v0": HumanoidStand(),
    "dm_control/humanoid-walk-v0": HumanoidWalk(),
}

__all__ = [
    "DMC_SPECS",
    "XMLS_DIR",
    "AcrobotSwingUp",
    "AcrobotSwingUpSparse",
    "DMControlEnvNames",
    "HumanoidRun",
    "HumanoidStand",
    "HumanoidWalk",
]
