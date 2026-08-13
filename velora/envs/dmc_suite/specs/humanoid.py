# Copyright 2026 Achronus
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


import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import ClassVar, Literal

import torch
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import reset_joints_by_offset, time_out
from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.envs.mdp.observations import builtin_sensor
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from velora.envs.base import MjlabEnvSpec
from velora.envs.dmc_suite.paths import XMLS_DIR
from velora.envs.utils import get_spec, resolved_ids

_HUMANOID_XML: Path = XMLS_DIR / "humanoid.xml"

Sigmoid = Literal["gaussian", "linear", "quadratic"]


@dataclass(frozen=True)
class HumanoidCfg:
    """
    Tunable quantities for Humanoid tasks.

    Every default reproduces `dm_control.suite.humanoid`.

    Parameters
    ----------
    stand_height : float (optional)
        The head height at which standing scores `1`. Default is `1.4`
    upright_bound : float (optional)
        The torso uprightness above which the posture scores `1`.
        Default is `0.9`
    upright_margin : float (optional)
        The uprightness falloff width. Default is `1.9`
    control_margin : float (optional)
        The actuation magnitude at which the effort penalty bottoms
        out. Default is `1.0`
    dont_move_margin : float (optional)
        The horizontal speed at which the stand task stops rewarding
        stillness. Default is `2.0`
    time_limit_s : float (optional)
        The episode duration in seconds. Default is `25.0`
    reset_position_range : tuple[float, float] (optional)
        The range joints are offset by on reset. Default is
        `(-pi, pi)`
    reset_velocity_range : tuple[float, float] (optional)
        The range joint velocities are drawn from on reset. Default is
        `(0.0, 0.0)`
    """

    stand_height: float = 1.4
    upright_bound: float = 0.9
    upright_margin: float = 1.9
    control_margin: float = 1.0
    dont_move_margin: float = 2.0
    time_limit_s: float = 25.0
    reset_position_range: tuple[float, float] = (-math.pi, math.pi)
    reset_velocity_range: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class HumanoidParts:
    """
    Describes the parts of the Humanoid model that terms select against.

    Holds names declared in the model's MJCF, which the specification
    turns into `SceneEntityCfg` selections. A single name is one
    element, while a tuple is an ordered group.

    Parameters
    ----------
    name : str (optional)
        The key the model is registered under in the scene. Default is
        `humanoid`
    torso_body : str (optional)
        The body every egocentric measurement is taken relative to.
        Default is `torso`
    head_body : str (optional)
        The body whose height decides whether the humanoid is standing.
        Default is `head`
    extremity_bodies : tuple[str, ...] (optional)
        The four end effectors, ordered left before right and hand
        before foot to match `dm_control`. Default is
        `("left_hand", "left_foot", "right_hand", "right_foot")`
    com_sensor : str (optional)
        The subtree linear velocity sensor giving centre of mass
        velocity. Default is `torso_subtreelinvel`
    """

    name: str = "humanoid"
    torso_body: str = "torso"
    head_body: str = "head"
    extremity_bodies: tuple[str, ...] = (
        "left_hand",
        "left_foot",
        "right_hand",
        "right_foot",
    )
    com_sensor: str = "torso_subtreelinvel"


def _rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
    """
    Converts MuJoCo quaternions into rotation matrices.

    Parameters
    ----------
    quat : torch.Tensor
        Quaternions in `(w, x, y, z)` order `(num_envs, 4)`

    Returns
    -------
    matrix : torch.Tensor
        Body to world rotation matrices `(num_envs, 3, 3)`
    """
    w, x, y, z = quat.unbind(-1)

    return torch.stack(
        [
            torch.stack(
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1
            ),
            torch.stack(
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1
            ),
            torch.stack(
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1
            ),
        ],
        dim=-2,
    )


def _tolerance(
    x: torch.Tensor,
    lower: float,
    upper: float,
    margin: float,
    sigmoid: Sigmoid,
    value_at_margin: float,
) -> torch.Tensor:
    """
    The `dm_control` tolerance shaping function.

    Scores `1` while `x` lies within the bounds and decays towards `0`
    outside them, reaching `value_at_margin` once `x` is `margin` away
    from the nearest bound.

    Parameters
    ----------
    x : torch.Tensor
        The quantity being shaped
    lower : float
        The inclusive lower bound
    upper : float
        The inclusive upper bound
    margin : float
        The distance beyond a bound at which the score reaches
        `value_at_margin`. A margin of `0` gives a hard indicator
    sigmoid : Sigmoid
        The falloff shape applied beyond the bounds
    value_at_margin : float
        The score at exactly `margin` beyond a bound

    Returns
    -------
    value : torch.Tensor
        The shaped score, in `[0, 1]`

    Raises
    ------
    unknown_sigmoid : ValueError
        Error when `sigmoid` is not a supported falloff
    """
    in_bounds = (x >= lower) & (x <= upper)

    if margin == 0.0:
        return in_bounds.to(x.dtype)

    distance = torch.where(x < lower, lower - x, x - upper) / margin

    if sigmoid == "gaussian":
        scale = math.sqrt(-2 * math.log(value_at_margin))
        decayed = torch.exp(-0.5 * (distance * scale) ** 2)
    elif sigmoid == "linear":
        scaled = distance * (1 - value_at_margin)
        decayed = torch.where(scaled.abs() < 1, 1 - scaled, torch.zeros_like(scaled))
    elif sigmoid == "quadratic":
        scaled = distance * math.sqrt(1 - value_at_margin)
        decayed = torch.where(scaled.abs() < 1, 1 - scaled**2, torch.zeros_like(scaled))
    else:
        raise ValueError(f"Unknown sigmoid '{sigmoid}'.")

    return torch.where(in_bounds, torch.ones_like(x), decayed)


def joint_angles(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    The pose of every hinge, excluding the free root.

    Matches `dm_control`'s `qpos[7:]`, which drops the seven degrees of
    freedom belonging to the floating base.

    Parameters
    ----------
    env : ManagerBasedRlEnv
        The environment being observed
    asset_cfg : SceneEntityCfg
        Selects the humanoid

    Returns
    -------
    angles : torch.Tensor
        The joint angles `(num_envs, 21)`
    """
    asset: Entity = env.scene[asset_cfg.name]

    return asset.data.joint_pos


def head_height(env: ManagerBasedRlEnv, head_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    The world height of the head.

    Parameters
    ----------
    env : ManagerBasedRlEnv
        The environment being observed
    head_cfg : SceneEntityCfg
        Selects the head body

    Returns
    -------
    height : torch.Tensor
        The height above the floor `(num_envs, 1)`
    """
    asset: Entity = env.scene[head_cfg.name]
    body_id = resolved_ids(head_cfg.body_ids, 1, "body")[0]

    return asset.data.body_link_pos_w[:, body_id, 2:3]


def torso_vertical(env: ManagerBasedRlEnv, torso_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    The world z-components of the torso's own axes.

    The final component is the uprightness used by the reward, being
    `1` when the torso's z-axis points straight up and `-1` when it is
    inverted.

    Parameters
    ----------
    env : ManagerBasedRlEnv
        The environment being observed
    torso_cfg : SceneEntityCfg
        Selects the torso body

    Returns
    -------
    vertical : torch.Tensor
        The z-components of the torso's x, y, and z axes
        `(num_envs, 3)`
    """
    asset: Entity = env.scene[torso_cfg.name]
    body_id = resolved_ids(torso_cfg.body_ids, 1, "body")[0]
    w, x, y, z = asset.data.body_link_quat_w[:, body_id].unbind(-1)

    return torch.stack(
        [
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )


def extremities(
    env: ManagerBasedRlEnv,
    torso_cfg: SceneEntityCfg,
    extremity_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    The end effector positions in the torso's frame.

    Expressing the hands and feet egocentrically keeps the observation
    invariant to where the humanoid stands and which way it faces.

    Parameters
    ----------
    env : ManagerBasedRlEnv
        The environment being observed
    torso_cfg : SceneEntityCfg
        Selects the torso body
    extremity_cfg : SceneEntityCfg
        Selects the four end effectors, in order

    Returns
    -------
    positions : torch.Tensor
        The flattened egocentric positions `(num_envs, 12)`
    """
    asset: Entity = env.scene[torso_cfg.name]
    torso_id = resolved_ids(torso_cfg.body_ids, 1, "body")[0]
    limb_ids = resolved_ids(extremity_cfg.body_ids, 4, "body")

    torso_pos = asset.data.body_link_pos_w[:, torso_id]
    rotation = _rotation_matrix(asset.data.body_link_quat_w[:, torso_id])
    offsets = asset.data.body_link_pos_w[:, limb_ids] - torso_pos.unsqueeze(1)

    return torch.matmul(offsets, rotation).flatten(start_dim=1)


def com_velocity(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """
    The linear velocity of the humanoid's centre of mass.

    Parameters
    ----------
    env : ManagerBasedRlEnv
        The environment being observed
    sensor_name : str
        The subtree linear velocity sensor declared in the model

    Returns
    -------
    velocity : torch.Tensor
        The centre of mass velocity `(num_envs, 3)`
    """
    return builtin_sensor(env, sensor_name)


def velocity(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    The full generalised velocity, as `dm_control` reports it.

    Concatenates the floating base's linear and angular velocity with
    every hinge velocity, matching `qvel`. The base linear velocity is
    in the world frame and its angular velocity in the body frame,
    which is MuJoCo's free joint convention.

    Parameters
    ----------
    env : ManagerBasedRlEnv
        The environment being observed
    asset_cfg : SceneEntityCfg
        Selects the humanoid

    Returns
    -------
    velocity : torch.Tensor
        The generalised velocity `(num_envs, 27)`
    """
    asset: Entity = env.scene[asset_cfg.name]

    return torch.cat(
        [
            asset.data.root_link_lin_vel_w,
            asset.data.root_link_ang_vel_b,
            asset.data.joint_vel,
        ],
        dim=-1,
    )


def humanoid_reward(
    env: ManagerBasedRlEnv,
    torso_cfg: SceneEntityCfg,
    head_cfg: SceneEntityCfg,
    com_sensor: str,
    move_speed: float,
    stand_height: float,
    upright_bound: float,
    upright_margin: float,
    control_margin: float,
    dont_move_margin: float,
) -> torch.Tensor:
    """
    The `dm_control` Humanoid reward.

    Multiplies three independent factors, so a policy has to satisfy
    all of them at once. Standing rewards head height and an upright
    torso, effort rewards small actions, and movement rewards either
    stillness or a target horizontal speed depending on `move_speed`.

    Parameters
    ----------
    env : ManagerBasedRlEnv
        The environment being rewarded
    torso_cfg : SceneEntityCfg
        Selects the torso body
    head_cfg : SceneEntityCfg
        Selects the head body
    com_sensor : str
        The centre of mass velocity sensor
    move_speed : float
        The target horizontal speed. A speed of `0` rewards standing
        still instead
    stand_height : float
        The head height at which standing scores `1`
    upright_bound : float
        The uprightness above which posture scores `1`
    upright_margin : float
        The uprightness falloff width
    control_margin : float
        The actuation magnitude at which the effort penalty bottoms out
    dont_move_margin : float
        The horizontal speed at which stillness stops being rewarded

    Returns
    -------
    reward : torch.Tensor
        The reward `(num_envs,)`, in `[0, 1]`
    """
    standing = _tolerance(
        head_height(env, head_cfg).squeeze(-1),
        lower=stand_height,
        upper=math.inf,
        margin=stand_height / 4,
        sigmoid="gaussian",
        value_at_margin=0.1,
    )
    upright = _tolerance(
        torso_vertical(env, torso_cfg)[:, 2],
        lower=upright_bound,
        upper=math.inf,
        margin=upright_margin,
        sigmoid="linear",
        value_at_margin=0.0,
    )
    stand_reward = standing * upright

    effort = _tolerance(
        env.action_manager.action,
        lower=0.0,
        upper=0.0,
        margin=control_margin,
        sigmoid="quadratic",
        value_at_margin=0.0,
    ).mean(dim=-1)
    small_control = (4 + effort) / 5

    horizontal = com_velocity(env, com_sensor)[:, :2]

    if move_speed == 0.0:
        dont_move = _tolerance(
            horizontal,
            lower=0.0,
            upper=0.0,
            margin=dont_move_margin,
            sigmoid="gaussian",
            value_at_margin=0.1,
        ).mean(dim=-1)

        return small_control * stand_reward * dont_move

    move = _tolerance(
        torch.linalg.norm(horizontal, dim=-1),
        lower=move_speed,
        upper=math.inf,
        margin=move_speed,
        sigmoid="linear",
        value_at_margin=0.0,
    )

    return small_control * stand_reward * (5 * move + 1) / 6


class HumanoidStand(MjlabEnvSpec):
    """
    The `dm_control` Humanoid stand task.

    A 21 degree of freedom humanoid that begins each episode collapsed
    in a randomised pose and must get its head to `1.4` metres with an
    upright torso, then hold still.

    The reward multiplies three factors, so partial success in one
    cannot compensate for failure in another. Standing combines head
    height with torso uprightness, effort rewards small actions, and
    the third factor rewards keeping the centre of mass horizontally
    stationary. Episodes run a fixed twenty five seconds with no
    failure state, so a fallen humanoid simply scores near zero for the
    remainder.

    Observations are the joint angles, head height, egocentric hand and
    foot positions, the torso's vertical orientation, the centre of
    mass velocity, and the full generalised velocity.

    Parameters
    ----------
    cfg : HumanoidCfg (optional)
        The task's tunable quantities. Default is `HumanoidCfg()`
    parts : HumanoidParts (optional)
        The parts of the model that terms select against. Default is
        `HumanoidParts()`

    Attributes
    ----------
    name : str
        The environment ID
    move_speed : float
        The target horizontal speed, or `0` to reward standing still
    asset : SceneEntityCfg
        Selects the humanoid as a whole
    torso : SceneEntityCfg
        Selects the torso body
    head : SceneEntityCfg
        Selects the head body
    extremities : SceneEntityCfg
        Selects the four end effectors, in order
    """

    name = "dm_control/humanoid-stand-v0"
    move_speed: ClassVar[float] = 0.0

    def __init__(
        self,
        cfg: HumanoidCfg = HumanoidCfg(),
        parts: HumanoidParts = HumanoidParts(),
    ) -> None:
        self.cfg = cfg
        self.parts = parts

        self.com_sensor = f"{parts.name}/{parts.com_sensor}"

        self.asset = SceneEntityCfg(parts.name)
        self.torso = SceneEntityCfg(parts.name, body_names=(parts.torso_body,))
        self.head = SceneEntityCfg(parts.name, body_names=(parts.head_body,))
        self.extremities = SceneEntityCfg(
            parts.name,
            body_names=parts.extremity_bodies,
            preserve_order=True,
        )

    def config(self) -> ManagerBasedRlEnvCfg:
        return ManagerBasedRlEnvCfg(
            scene=self._scene(),
            observations=self._observations(),
            actions=self._actions(),
            events=self._events(),
            rewards=self._rewards(),
            terminations=self._terminations(),
            viewer=self._viewer(),
            sim=self._sim(),
            decimation=5,
            episode_length_s=self.cfg.time_limit_s,
            scale_rewards_by_dt=False,
        )

    def _scene(self) -> SceneCfg:
        """
        Builds the scene.

        Holds the humanoid alone. The model brings its own floor, so no
        terrain is added.

        Returns
        -------
        cfg : SceneCfg
            The scene the task takes place in
        """
        return SceneCfg(
            entities={
                self.parts.name: EntityCfg(
                    spec_fn=partial(get_spec, filepath=_HUMANOID_XML),
                    articulation=EntityArticulationInfoCfg(
                        actuators=(XmlActuatorCfg(target_names_expr=(".*",)),),
                    ),
                    init_state=EntityCfg.InitialStateCfg(
                        pos=(0.0, 0.0, 1.5),
                        joint_vel={".*": 0.0},
                    ),
                )
            },
            num_envs=1,
            env_spacing=4.0,
        )

    def _observations(self) -> dict[str, ObservationGroupCfg]:
        """
        Builds the observation groups.

        The six terms reproduce `dm_control`'s egocentric feature set
        and concatenate to 67 values. The critic sees exactly what the
        actor sees, as the task holds nothing back from the policy.

        Returns
        -------
        groups : dict[str, ObservationGroupCfg]
            The `actor` and `critic` groups
        """
        terms = {
            "joint_angles": ObservationTermCfg(
                func=joint_angles,
                params={"asset_cfg": self.asset},
            ),
            "head_height": ObservationTermCfg(
                func=head_height,
                params={"head_cfg": self.head},
            ),
            "extremities": ObservationTermCfg(
                func=extremities,
                params={"torso_cfg": self.torso, "extremity_cfg": self.extremities},
            ),
            "torso_vertical": ObservationTermCfg(
                func=torso_vertical,
                params={"torso_cfg": self.torso},
            ),
            "com_velocity": ObservationTermCfg(
                func=com_velocity,
                params={"sensor_name": self.com_sensor},
            ),
            "velocity": ObservationTermCfg(
                func=velocity,
                params={"asset_cfg": self.asset},
            ),
        }

        return {
            "actor": ObservationGroupCfg(terms, enable_corruption=True),
            "critic": ObservationGroupCfg({**terms}),
        }

    def _actions(self) -> dict[str, ActionTermCfg]:
        """
        Builds the action terms.

        Returns
        -------
        actions : dict[str, ActionTermCfg]
            A torque for each of the model's twenty one actuators
        """
        return {
            "effort": JointEffortActionCfg(
                entity_name=self.parts.name,
                actuator_names=(".*",),
                scale=1.0,
            ),
        }

    def _events(self) -> dict[str, EventTermCfg]:
        """
        Builds the event terms.

        Returns
        -------
        events : dict[str, EventTermCfg]
            A reset scattering every hinge across its range, leaving
            the humanoid collapsed in a randomised pose
        """
        return {
            "reset_joints": EventTermCfg(
                func=reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": self.cfg.reset_position_range,
                    "velocity_range": self.cfg.reset_velocity_range,
                    "asset_cfg": self.asset,
                },
            ),
        }

    def _rewards(self) -> dict[str, RewardTermCfg]:
        """
        Builds the reward terms.

        Returns
        -------
        rewards : dict[str, RewardTermCfg]
            A single term combining posture, effort, and movement
        """
        return {
            "humanoid": RewardTermCfg(
                func=humanoid_reward,
                weight=1.0,
                params={
                    "torso_cfg": self.torso,
                    "head_cfg": self.head,
                    "com_sensor": self.com_sensor,
                    "move_speed": self.move_speed,
                    "stand_height": self.cfg.stand_height,
                    "upright_bound": self.cfg.upright_bound,
                    "upright_margin": self.cfg.upright_margin,
                    "control_margin": self.cfg.control_margin,
                    "dont_move_margin": self.cfg.dont_move_margin,
                },
            ),
        }

    def _terminations(self) -> dict[str, TerminationTermCfg]:
        """
        Builds the termination terms.

        Returns
        -------
        terminations : dict[str, TerminationTermCfg]
            The time limit alone. The task has no failure state, so
            every episode ends as a truncation
        """
        return {
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
        }

    def _sim(self) -> SimulationCfg:
        """
        Builds the simulation config.

        Mirrors the solver settings declared in the model's `option`
        block, which `mjlab` does not read from the spec itself.
        Contacts stay enabled, as the humanoid has to push against the
        floor.

        Returns
        -------
        cfg : SimulationCfg
            The physics timestep, solver settings, and disabled flags
        """
        return SimulationCfg(
            mujoco=MujocoCfg(
                timestep=0.005,
                solver="newton",
                iterations=4,
                ls_iterations=8,
                disableflags=("eulerdamp",),
            ),
        )

    def _viewer(self) -> ViewerConfig:
        """
        Builds the viewer config.

        Returns
        -------
        cfg : ViewerConfig
            A camera tracking the torso from slightly above
        """
        return ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name=self.parts.name,
            body_name=self.parts.torso_body,
            distance=4.0,
            elevation=-15.0,
            azimuth=120.0,
        )


class HumanoidWalk(HumanoidStand):
    """
    The `dm_control` Humanoid walk task.

    Matches `HumanoidStand` except that the third reward factor asks
    for a horizontal centre of mass speed of `1` metre per second
    rather than stillness. The humanoid must therefore stand up first
    and only then start moving, since the standing and movement factors
    multiply.

    Parameters
    ----------
    cfg : HumanoidCfg (optional)
        The task's tunable quantities. Default is `HumanoidCfg()`
    parts : HumanoidParts (optional)
        The parts of the model that terms select against. Default is
        `HumanoidParts()`

    Attributes
    ----------
    name : str
        The environment ID
    move_speed : float
        Always `1.0`
    """

    name = "dm_control/humanoid-walk-v0"
    move_speed = 1.0


class HumanoidRun(HumanoidStand):
    """
    The `dm_control` Humanoid run task.

    Matches `HumanoidWalk` but asks for `10` metres per second, a speed
    no policy reaches in practice. The movement factor therefore acts
    as a near linear incentive to go faster, and returns stay well
    below the theoretical maximum.

    Parameters
    ----------
    cfg : HumanoidCfg (optional)
        The task's tunable quantities. Default is `HumanoidCfg()`
    parts : HumanoidParts (optional)
        The parts of the model that terms select against. Default is
        `HumanoidParts()`

    Attributes
    ----------
    name : str
        The environment ID
    move_speed : float
        Always `10.0`
    """

    name = "dm_control/humanoid-run-v0"
    move_speed = 10.0
