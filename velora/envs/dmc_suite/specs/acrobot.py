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
from typing import ClassVar

import torch
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import joint_vel_rel, reset_joints_by_offset, time_out
from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from velora.envs.dmc_suite.paths import XMLS_DIR
from velora.envs.mjlab import MjlabEnvSpec, get_spec, resolved_ids

_ACROBOT_XML: Path = XMLS_DIR / "acrobot.xml"


@dataclass(frozen=True)
class AcrobotCfg:
    """
    Tunable quantities for Acrobot tasks.

    Parameters
    ----------
    target_radius : float (optional)
        The radius of the target sphere. Default is `0.2`
    time_limit_s : float (optional)
        The episode duration in seconds. Default is `10.0`
    gaussian_scale : float (optional)
        The width of the reward's Gaussian falloff, chosen so the
        shaped value reaches `0.1` at the margin. Default is
        `sqrt(-2 * log(0.1))`
    """

    target_radius: float = 0.2
    time_limit_s: float = 10.0
    gaussian_scale: float = math.sqrt(-2 * math.log(0.1))


@dataclass(frozen=True)
class AcrobotParts:
    """
    Describes the parts of the Acrobot model that terms select against.

    Holds names declared in the model's MJCF, which the specification
    turns into `SceneEntityCfg` selections. A single name is one
    element, while a tuple is an ordered group.

    Parameters
    ----------
    name : str (optional)
        The key the model is registered under in the scene. Default is
        `acrobot`
    arm_bodies : tuple[str, ...] (optional)
        The two arm links, ordered from the shoulder outwards. Default
        is `("upper_arm", "lower_arm")`
    arm_joints : tuple[str, ...] (optional)
        The two hinges, ordered from the shoulder outwards. Default is
        `("shoulder", "elbow")`
    tip_site : str (optional)
        The site at the end of the lower arm. Default is `tip`
    target_site : str (optional)
        The site the tip must reach. Default is `target`
    """

    name: str = "acrobot"
    arm_bodies: tuple[str, ...] = ("upper_arm", "lower_arm")
    arm_joints: tuple[str, ...] = ("shoulder", "elbow")
    tip_site: str = "tip"
    target_site: str = "target"


def link_orientations(env: ManagerBasedRlEnv, arm_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    The orientation of both arm links, as `dm_control` reports them.

    Each link contributes the horizontal and vertical components of its
    own z-axis expressed in the world frame, which is the third column
    of its rotation matrix. This avoids the wrap-around discontinuity a
    raw hinge angle would introduce.

    Parameters
    ----------
    env : ManagerBasedRlEnv
        The environment being observed
    arm_cfg : SceneEntityCfg
        Selects the upper and lower arm bodies

    Returns
    -------
    orientations : torch.Tensor
        The horizontal then vertical components `(num_envs, 4)`
    """
    asset: Entity = env.scene[arm_cfg.name]
    body_ids = resolved_ids(arm_cfg.body_ids, 2, "body")
    quat = asset.data.body_link_quat_w[:, body_ids]
    w, x, y, z = quat.unbind(-1)

    horizontal = 2.0 * (x * z + w * y)
    vertical = 1.0 - 2.0 * (x * x + y * y)

    return torch.cat([horizontal, vertical], dim=-1)


def tip_to_target(
    env: ManagerBasedRlEnv,
    tip_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    The distance from the tip of the lower arm to the target.

    Parameters
    ----------
    env : ManagerBasedRlEnv
        The environment being measured
    tip_cfg : SceneEntityCfg
        Selects the `tip` site
    target_cfg : SceneEntityCfg
        Selects the `target` site

    Returns
    -------
    distance : torch.Tensor
        The Euclidean distance `(num_envs,)`
    """
    asset: Entity = env.scene[tip_cfg.name]
    tip = asset.data.site_pos_w[:, resolved_ids(tip_cfg.site_ids, 1, "site")[0]]
    target = asset.data.site_pos_w[:, resolved_ids(target_cfg.site_ids, 1, "site")[0]]

    return torch.linalg.norm(target - tip, dim=-1)


def _gaussian_tolerance(
    x: torch.Tensor,
    margin: float,
    scale: float,
) -> torch.Tensor:
    """
    Gaussian sigmoid tolerance, `1` at `x=0` falling to `0.1` at
    `x=margin`.

    Parameters
    ----------
    x : torch.Tensor
        The error to shape
    margin : float
        The error at which the result reaches `0.1`
    scale : float
        The falloff width that places `0.1` at the margin

    Returns
    -------
    value : torch.Tensor
        The shaped error, in `(0, 1]`
    """
    scaled = x / margin * scale

    return torch.exp(-0.5 * scaled**2)


def acrobot_reward(
    env: ManagerBasedRlEnv,
    tip_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    target_radius: float,
    gaussian_scale: float,
    sparse: bool = False,
) -> torch.Tensor:
    """
    The `dm_control` Acrobot reward.

    Scores `1` whenever the tip sits inside the target sphere. Outside
    it, the sparse variant scores `0` while the dense variant decays
    smoothly with the distance beyond the target's radius.

    Parameters
    ----------
    env : ManagerBasedRlEnv
        The environment being rewarded
    tip_cfg : SceneEntityCfg
        Selects the `tip` site
    target_cfg : SceneEntityCfg
        Selects the `target` site
    target_radius : float
        The radius inside which the reward is `1`
    gaussian_scale : float
        The falloff width used by the shaped reward
    sparse : bool (optional)
        Whether to return an indicator instead of a shaped reward.
        Default is `False`

    Returns
    -------
    reward : torch.Tensor
        The reward `(num_envs,)`, in `[0, 1]`
    """
    distance = tip_to_target(env, tip_cfg, target_cfg)
    inside = distance <= target_radius

    if sparse:
        return inside.float()

    excess = (distance - target_radius).clamp(min=0.0)

    return torch.where(
        inside,
        torch.ones_like(distance),
        _gaussian_tolerance(excess, margin=1.0, scale=gaussian_scale),
    )


class AcrobotSwingUp(MjlabEnvSpec):
    """
    The `dm_control` Acrobot swingup task.

    An underactuated double pendulum driven by a single torque at the
    elbow. The shoulder is passive, so a policy cannot lift the arm
    directly and must instead pump energy into the system across
    several swings before the tip can reach the target suspended four
    metres above the origin.

    Both joints start at a uniformly random angle in `[-pi, pi)` and
    the episode runs for a fixed ten seconds, ending only on the time
    limit. Observations are the horizontal and vertical components of
    each link's orientation, taken from the links themselves rather
    than their joint angles to avoid a wrap-around discontinuity,
    followed by both joint velocities.

    The reward decays smoothly with the distance from the tip to the
    target, reaching `1` once the tip lies inside it.

    Parameters
    ----------
    cfg : AcrobotCfg (optional)
        The task's tunable quantities. Default is `AcrobotCfg()`
    parts : AcrobotParts (optional)
        The model description. Default is `AcrobotParts()`

    Attributes
    ----------
    name : str
        The environment ID
    sparse : bool
        Whether the reward is an indicator rather than a shaped value
    arm : SceneEntityCfg
        Selects both arm links
    joints : SceneEntityCfg
        Selects both hinges
    tip : SceneEntityCfg
        Selects the site at the end of the lower arm
    target : SceneEntityCfg
        Selects the site the tip must reach
    """

    name = "dm_control/acrobot-swingup-v0"
    sparse: ClassVar[bool] = False

    def __init__(
        self,
        cfg: AcrobotCfg = AcrobotCfg(),
        parts: AcrobotParts = AcrobotParts(),
    ) -> None:
        self.cfg = cfg
        self.parts = parts

        self.arm = SceneEntityCfg(
            self.parts.name,
            body_names=self.parts.arm_bodies,
        )
        self.joints = SceneEntityCfg(
            self.parts.name,
            joint_names=self.parts.arm_joints,
        )
        self.tip = SceneEntityCfg(
            self.parts.name,
            site_names=(self.parts.tip_site,),
        )
        self.target = SceneEntityCfg(
            self.parts.name,
            site_names=(self.parts.target_site,),
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
            decimation=1,
            episode_length_s=self.cfg.time_limit_s,
            scale_rewards_by_dt=False,
        )

    def _scene(self) -> SceneCfg:
        """
        Builds the scene.

        Holds the Acrobot alone, at rest with both hinges upright. The
        model brings its own floor, so no terrain is added.

        Returns
        -------
        cfg : SceneCfg
            The scene the task takes place in
        """
        return SceneCfg(
            entities={
                self.parts.name: EntityCfg(
                    spec_fn=partial(get_spec, filepath=_ACROBOT_XML),
                    articulation=EntityArticulationInfoCfg(
                        actuators=(XmlActuatorCfg(target_names_expr=("elbow",)),),
                    ),
                    init_state=EntityCfg.InitialStateCfg(
                        pos=(0.0, 0.0, 0.0),
                        joint_pos={joint: 0.0 for joint in self.parts.arm_joints},
                        joint_vel={".*": 0.0},
                    ),
                )
            },
            num_envs=1,
            env_spacing=6.0,
        )

    def _observations(self) -> dict[str, ObservationGroupCfg]:
        """
        Builds the observation groups.

        The critic sees exactly what the actor sees, as the task holds
        nothing back from the policy.

        Returns
        -------
        groups : dict[str, ObservationGroupCfg]
            The `actor` and `critic` groups
        """
        terms = {
            "orientations": ObservationTermCfg(
                func=link_orientations,
                params={"arm_cfg": self.arm},
            ),
            "velocity": ObservationTermCfg(
                func=joint_vel_rel,
                params={"asset_cfg": self.joints},
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
            A single torque applied to the elbow, the only actuated
            joint in the model
        """
        return {
            "effort": JointEffortActionCfg(
                entity_name=self.parts.name,
                actuator_names=("elbow",),
                scale=1.0,
            ),
        }

    def _events(self) -> dict[str, EventTermCfg]:
        """
        Builds the event terms.

        Returns
        -------
        events : dict[str, EventTermCfg]
            A reset placing both joints at a uniformly random angle in
            `[-pi, pi)` and at rest
        """
        return {
            "reset_joints": EventTermCfg(
                func=reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": (-math.pi, math.pi),
                    "velocity_range": (0.0, 0.0),
                    "asset_cfg": self.joints,
                },
            ),
        }

    def _rewards(self) -> dict[str, RewardTermCfg]:
        """
        Builds the reward terms.

        Returns
        -------
        rewards : dict[str, RewardTermCfg]
            A single term scoring the tip's proximity to the target,
            shaped or sparse according to `sparse`
        """
        return {
            "tip_to_target": RewardTermCfg(
                func=acrobot_reward,
                weight=1.0,
                params={
                    "tip_cfg": self.tip,
                    "target_cfg": self.target,
                    "target_radius": self.cfg.target_radius,
                    "gaussian_scale": self.cfg.gaussian_scale,
                    "sparse": self.sparse,
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

        Returns
        -------
        cfg : SimulationCfg
            The physics timestep, solver iterations, and disabled flags
        """
        return SimulationCfg(
            mujoco=MujocoCfg(
                timestep=0.01,
                iterations=2,
                ls_iterations=4,
                disableflags=("constraint", "eulerdamp"),
            ),
        )

    def _viewer(self) -> ViewerConfig:
        """
        Builds the viewer config.

        Returns
        -------
        cfg : ViewerConfig
            A camera tracking the upper arm from far enough back to
            keep the target in frame
        """
        return ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name=self.parts.name,
            body_name=self.parts.arm_bodies[0],
            distance=8.0,
            elevation=-15.0,
            azimuth=0.0,
        )


class AcrobotSwingUpSparse(AcrobotSwingUp):
    """
    The `dm_control` Acrobot sparse swingup task.

    Matches `AcrobotSwingUp` in every respect but the reward, which
    becomes an indicator: `1` while the tip sits inside the target
    sphere and `0` everywhere else.

    Dropping the shaping turns this into a hard exploration problem.
    Nothing rewards the early swings that build the energy needed to
    reach the target, so a policy receives no signal at all until it
    stumbles into success.

    Parameters
    ----------
    cfg : AcrobotCfg (optional)
        The task's tunable quantities. Default is `AcrobotCfg()`
    parts : AcrobotParts (optional)
        The model description. Default is `AcrobotParts()`

    Attributes
    ----------
    name : str
        The environment ID
    sparse : bool
        Always `True`
    arm : SceneEntityCfg
        Selects both arm links
    joints : SceneEntityCfg
        Selects both hinges
    tip : SceneEntityCfg
        Selects the site at the end of the lower arm
    target : SceneEntityCfg
        Selects the site the tip must reach
    """

    name = "dm_control/acrobot-swingup_sparse-v0"
    sparse = True
