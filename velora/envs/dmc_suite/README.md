# dm_control Suite Models

MJCF models used by velora's MuJoCo Warp environments, vendored so that
the Warp backend does not depend on the `mujoco_playground` package
(and, through it, on JAX).

## Provenance

These files are copied verbatim, without modification, from
[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)
(`playground` 0.2.0), at `mujoco_playground/_src/dm_control_suite/xmls`.

MuJoCo Playground in turn derives them from
[dm_control](https://github.com/google-deepmind/dm_control), whose Control
Suite models are the original source. The changes Playground made relative
to dm_control are documented under
[Upstream Modifications](#upstream-modifications) below.

## License

Copyright 2025 DeepMind Technologies Limited.

The MuJoCo Playground and dm_control projects are both licensed under the
Apache License, Version 2.0. You may obtain a copy of the License at:

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.

velora is likewise Apache-2.0 licensed. These files retain their original
copyright and remain subject to the terms above.

## Upstream Modifications

The remainder of this document is reproduced from MuJoCo Playground's own
`README.md` for the same directory, and describes how these models differ
from the original dm_control versions.

Several changes are made to DM Control envs for them to be performant on XPU, although changes are kept to a minimum. For all envs, we disable `eulerdamp` and reduce solver `iterations` and `ls_iterations`. For a few envs, we increase the simulator timestep, and set `max_contact_points` and `max_geom_pairs` for contact culling. The full set of changes are listed below:

* acrobot
  * `iterations`="2", `ls_iterations`="4`
* ball_in_cup
  * `iterations`="1", `ls_iterations`="4"
* cartpole
  * `iterations`="1", `ls_iterations`="4"
* cheetah
  * `iterations`="4", `ls_iterations`="8"
  * `max_contact_points`=6, `max_geom_pairs`=4
* finger
  * `iterations`="2", `ls_iterations`="8"
  * `max_contact_points`=4, `max_geom_pairs`=2
  * change cylinder contype/conaffinity to 0
* fish
  * `iterations`="2", `ls_iterations`="6"
  * contacts disabled
* hopper
  * `iterations`="4", `ls_iterations`="8"
  * `max_contact_points`=6, `max_geom_pairs`=2
* humanoid
  * Removed touch sensors
  * `timestep`="0.005" compared to 0.0025 in DM Control
  * `max_contact_points`=8, `max_geom_pairs`=8
* manipulator
  * `timestep`="0.005" from "0.002" in DM Control
  * `max_contact_points`=8, `max_geom_pairs`=8
* pendulum
  * `timestep`="0.01" compared to "0.02" in DM Control
  * `iterations`="4", `ls_iterations`="8"
* point_mass
  * `iterations`="1", `ls_iterations`="4"
* reacher
  * `timestep`="0.005" from "0.02" in DM Control
  * `iterations`="1", `ls_iterations`="6"
* swimmer
  * `timestep`="0.003" from "0.002" in DM Control
  * `iterations`="4", `ls_iterations`="8"
  * geom `contype`/`conaffinity` set to zero, since contacts are disabled
* walker
  * `timestep`="0.005", from "0.0025" in DM Control
  * `iterations`="2", `ls_iterations`="5"
  * `max_contact_points`=4, `max_geom_pairs`=4
