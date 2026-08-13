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


from pathlib import Path

import mujoco


def get_spec(filepath: Path | str) -> mujoco.MjSpec:
    """
    Loads a MuJoCo model specification from an XML file.

    Parameters
    ----------
    filepath : Path | str
        The path to the model's XML file

    Returns
    -------
    spec : mujoco.MjSpec
        The editable specification parsed from the file
    """
    return mujoco.MjSpec.from_file(str(filepath))


def resolved_ids(ids: list[int] | slice, count: int, kind: str) -> list[int]:
    """
    Checks that a `SceneEntityCfg` field resolved to the expected
    elements.

    A cfg that never reached a manager keeps its ids as `slice(None)`,
    which silently selects everything. Indexing with that produces a
    wrongly shaped tensor rather than an error, so terms validate their
    selection before using it.

    Parameters
    ----------
    ids : list[int] | slice
        The resolved ids taken from a `SceneEntityCfg`
    count : int
        The number of elements the term expects
    kind : str
        The element being selected, used in the error message

    Returns
    -------
    ids : list[int]
        The validated ids

    Raises
    ------
    unresolved_cfg : TypeError
        Error when the cfg was never resolved against a scene
    wrong_count : ValueError
        Error when the cfg selects the wrong number of elements
    """
    if isinstance(ids, slice):
        raise TypeError(
            f"The {kind} selection was never resolved against a scene. "
            "Pass a 'SceneEntityCfg' that a manager has resolved."
        )

    if len(ids) != count:
        raise ValueError(
            f"Expected the selection to match {count} {kind}(s), got {len(ids)}."
        )

    return ids
