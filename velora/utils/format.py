# Copyright 2025 Achronus
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


from datetime import datetime
from pathlib import Path


def number_to_short(value: int) -> str:
    """
    Converts a number into a human-readable format like `1M` or `1.25K`.

    Parameters
    ----------
    value : int
        The number to convert

    Returns
    -------
    num_text : str
        The shortened version as a string
    """
    is_negative = value < 0
    abs_value = abs(value)

    suffixes = [(1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")]

    for threshold, suffix in suffixes:
        if abs_value >= threshold:
            short_value = round(abs_value / threshold, 2)
            result = f"{short_value:g}{suffix}"
            return f"-{result}" if is_negative else result

    return str(value)


def create_directory(
    root_dir: str | Path,
    folder_name: str,
    timestamp: bool = True,
) -> Path:
    """
    Creates a directory path with the sub-folder name containing optional timestamps.

    Parameters
    ----------
    root_dir : str | Path
        Root directory path.
    folder_name : str
        Unique folder name stored inside `root_dir`. Merged with `timestamp`
    timestamp : bool (optional)
        Whether to append timestamps to experiment directory.
        Uses timestamp format: `ddmmyy_hhmmss`. Default is `True`

    Returns
    -------
    dirpath : Path
        Directory path in format: `[root_dir]/[folder_name]_[ddmmyy]_[hhmmss]/`
        or `[root_dir]/[folder_name]/`
    """
    ts = datetime.now().strftime("%d%m%y_%H%M%S")
    name = f"{folder_name}_{ts}" if timestamp else folder_name
    return Path(root_dir, name).resolve()
