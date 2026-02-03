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

from dataclasses import fields
from typing import TypeVar

T = TypeVar("T")


def get_fields_by_index(dc: T, idx: int) -> T:
    """
    Extracts dataclass field values at a given index.

    Parameters
    ----------
    dc : T
        A dataclass instance
    idx : int
        Index to select along the first axis of each field

    Returns
    -------
    indexed : T
        New dataclass instance with indexed fields
    """
    return dc.__class__(
        **{
            f.name: getattr(dc, f.name)[idx]
            for f in fields(dc)  # type: ignore
        }
    )
