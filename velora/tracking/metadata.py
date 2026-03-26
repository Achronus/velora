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

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Self


@dataclass(frozen=True)
class CheckpointMetadata:
    """
    Base class for checkpoint metadata.

    Subclass this with the specific fields needed for your trainer,
    then pass instances to `CheckpointManager.save()` for automatic
    JSON serialization alongside Orbax checkpoints.

    Examples
    --------
    >>> @dataclass(frozen=True)
    ... class MyTrainerMetadata(CheckpointMetadata):
    ...     learning_rate: float
    ...     num_envs: int
    ...
    >>> meta = MyTrainerMetadata(learning_rate=0.001, num_envs=57)
    >>> meta.to_dict()
    {'learning_rate': 0.001, 'num_envs': 57}
    """

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize all fields to a JSON-compatible dictionary.

        Returns
        -------
        data : Dict[str, Any]
            Dictionary representation of all fields
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """
        Reconstruct an instance from a dictionary.

        Ignores keys in `data` that don't match any field, so metadata
        files remain forward-compatible when fields are removed.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary produced by `to_dict`

        Returns
        -------
        instance : Self
            Reconstructed metadata instance
        """
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})
