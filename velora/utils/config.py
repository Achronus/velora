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
from pathlib import Path
from typing import Any


def dump_config(config: Any) -> dict[str, Any]:
    """
    Recursively serialize a `struct.dataclass` to a JSON-compatible dict.

    Walks all dataclass fields, serializing primitive values directly and
    recursing into nested dataclass instances. `Path` values are converted
    to strings.

    Parameters
    ----------
    config : Any
        A `struct.dataclass` (or standard dataclass) instance to serialize

    Returns
    -------
    data : dict[str, Any]
        JSON-compatible dictionary representation
    """
    result = {}
    for f in fields(config):
        value = getattr(config, f.name)

        if isinstance(value, Path):
            result[f.name] = str(value)
        elif hasattr(value, "__dataclass_fields__"):
            result[f.name] = dump_config(value)
        else:
            result[f.name] = value

    return result


def load_config(cls: type, data: dict[str, Any]) -> Any:
    """
    Recursively reconstruct a `struct.dataclass` from a serialized dict.

    Uses the class's field type annotations to determine when to recurse
    into nested dataclass fields. `Path` fields are reconstructed from
    their string representations.

    Parameters
    ----------
    cls : type
        The dataclass class to reconstruct
    data : dict[str, Any]
        Dictionary produced by `dump_config`

    Returns
    -------
    instance : Any
        Reconstructed dataclass instance
    """
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue

        value = data[f.name]
        # Resolve the actual type, unwrapping Optional/Union if needed
        ftype = (
            f.type
            if not isinstance(f.type, str)
            else cls.__annotations__.get(f.name, f.type)
        )
        origin = getattr(ftype, "__origin__", None)
        args = getattr(ftype, "__args__", ())

        # Unwrap Optional[X] -> X
        if origin is type(None):
            kwargs[f.name] = None
            continue

        actual_type = (
            next((a for a in args if a is not type(None)), ftype) if args else ftype
        )

        if isinstance(actual_type, type) and issubclass(actual_type, Path):
            kwargs[f.name] = Path(value)
        elif (
            isinstance(value, dict)
            and isinstance(actual_type, type)
            and hasattr(actual_type, "__dataclass_fields__")
        ):
            kwargs[f.name] = load_config(actual_type, value)
        else:
            kwargs[f.name] = value

    return cls(**kwargs)
