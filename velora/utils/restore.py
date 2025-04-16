import json
from pathlib import Path
from pydoc import locate
from typing import TYPE_CHECKING, Any, Dict, Literal, Type, get_args

if TYPE_CHECKING:
    from velora.models.base import RLModuleAgent  # pragma: no cover

import torch
from safetensors.torch import load_file, save_file

TensorDictKeys = Literal[
    "actor",
    "critic",
    "critic2",
    "critic_target",
    "critic2_target",
]
MetadataKeys = Literal[
    "model",
    "actor_optim",
    "critic_optim",
    "critic2_optim",
    "entropy_optim",
    "buffer",
]
OptimDictKeys = Literal["actor_optim", "critic_optim", "critic2_optim", "entropy_optim"]
ModuleNames = Literal["actor", "critic", "entropy"]


def optim_to_tensor(name: str, state_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Extracts an optimizers `state` from it's `state_dict()` and converts it into
    a `tensor_dict` for storage.

    Parameters:
        name (str): name of the optimizer
        state_dict (Dict[str, Any]): PyTorch optimizer state dictionary

    Returns:
        tensor_dict (Dict[str, torch.Tensor]): the converted state as a tensor dictionary
    """
    tensor_dict: Dict[str, torch.Tensor] = {}

    if "state" in state_dict:
        for param_id, param_state in state_dict["state"].items():
            for k, v in param_state.items():
                tensor_key = f"{name}.{param_id}.{k}"
                tensor_dict[tensor_key] = v.cpu()

    return tensor_dict


def optim_from_tensor(tensor_dict: Dict[str, torch.Tensor]) -> Dict[OptimDictKeys, Any]:
    """
    Converts an optimizer's `tensor_dict` back into a `state_dict()` without the
    metadata.

    Parameters:
        tensor_dict (Dict[str, torch.Tensor]): PyTorch optimizers tensor dictionary

    Returns:
        state_dict (Dict[str, Any]): the converted state as a normal dictionary
    """
    state_dict: Dict[OptimDictKeys, Any] = {}

    for key, tensor in tensor_dict.items():
        optim_name, param_id, param_key = key.split(".")
        param_id = int(param_id)

        if optim_name not in state_dict:
            state_dict[optim_name] = {}

        if param_id not in state_dict[optim_name]:
            state_dict[optim_name][param_id] = {}

        state_dict[optim_name][param_id][param_key] = tensor

    return state_dict


def model_from_tensor(
    tensor_dict: Dict[str, torch.Tensor],
) -> Dict[TensorDictKeys, Any]:
    """
    Converts a model's `tensor_dict` back into a `state_dict()`.

    Parameters:
        tensor_dict (Dict[str, torch.Tensor]): PyTorch model tensor dictionary

    Returns:
        state_dict (Dict[str, Any]): the converted state as a normal dictionary
    """
    state_dict: Dict[TensorDictKeys, Any] = {}

    for key, tensor in tensor_dict.items():
        model_name, param_name = key.split(".", 1)

        if model_name not in state_dict:
            state_dict[model_name] = {}

        state_dict[model_name][param_name] = tensor

    return state_dict


def save_model(
    agent: "RLModuleAgent",
    dirpath: str | Path,
    *,
    buffer: bool = False,
    config: bool = False,
    force: bool = False,
) -> None:
    """
    Saves the current model state into `safetensors` and `json` files.

    !!! warning

        `model_config.json` is stored in the `dirpath.parent`.

    Includes:
    - `model_config.json` - contains the core details of the agent (optional)
    - `metadata.json` - contains the model, optimizer and buffer (optional) metadata
    - `model_state.safetensors` - contains the model weights and biases
    - `optim_state.safetensors` - contains the optimizer states (actor and critic)
    - `buffer_state.safetensors` - contains the buffer state (only if `buffer=True`)

    Parameters:
        agent (RLModuleAgent): the agent's state to save
        dirpath (str | Path): the location to store the model state. Should only
            consist of `folder` names. E.g., `<folder>/<folder>`
        buffer (bool, optional): a flag for storing the buffer state
        config (bool, optional): a flag for storing the model's config
        force (bool, optional): enables file overwriting, ignoring existing state
            files. Useful for continuing model training
    """
    save_path = Path(dirpath)
    save_path.mkdir(parents=True, exist_ok=True)

    config_path = Path(save_path.parent, "model_config").with_suffix(".json")
    metadata_path = Path(save_path, "metadata").with_suffix(".json")
    model_state_path = Path(save_path, "model_state").with_suffix(".safetensors")
    optim_state_path = Path(save_path, "optim_state").with_suffix(".safetensors")
    buffer_state_path = Path(save_path, "buffer_state").with_suffix(".safetensors")

    if not force and model_state_path.exists():
        raise FileExistsError(
            f"A model state already exists in the '{save_path}' directory! Either change the 'dirpath', delete the folders contents, or use 'force=True' to allow overwriting."
        )

    # Handle module state dicts
    agent_state = agent.state_dict()
    tensor_dict: Dict[str, torch.Tensor] = {}

    for model_name, state_dict in agent_state["modules"].items():
        for param_name, tensor in state_dict.items():
            tensor_dict[f"{model_name}.{param_name}"] = tensor.contiguous()

    # Handle optimizers
    optim_dict: Dict[OptimDictKeys, Dict[str, Any]] = {}
    param_dict: Dict[OptimDictKeys, Dict[str, Any]] = {}

    for optim_name, state_dict in agent_state["optimizers"].items():
        optim_dict |= optim_to_tensor(optim_name, state_dict)
        param_dict[optim_name] = state_dict["param_groups"]

    metadata: Dict[MetadataKeys, Any] = {
        "model": agent.metadata,
        **param_dict,
    }

    # Save tensors (weights and biases only)
    save_file(tensor_dict, model_state_path)
    save_file(optim_dict, optim_state_path)

    # Add buffer (if applicable)
    if buffer:
        metadata["buffer"] = agent.buffer.metadata()
        save_file(agent.buffer.state_dict(), buffer_state_path)

    # Write to files
    if config and not config_path.exists():
        with config_path.open("w") as f:
            f.write(agent.config.model_dump_json(indent=2, exclude_none=True))

    if not metadata_path.exists():
        with metadata_path.open("w") as f:
            f.write(json.dumps(metadata, indent=2))


def load_model(
    agent: Type["RLModuleAgent"], dirpath: str | Path, *, buffer: bool = False
) -> Any:
    """
    Creates a new agent instance by loading a saved one from the `dirpath`.
    Also, loads the original training buffer if `buffer=True`.

    These files must exist in the `dirpath`:
    - `metadata.json` - contains the model, optimizer and buffer (optional) metadata
    - `model_state.safetensors` - contains the model weights and biases
    - `optim_state.safetensors` - contains the optimizer states (actor and critic)
    - `buffer_state.safetensors` - contains the buffer state (only if `buffer=True`)

    Parameters:
        agent (Type[RLModuleAgent]): the type of agent to load
        dirpath (str | Path): the location to store the model state. Should only
            consist of `folder` names. E.g., `<folder>/<folder>`
        buffer (bool, optional): a flag for storing the buffer state

    Returns:
        agent (RLModuleAgent): a new agent instance with the saved state
    """
    load_path = Path(dirpath)

    metadata_path = Path(load_path, "metadata").with_suffix(".json")
    model_state_path = Path(load_path, "model_state").with_suffix(".safetensors")
    optim_state_path = Path(load_path, "optim_state").with_suffix(".safetensors")
    buffer_state_path = Path(load_path, "buffer_state").with_suffix(".safetensors")

    if not model_state_path.exists():
        raise FileNotFoundError(f"Model state '{model_state_path}' does not exist!")

    if not optim_state_path.exists():
        raise FileNotFoundError(f"Optimizer state '{optim_state_path}' does not exist!")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata '{metadata_path}' does not exist!")

    if buffer and not buffer_state_path.exists():
        raise FileNotFoundError(
            f"Buffer state '{buffer_state_path}' does not exist! Try with 'buffer=False'."
        )

    # Load metadata first
    with metadata_path.open("r") as f:
        metadata: Dict[MetadataKeys, Any] = json.load(f)

    device: str = metadata["model"]["device"]
    metadata["model"]["device"] = torch.device(device)
    metadata["model"]["optim"] = locate(metadata["model"]["optim"])

    # Create new model instance
    model = agent(**metadata["model"])

    # Load module and optimizer parameters from safetensors
    tensor_dict: Dict[str, torch.Tensor] = load_file(model_state_path, device)
    agent_state = model_from_tensor(tensor_dict)

    tensor_dict: Dict[str, torch.Tensor] = load_file(optim_state_path, device)
    optim_state = optim_from_tensor(tensor_dict)

    metadata.pop("model")
    for key in metadata.keys():
        state = optim_state[key] if key in optim_state else {}

        agent_state[key] = {
            "state": state,
            "param_groups": metadata[key],
        }

    # Restore module states
    for name in get_args(ModuleNames):
        module = getattr(model, name)
        module.load_state_dict(agent_state)

    # Load buffer
    if buffer:
        model.buffer = model.buffer.load(buffer_state_path, metadata["buffer"])

    print(f"Loaded model:\n  {model}\n  buffer_restored={buffer}")
    return model
