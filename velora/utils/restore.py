import json
from pathlib import Path
from pydoc import locate
from typing import TYPE_CHECKING, Any, Dict, Literal, Type

import torch
from safetensors.torch import load_file, save_file

if TYPE_CHECKING:
    from velora.models.base import RLAgent  # pragma: no cover


TensorDictKeys = Literal["actor", "critic", "actor_target", "critic_target"]
MetadataKeys = Literal["model", "critic_optim", "actor_optim", "buffer"]
ModelMetadataKeys = Literal[
    "state_dim",
    "n_neurons",
    "action_dim",
    "buffer_size",
    "device",
]
OptimDictKeys = Literal["actor_optim", "critic_optim"]


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
    state_dict: Dict[OptimDictKeys, Any] = {
        "actor_optim": {},
        "critic_optim": {},
    }

    for key, tensor in tensor_dict.items():
        optim_name, param_id, param_key = key.split(".")
        param_id = int(param_id)

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
    state_dict: Dict[TensorDictKeys, Any] = {
        "actor": {},
        "critic": {},
        "actor_target": {},
        "critic_target": {},
    }

    for key, tensor in tensor_dict.items():
        model_name, param_name = key.split(".", 1)
        state_dict[model_name][param_name] = tensor

    return state_dict


def save_model(
    agent: "RLAgent",
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
        agent (RLAgent): the agent's state to save
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

    model_tuples = [
        ("actor", agent.actor.state_dict()),
        ("critic", agent.critic.state_dict()),
    ]

    # Add target networks if used
    if agent.actor_target is not None:
        model_tuples.extend(
            [
                ("actor_target", agent.actor_target.state_dict()),
                ("critic_target", agent.critic_target.state_dict()),
            ]
        )

    tensor_dict: Dict[str, torch.Tensor] = {}

    for model_name, state_dict in model_tuples:
        for param_name, tensor in state_dict.items():
            tensor_dict[f"{model_name}.{param_name}"] = tensor.contiguous()

    actor_dict = optim_to_tensor("actor_optim", agent.actor_optim.state_dict())
    critic_dict = optim_to_tensor("critic_optim", agent.critic_optim.state_dict())
    optim_dict = dict(**actor_dict, **critic_dict)

    # Save tensors (weights and biases only)
    save_file(tensor_dict, model_state_path)
    save_file(optim_dict, optim_state_path)

    # Store metadata as JSON
    metadata: Dict[MetadataKeys, Any] = {
        "model": {
            "state_dim": agent.state_dim,
            "n_neurons": agent.n_neurons,
            "action_dim": agent.action_dim,
            "buffer_size": agent.buffer_size,
            "optim": f"torch.optim.{type(agent.actor_optim).__name__}",
            "device": str(agent.device) if agent.device is not None else "cpu",
        },
        "actor_optim": agent.actor_optim.state_dict()["param_groups"],
        "critic_optim": agent.critic_optim.state_dict()["param_groups"],
    }

    if buffer:
        metadata["buffer"] = agent.buffer.metadata()
        save_file(agent.buffer.state_dict(), buffer_state_path)

    if config and not config_path.exists():
        with config_path.open("w") as f:
            f.write(agent.config.model_dump_json(indent=2, exclude_none=True))

    if not metadata_path.exists():
        with metadata_path.open("w") as f:
            f.write(json.dumps(metadata, indent=2))


def load_model(
    agent: Type["RLAgent"], dirpath: str | Path, *, buffer: bool = False
) -> "RLAgent":
    """
    Creates a new agent instance by loading a saved one from the `dirpath`.
    Also, loads the original training buffer if `buffer=True`.

    These files must exist in the `dirpath`:
    - `metadata.json` - contains the model, optimizer and buffer (optional) metadata
    - `model_state.safetensors` - contains the model weights and biases
    - `optim_state.safetensors` - contains the optimizer states (actor and critic)
    - `buffer_state.safetensors` - contains the buffer state (only if `buffer=True`)

    Parameters:
        agent (Type[RLAgent]): the type of agent to load
        dirpath (str | Path): the location to store the model state. Should only
            consist of `folder` names. E.g., `<folder>/<folder>`
        buffer (bool, optional): a flag for storing the buffer state

    Returns:
        agent (RLAgent): a new agent instance with the saved state
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

    device: str = metadata["model"]["device"] or "cpu"
    metadata["model"]["device"] = torch.device(device)
    metadata["model"]["optim"] = locate(metadata["model"]["optim"])

    # Create new model instance
    model = agent(**metadata["model"])

    # Load model parameters from safetensors
    tensor_dict: Dict[str, torch.Tensor] = load_file(model_state_path, device)
    model_state = model_from_tensor(tensor_dict)

    model.actor.load_state_dict(model_state["actor"])
    model.critic.load_state_dict(model_state["critic"])

    if "actor_target" in model_state.keys():
        model.actor_target.load_state_dict(model_state["actor_target"])
        model.critic_target.load_state_dict(model_state["critic_target"])

    # Load optimizer parameters from safetensors
    tensor_dict: Dict[str, torch.Tensor] = load_file(optim_state_path, device)
    optim_state = optim_from_tensor(tensor_dict)

    model.actor_optim.load_state_dict(
        {
            "state": optim_state["actor_optim"],
            "param_groups": metadata["actor_optim"],
        }
    )
    model.critic_optim.load_state_dict(
        {
            "state": optim_state["critic_optim"],
            "param_groups": metadata["critic_optim"],
        }
    )

    # Load buffer
    if buffer:
        model.buffer = model.buffer.load(buffer_state_path, metadata["buffer"])

    print(f"Loaded model:\n  {model}\n  buffer_restored={buffer}")
    return model
