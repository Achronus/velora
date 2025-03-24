from typing import TYPE_CHECKING, List

from velora.utils.format import number_to_short

if TYPE_CHECKING:
    from velora.callbacks import TrainCallback  # pragma: no cover
    from velora.models.base import RLAgent  # pragma: no cover

NAME_STR = """
__     __   _
\\ \\   / /__| | ___  _ __ __ _
 \\ \\ / / _ \\ |/ _ \\| '__/ _` |
  \\ V /  __/ | (_) | | | (_| |
   \\_/ \\___|_|\\___/|_|  \\__,_|
"""


def training_info(
    agent: "RLAgent",
    env_id: str,
    n_episodes: int,
    batch_size: int,
    window_size: int,
    callbacks: List["TrainCallback"],
    device: str,
) -> None:
    """
    Display's starting information to the console for a training run.

    Parameters:
        agent (RLAgent): the agent being trained
        env_id (str): the environment ID
        n_episodes (int): maximum number of training episodes
        batch_size (int): sampling batch size
        window_size (int): moving average window size
        callbacks (List[TrainCallback]): applied training callbacks
        device (str): the device to perform computations on
    """
    output = NAME_STR.strip()
    params_str = f"{agent.active_params:,}/{agent.total_params:,}"

    if agent.active_params > 10_000:
        active, total = (
            number_to_short(agent.active_params),
            number_to_short(agent.total_params),
        )
        params_str += f"({active}/{total})"

    cb_str = "\n\nActive Callbacks:"
    cb_str += "\n---------------------------------\n"
    cb_str += "\n".join(cb.info().lstrip() for cb in callbacks)
    cb_str += "\n---------------------------------\n"

    output += cb_str if callbacks else "\n\n"
    output += f"Training '{agent.__class__.__name__}' agent on '{env_id}' for '{number_to_short(n_episodes)}' episodes.\n"
    output += f"Using '{agent.buffer.__class__.__name__}' with 'capacity={number_to_short(agent.buffer.capacity)}'.\n"
    output += f"Sampling episodes with '{batch_size=}'.\n"
    output += f"Running computations on device '{device}'.\n"
    output += f"Moving averages computed based on 'window_size={number_to_short(window_size)}'.\n"
    output += f"Using networks with '{params_str}' active parameters.\n"
    output += "---------------------------------"

    print(output)
