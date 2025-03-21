from typing import TYPE_CHECKING, List

from velora.utils.format import number_to_short

if TYPE_CHECKING:
    from velora.callbacks import TrainCallback  # pragma: no cover


NAME_STR = """
__     __   _                 
\\ \\   / /__| | ___  _ __ __ _ 
 \\ \\ / / _ \\ |/ _ \\| '__/ _` |
  \\ V /  __/ | (_) | | | (_| |
   \\_/ \\___|_|\\___/|_|  \\__,_|
"""


def training_info(
    agent_name: str,
    env_id: str,
    n_episodes: int,
    batch_size: int,
    window_size: int,
    callbacks: List["TrainCallback"],
    device: str,
) -> None:
    """
    Display's starting information to the console for a training run.
    """
    output = NAME_STR.strip()

    cb_str = "\n\nActive Callbacks:"
    cb_str += "\n---------------------------------\n"
    cb_str += "\n".join(cb.info().lstrip() for cb in callbacks)
    cb_str += "\n---------------------------------\n"

    output += cb_str if callbacks else "\n\n"
    output += f"Training '{agent_name}' agent on '{env_id}' for '{number_to_short(n_episodes)}' episodes.\n"
    output += f"Sampling episodes with '{batch_size=}'.\n"
    output += f"Running computations on device '{device}'.\n"
    output += f"Moving averages computed based on 'window_size={number_to_short(window_size)}'.\n"
    output += "---------------------------------"

    print(output)
