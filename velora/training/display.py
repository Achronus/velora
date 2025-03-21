import textwrap
from typing import TYPE_CHECKING, List

from velora.utils.format import number_to_short

if TYPE_CHECKING:
    from velora.callbacks import TrainCallback  # pragma: no cover


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
    cb_str = ""

    if len(callbacks) > 0:
        cb_str += "Active Callbacks:"
        cb_str += "\n---------------------------------"

        for cb in callbacks:
            cb_str += cb.info()

        cb_str += "\n---------------------------------\n"

    print(
        textwrap.dedent(f"""
        __     __   _                 
        \\ \\   / /__| | ___  _ __ __ _ 
         \\ \\ / / _ \\ |/ _ \\| '__/ _` |
          \\ V /  __/ | (_) | | | (_| |
           \\_/ \\___|_|\\___/|_|  \\__,_|
        ---------------------------------
        {cb_str}Training '{agent_name}' agent on '{env_id}' for '{number_to_short(n_episodes)}' episodes.
        Sampling episodes with '{batch_size=}'.
        Running computations on device '{device}'.
        Moving averages computed based on 'window_size={number_to_short(window_size)}'.
        ---------------------------------""")
    )
