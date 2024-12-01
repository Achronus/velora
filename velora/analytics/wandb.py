from typing import Any
from pydantic import PrivateAttr

import wandb
from wandb.sdk.wandb_run import Run

from velora.analytics.base import Analytics
from velora.exc import RunNotFoundError


class WeightsAndBiases(Analytics):
    """A class dedicated to working with [Weights and Biases](https://wandb.ai/)."""

    _run: Run = PrivateAttr(None)

    def model_post_init(self, __context):
        wandb.login()

    @property
    def run(self) -> Run:
        """Returns the active W&B run."""
        return self._run

    def init(
        self, project_name: str, run_name: str, config: dict[str, Any], **kwargs
    ) -> Run:
        """
        Starts a new run to track and log W&B and returns it.

        Args:
            project_name (str): The name of the project the run belongs to
            run_name (str): The unique display name for the run
            config (dict[str, Any]): A dictionary of inputs and hyperparameters used during the run
        """
        self._run = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            **kwargs,
        )
        return self._run

    def log(self, metrics: dict[str, Any], **kwargs) -> None:
        """
        Logs metrics to the Weights and Biases run.

        Args:
            metrics (dict[str, Any]): a dictionary of metrics to log with names and values
        """
        if not isinstance(self._run, Run):
            raise RunNotFoundError(
                "No run instance found. Have you called the 'init' method?"
            )

        self._run.log(data=metrics, **kwargs)

    def finish(self, **kwargs) -> None:
        """Marks the run as finished, uploads final data, and resets run instance to None."""
        if not isinstance(self._run, Run):
            raise RunNotFoundError(
                "No run instance found. Have you called the 'init' method?"
            )

        self._run.finish(**kwargs)
        self._run = None
