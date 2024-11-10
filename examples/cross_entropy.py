from pathlib import Path
from typing import Any, Generator
import gymnasium as gym
import numpy as np
from pydantic import BaseModel

from velora import History, Trajectory, Episodes

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import wandb

from velora.config import Config


class WBConfig(BaseModel):
    env: str
    seed: int
    hidden_size: int
    batch_size: int
    percentile: int | float
    solve_threshold: int | float
    model: str
    optimizer: dict[str, Any]
    criterion: str


def save_net_spec(model: nn.Module, folder_path: Path | str) -> None:
    def write_file(filepath: str, content: str) -> None:
        with open(Path(folder_path, filepath), "w") as f:
            f.write(content)

    folder_path = Path(folder_path)
    if folder_path.is_dir():
        folder_path.mkdir(exist_ok=True)

        write_file(f"{model._get_name()}.txt", str(model))
        torch.save(model.state_dict(), Path(folder_path, f"{model._get_name()}.pt"))
    else:
        raise NotADirectoryError(f"'{folder_path}' is not a directory!")


def iterate_batches(
    env: gym.Env, net: nn.Module, batch_size: int
) -> Generator[Episodes, None, None]:
    batch = Episodes()
    episode = History()
    obs, _ = env.reset()

    while True:
        action_scores = net(obs.unsqueeze(0))
        probs: torch.Tensor = torch.softmax(action_scores, dim=-1)
        action = Categorical(probs).sample().item()

        next_obs, reward, terminated, truncated, _ = env.step(action)

        episode.add(
            Trajectory(
                action=action,
                observation=obs,
                reward=float(reward),
            )
        )

        if terminated or truncated:
            batch.add(episode)
            episode = History()
            next_obs, _ = env.reset()

            if len(batch) == batch_size:
                yield batch
                batch = Episodes()

        obs = next_obs


def filter_batch_cartpole(
    batch: Episodes, percentile: int
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    ep_scores = batch.scores().numpy()
    reward_bound = np.percentile(ep_scores, percentile).item()
    reward_mean = np.mean(ep_scores).item()

    best_batches = Episodes()

    for ep in batch:
        if ep.score() >= reward_bound:
            best_batches.add(ep)

    return (
        best_batches.observations(),
        best_batches.actions(),
        reward_bound,
        reward_mean,
    )


def train_cartpole(
    env: gym.Env,
    net: nn.Module,
    loss: nn.Module,
    optimizer: optim.Optimizer,
    config: Config,
    run_idx: int,
) -> None:
    net_name = f"{net._get_name()}"

    optim_config = {"name": optimizer.__class__.__name__}
    if config.optimizer:
        optim_config.update(**config.optimizer)

    wb_config = WBConfig(
        env=config.env.name,
        seed=config.env.seed,
        **config.model.model_dump(),
        **config.other.model_dump(),
        model=net_name,
        optimizer=optim_config,
        criterion=loss._get_name(),
    )

    run = wandb.init(
        project="CrossEntropyTest",
        name=f"CP-run-{run_idx}",
        config=wb_config.model_dump(),
        job_type=env.spec.name,
        tags=[env.spec.name, net_name],
    )
    run.watch(net, criterion=loss, log="all")

    model_artifact = wandb.Artifact(
        net_name, type="model", metadata=wb_config.model_dump()
    )
    model_artifact.add_file(f"saved/{net_name}.pt")
    model_artifact.add_file(f"saved/{net_name}.txt")
    run.log_artifact(model_artifact)

    for i_batch, batch in enumerate(iterate_batches(env, net, config.model.batch_size)):
        obs_v, acts_v, reward_b, reward_m = filter_batch_cartpole(
            batch, config.other.percentile
        )

        if len(obs_v) == 0:
            continue

        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v: torch.Tensor = loss(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()

        print(
            f"{i_batch}: loss={loss_v:.3f}, reward_mean={reward_m:.1f}, rw_bound={reward_b:.1f}"
        )
        run.log(
            {
                "ep_idx": i_batch,
                "loss": loss_v,
                "reward_mean": reward_m,
                "reward_bound": reward_b,
            }
        )

        if reward_m > config.other.solve_threshold:
            print("Solved!")
            run.finish()
            break
