from typing import Any, Dict, Literal, Self

from pydantic import BaseModel


class BufferConfig(BaseModel):
    """
    A config model for buffer details.

    Attributes:
        type: the type of buffer
        capacity: the maximum capacity of the buffer
        state_dim: dimension of state observations
        action_dim: dimension of actions
        hidden_dim: dimension of hidden state
    """

    type: Literal["ReplayBuffer", "RolloutBuffer"]
    capacity: int
    state_dim: int
    action_dim: int
    hidden_dim: int


class TorchConfig(BaseModel):
    """
    A config model for PyTorch details.

    Attributes:
        device: the device used to train the model
        optimizer: the type of optimizer used
        loss: the type of optimizer used
    """

    device: str
    optimizer: str
    loss: str


class TrainConfig(BaseModel):
    """
    A base config model for training parameter details.

    Attributes:
        batch_size: the size of the training batch
        n_episodes: the total number of episodes trained for
        window_size: reward moving average size (in episodes)
        display_count: console training progress frequency (in episodes)
        gamma: the reward discount factor
        callbacks: a dictionary of callback details. Default is `None`
    """

    batch_size: int
    n_episodes: int
    window_size: int
    display_count: int
    gamma: float
    callbacks: Dict[str, Any] | None = None


class TrainConfig(TrainConfig):
    """
    A config model for episodic training parameter details.

    Attributes:
        batch_size: the size of the training batch
        n_episodes: the total number of episodes trained for
        window_size: the episodic rate for calculating the reward moving
            average
        gamma: the reward discount factor
        callbacks: a dictionary of callback details
        max_steps: the maximum number of steps per training episode
        noise_scale: the exploration noise added when selecting
            an action. Default is `None`
        tau: the soft update factor used to slowly update the
            target networks. Default is `None`
        warmup_steps: number of random steps to take before starting
            training. Default is `None`
        update_every: how often to update the networks. Default is `None`
    """

    max_steps: int
    tau: float | None = None
    noise_scale: float | None = None
    warmup_steps: int | None = None
    update_every: int | None = None


class ModuleConfig(BaseModel):
    """
    A config model for a module's details.

    Attributes:
        active_params: active module parameters count
        total_params: total module parameter count
        architecture: a summary of the module's architecture
    """

    active_params: int
    total_params: int
    architecture: Dict[str, Any]


class SACExtraParameters(BaseModel):
    """
    A config model for extra parameters for the Soft Actor-Critic (SAC) agent.

    Attributes:
        alpha_lr: the entropy parameter learning rate
        initial_alpha: the starting entropy coefficient value
        target_entropy: the target entropy for automatic adjustment
        log_std_min: lower bound for the log standard deviation of the
            action distribution
        log_std_max: upper bound for the log standard deviation of the
            action distribution
    """

    alpha_lr: float
    initial_alpha: float
    target_entropy: float
    log_std_min: float
    log_std_max: float


class ModelDetails(BaseModel):
    """
    A config model for storing an agent's network model details.

    Attributes:
        type: the type of architecture used
        state_dim: number of input features
        n_neurons: number of hidden node
        action_dim: number of output features
        action_type: the type of action space. Default is `continuous`
        target_networks: whether the agent uses target networks or not.
            Default is `False`
        exploration_type: the type of agent exploration used
        actor: details about the Actor network
        critic: details about the Critic network
        critic2: details about the second Critic network. Default is `None`
        extras: extra parameters unique to the agent. Default is `None`
    """

    type: Literal["actor-critic"]
    state_dim: int
    n_neurons: int
    action_dim: int
    action_type: Literal["discrete", "continuous"] = "continuous"
    target_networks: bool = False
    exploration_type: Literal["OUNoise", "Entropy"]
    actor: ModuleConfig
    critic: ModuleConfig
    critic2: ModuleConfig | None = None
    extras: SACExtraParameters | None = None


class RLAgentConfig(BaseModel):
    """
    A config model for RL agents. Stored with agent states during the `save()` method.

    Attributes:
        agent: the type of agent used
        env: the name of the environment the model was trained on. Default is `None`
        seed: random number generator value
        model_details: the agent's network model details
        buffer: the buffer details
        torch: the PyTorch details
        train_params: the agents training parameters. Default is `None`
    """

    agent: str
    env: str | None = None
    seed: int
    model_details: ModelDetails
    buffer: BufferConfig
    torch: TorchConfig
    train_params: TrainConfig | None = None

    def update(self, env: str, train_params: TrainConfig) -> Self:
        """
        Updates the training details of the model.

        Parameters:
            env (str): the environment name
            train_params (TrainConfig): a config containing training parameters

        Returns:
            self (Self): a new config model with the updated values.
        """
        return RLAgentConfig(
            env=env,
            train_params=train_params,
            **self.model_dump(exclude={"env", "train_params"}),
        )
