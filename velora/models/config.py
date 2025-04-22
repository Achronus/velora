from typing import Any, Dict, Literal, Self, Tuple

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
    A config model for training parameter details.

    Attributes:
        batch_size: the size of the training batch
        n_episodes: the total number of episodes trained for. Default is `None`
        window_size: reward moving average size (in episodes)
        display_count: console training progress frequency (in episodes)
        log_freq: metric logging frequency (in episodes)
        callbacks: a dictionary of callback details. Default is `None`
        max_steps: the maximum number of steps per training episode.
            Default is `None`
        warmup_steps: number of random steps to take before starting
            training
    """

    batch_size: int
    n_episodes: int
    window_size: int
    display_count: int
    log_freq: int
    callbacks: Dict[str, Any] | None = None
    max_steps: int
    warmup_steps: int


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
            action distribution. Default is `None`
        log_std_max: upper bound for the log standard deviation of the
            action distribution. Default is `None`
    """

    alpha_lr: float
    initial_alpha: float
    target_entropy: float
    log_std_min: float | None = None
    log_std_max: float | None = None


class EntropyParameters(BaseModel):
    """
    A config model for extra parameters for NeuroFlow agents.

    Attributes:
        lr: the entropy parameter learning rate
        initial_alpha: the starting entropy coefficient value
        target: the target entropy for automatic adjustment
    """

    lr: float
    initial_alpha: float
    target: float


class CuriosityConfig(BaseModel):
    """
    A config model for the Intrinsic Curiosity Module (ICM).

    Attributes:
        icm: details about the ICM
        lr: the optimizers learning rate
        eta: importance scaling factor for intrinsic reward
        beta: weight balancing for inverse vs. forward model
    """

    icm: ModuleConfig
    lr: float
    eta: float
    beta: float


class CriticConfig(BaseModel):
    """
    A critic config model for storing a NeuroFlow agent's critic module details.

    Attributes:
        critic1: details about the first critic network
        critic2: details about the second critic network
    """

    critic1: ModuleConfig
    critic2: ModuleConfig


class ModelDetails(BaseModel):
    """
    A config model for storing an agent's network model details.

    Attributes:
        type: the type of architecture used. Default is `actor-critic`
        state_dim: number of input features
        actor_neurons: number of actor network decision nodes
        critic_neurons: number of critic network decision nodes
        action_dim: number of output features
        action_type: the type of action space. Default is `continuous`
        tau: the soft update factor for target networks
        gamma: the reward discount factor
        target_networks: whether the agent uses target networks or not.
            Default is `True`
        log_std: lower and upper bounds for the log standard deviation of the
            action distribution. Only required for `continuous` spaces.
            Default is `None`
        exploration_type: the type of agent exploration used
        actor: details about the Actor network
        critic: details about the Critic networks
        entropy: details about the entropy exploration
    """

    type: str = "actor-critic"
    state_dim: int
    actor_neurons: int
    critic_neurons: int
    action_dim: int
    tau: float
    gamma: float
    action_type: Literal["discrete", "continuous"] = "continuous"
    target_networks: bool = True
    log_std: Tuple[float, float] | None = None
    exploration_type: Literal["Entropy", "CAT-Entropy"]
    actor: ModuleConfig
    critic: CriticConfig
    entropy: EntropyParameters


class RLAgentConfig(BaseModel):
    """
    A config model for NeuroFlow agents. Stored with agent states during the
    `save()` method.

    Attributes:
        agent: the type of agent used
        env: the Gymnasium environment ID the model was trained on
        seed: random number generator value
        model_details: the agent's network model details
        buffer: the buffer details
        torch: the PyTorch details
        train_params: the agents training parameters. Default is `None`
    """

    agent: str
    env: str
    seed: int
    model_details: ModelDetails
    buffer: BufferConfig
    torch: TorchConfig
    train_params: TrainConfig | None = None

    def update(self, train_params: TrainConfig) -> Self:
        """
        Updates the training details of the model.

        Parameters:
            train_params (TrainConfig): a config containing training parameters

        Returns:
            self (Self): a new config model with the updated values.
        """
        return RLAgentConfig(
            train_params=train_params,
            **self.model_dump(exclude={"train_params"}),
        )
