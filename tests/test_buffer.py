import json
from typing import Tuple
import pytest

import gymnasium as gym
import torch

from velora.buffer.experience import BatchExperience
from velora.buffer.replay import ReplayBuffer
from velora.models.config import BufferConfig
from velora.models.ddpg import LiquidDDPG


class TestBatchExperience:
    @pytest.fixture
    def batch_experience(self) -> BatchExperience:
        return BatchExperience(
            states=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            actions=torch.tensor([1.0, 2.0]),
            rewards=torch.tensor([1.0, 2.0]),
            next_states=torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
            dones=torch.tensor([False, True]),
            hiddens=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        )

    def test_batch_experience_creation(self, batch_experience: BatchExperience) -> None:
        assert isinstance(batch_experience.states, torch.Tensor)
        assert isinstance(batch_experience.actions, torch.Tensor)
        assert isinstance(batch_experience.rewards, torch.Tensor)
        assert isinstance(batch_experience.next_states, torch.Tensor)
        assert isinstance(batch_experience.dones, torch.Tensor)
        assert isinstance(batch_experience.hiddens, torch.Tensor)

    def test_batch_experience_shapes(self, batch_experience: BatchExperience) -> None:
        assert batch_experience.states.shape == (2, 2)
        assert batch_experience.actions.shape == (2,)
        assert batch_experience.rewards.shape == (2,)
        assert batch_experience.next_states.shape == (2, 2)
        assert batch_experience.dones.shape == (2,)
        assert batch_experience.hiddens.shape == (2, 2)


class TestReplayBuffer:
    @pytest.fixture
    def replay_buffer(self) -> ReplayBuffer:
        return ReplayBuffer(
            capacity=100,
            state_dim=2,
            action_dim=1,
            hidden_dim=2,
            device="cpu",
        )

    @pytest.fixture
    def sample_experience(self) -> Tuple:
        return (
            torch.tensor([1.0, 2.0]),
            torch.tensor([1.0]),
            1.0,
            torch.tensor([2.0, 3.0]),
            False,
            torch.tensor([2.0, 3.0]),
        )

    @pytest.fixture
    def filled_buffer(self, replay_buffer: ReplayBuffer) -> ReplayBuffer:
        """Fixture that returns a replay buffer with 10 experiences."""
        for i in range(10):
            replay_buffer.add(
                torch.tensor([float(i), float(i + 1)]),
                torch.tensor([i]),
                float(i * 0.5),
                torch.tensor([float(i + 1), float(i + 2)]),
                (i == 9),
                torch.tensor([float(i), float(i + 1)]),
            )
        return replay_buffer

    def test_buffer_init(self, replay_buffer: ReplayBuffer) -> None:
        assert replay_buffer.capacity == 100
        assert replay_buffer.state_dim == 2
        assert replay_buffer.action_dim == 1
        assert replay_buffer.hidden_dim == 2
        assert isinstance(replay_buffer.states, torch.Tensor)
        assert isinstance(replay_buffer.actions, torch.Tensor)
        assert isinstance(replay_buffer.rewards, torch.Tensor)
        assert isinstance(replay_buffer.next_states, torch.Tensor)
        assert isinstance(replay_buffer.dones, torch.Tensor)
        assert isinstance(replay_buffer.hiddens, torch.Tensor)
        assert len(replay_buffer) == 0

    def test_config(self, replay_buffer: ReplayBuffer):
        config = replay_buffer.config()
        assert config == BufferConfig(
            type="ReplayBuffer",
            capacity=100,
            state_dim=2,
            action_dim=1,
            hidden_dim=2,
        )

    def test_push_experience(
        self, replay_buffer: ReplayBuffer, sample_experience: tuple
    ) -> None:
        replay_buffer.add(*sample_experience)
        assert len(replay_buffer) == 1
        assert isinstance(replay_buffer.states, torch.Tensor)

    def test_buffer_capacity(
        self, replay_buffer: ReplayBuffer, sample_experience: Tuple
    ) -> None:
        # Fill buffer beyond capacity
        for _ in range(150):
            replay_buffer.add(*sample_experience)
        assert len(replay_buffer) == 100  # Should not exceed capacity

    def test_sample_insufficient_experiences(
        self, replay_buffer: ReplayBuffer, sample_experience: tuple
    ) -> None:
        replay_buffer.add(*sample_experience)
        with pytest.raises(ValueError):
            replay_buffer.sample(batch_size=2)

    def test_sample_batch(
        self, replay_buffer: ReplayBuffer, sample_experience: tuple
    ) -> None:
        # Fill buffer with multiple experiences
        for _ in range(10):
            replay_buffer.add(*sample_experience)

        batch_size = 5
        batch = replay_buffer.sample(batch_size)

        assert isinstance(batch, BatchExperience)
        assert batch.states.shape[0] == batch_size
        assert batch.actions.shape[0] == batch_size
        assert batch.rewards.shape[0] == batch_size
        assert batch.next_states.shape[0] == batch_size
        assert batch.dones.shape[0] == batch_size
        assert batch.hiddens.shape[0] == batch_size

    def test_len_method(
        self, replay_buffer: ReplayBuffer, sample_experience: tuple
    ) -> None:
        assert len(replay_buffer) == 0
        replay_buffer.add(*sample_experience)
        assert len(replay_buffer) == 1

    def test_state_dict_empty_buffer(self, replay_buffer: ReplayBuffer) -> None:
        state_dict = replay_buffer.state_dict()

        # Check that state_dict returns the correct tensors
        assert "states" in state_dict
        assert "actions" in state_dict
        assert "rewards" in state_dict
        assert "next_states" in state_dict
        assert "dones" in state_dict
        assert "hiddens" in state_dict

        # Check that buffer fields are tensors with correct shapes
        assert state_dict["states"].shape == (100, 2)
        assert state_dict["actions"].shape == (100, 1)
        assert state_dict["rewards"].shape == (100, 1)
        assert state_dict["next_states"].shape == (100, 2)
        assert state_dict["dones"].shape == (100, 1)
        assert state_dict["hiddens"].shape == (100, 2)

        # Check metadata
        metadata = replay_buffer.metadata()
        assert metadata["capacity"] == 100
        assert metadata["state_dim"] == 2
        assert metadata["action_dim"] == 1
        assert metadata["hidden_dim"] == 2
        assert metadata["device"] == "cpu"
        assert metadata["position"] == 0
        assert metadata["size"] == 0

    def test_state_dict_filled_buffer(self, filled_buffer: ReplayBuffer) -> None:
        state_dict = filled_buffer.state_dict()

        # Check that state_dict returns the correct tensors
        assert "states" in state_dict
        assert "actions" in state_dict
        assert "rewards" in state_dict
        assert "next_states" in state_dict
        assert "dones" in state_dict
        assert "hiddens" in state_dict

        # Get metadata to verify buffer state
        metadata = filled_buffer.metadata()
        assert metadata["capacity"] == 100
        assert metadata["state_dim"] == 2
        assert metadata["action_dim"] == 1
        assert metadata["hidden_dim"] == 2
        assert metadata["device"] == "cpu"
        assert metadata["size"] == 10
        assert metadata["position"] == 10

        # Validate first few items in the buffer
        assert torch.allclose(state_dict["states"][0], torch.tensor([0.0, 1.0]))
        assert torch.allclose(state_dict["actions"][1], torch.tensor([1.0]))
        assert torch.allclose(state_dict["rewards"][2], torch.tensor([1.0]))
        assert torch.allclose(state_dict["next_states"][3], torch.tensor([4.0, 5.0]))
        assert torch.allclose(state_dict["dones"][9], torch.tensor([1.0]))
        assert torch.allclose(state_dict["hiddens"][0], torch.tensor([0.0, 1.0]))

    def test_save_load(self, filled_buffer: ReplayBuffer, tmp_path) -> None:
        # Create directory
        buffer_dir = tmp_path / "buffer_dir"
        buffer_dir.mkdir(parents=True, exist_ok=True)

        # Save buffer
        filled_buffer.save(buffer_dir)

        # Check files exist
        metadata_path = buffer_dir / "buffer_metadata.json"
        state_path = buffer_dir / "buffer_state.safetensors"

        assert metadata_path.exists()
        assert state_path.exists()

        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Load buffer
        loaded_buffer = ReplayBuffer.load(state_path, metadata)

        # Check properties
        assert loaded_buffer.capacity == filled_buffer.capacity
        assert str(loaded_buffer.device) == str(
            filled_buffer.device
        )  # Compare as strings
        assert loaded_buffer.position == filled_buffer.position
        assert loaded_buffer.size == filled_buffer.size
        assert len(loaded_buffer) == len(filled_buffer)

        # Sample and check
        original_batch = filled_buffer.sample(batch_size=5)
        loaded_batch = loaded_buffer.sample(batch_size=5)

        assert original_batch.states.shape == loaded_batch.states.shape
        assert original_batch.actions.shape == loaded_batch.actions.shape
        assert original_batch.rewards.shape == loaded_batch.rewards.shape
        assert original_batch.next_states.shape == loaded_batch.next_states.shape
        assert original_batch.dones.shape == loaded_batch.dones.shape
        assert original_batch.hiddens.shape == loaded_batch.hiddens.shape

    def test_save_directory_creation(
        self, filled_buffer: ReplayBuffer, tmp_path
    ) -> None:
        # Create a directory
        nested_dir = tmp_path / "nested_dir"
        nested_dir.mkdir(parents=True, exist_ok=True)

        # Save the buffer
        filled_buffer.save(nested_dir)

        # Check files
        assert (nested_dir / "buffer_metadata.json").exists()
        assert (nested_dir / "buffer_state.safetensors").exists()

    def test_buffer_warm(self):
        device = torch.device("cpu")
        env = gym.make("InvertedPendulum-v5", render_mode="rgb_array")

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        agent = LiquidDDPG(state_dim, 8, action_dim, device=device)
        hidden_dim = agent.actor.ncp.hidden_size

        buffer = ReplayBuffer(
            capacity=100,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            device=device,
        )

        # Verify initial empty state
        assert len(buffer) == 0

        # Test warming the buffer
        n_samples = 15
        buffer.warm(agent, env.spec.id, n_samples)

        # Verify buffer has been filled with experiences
        assert len(buffer) == n_samples

        # Add another experience manually
        buffer.add(
            torch.zeros(state_dim, device=device),
            torch.zeros(action_dim, device=device),
            1.0,
            torch.zeros(state_dim, device=device),
            False,
            torch.zeros(hidden_dim, device=device),
        )

        # Verify buffer length increases
        assert len(buffer) == n_samples + 1

        # Test sampling from buffer
        batch = buffer.sample(batch_size=n_samples)

        # Verify batch structure
        assert hasattr(batch, "states")
        assert hasattr(batch, "actions")
        assert hasattr(batch, "rewards")
        assert hasattr(batch, "next_states")
        assert hasattr(batch, "dones")
        assert hasattr(batch, "hiddens")

        # Verify batch shapes
        assert batch.states.shape[0] == n_samples
        assert batch.actions.shape[0] == n_samples
        assert batch.rewards.shape[0] == n_samples
        assert batch.next_states.shape[0] == n_samples
        assert batch.dones.shape[0] == n_samples
        assert batch.hiddens.shape[0] == n_samples
