import json
from typing import Tuple
import pytest

import gymnasium as gym
import torch

from velora.buffer.experience import BatchExperience
from velora.buffer.replay import ReplayBuffer
from velora.buffer.rollout import RolloutBuffer
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
        )

    def test_batch_experience_creation(self, batch_experience: BatchExperience) -> None:
        assert isinstance(batch_experience.states, torch.Tensor)
        assert isinstance(batch_experience.actions, torch.Tensor)
        assert isinstance(batch_experience.rewards, torch.Tensor)
        assert isinstance(batch_experience.next_states, torch.Tensor)
        assert isinstance(batch_experience.dones, torch.Tensor)

    def test_batch_experience_shapes(self, batch_experience: BatchExperience) -> None:
        assert batch_experience.states.shape == (2, 2)
        assert batch_experience.actions.shape == (2,)
        assert batch_experience.rewards.shape == (2,)
        assert batch_experience.next_states.shape == (2, 2)
        assert batch_experience.dones.shape == (2,)


class TestReplayBuffer:
    @pytest.fixture
    def replay_buffer(self) -> ReplayBuffer:
        return ReplayBuffer(capacity=100, state_dim=2, action_dim=1, device="cpu")

    @pytest.fixture
    def sample_experience(self) -> Tuple:
        return (
            torch.tensor([1.0, 2.0]),
            torch.tensor([1.0]),
            1.0,
            torch.tensor([2.0, 3.0]),
            False,
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
            )
        return replay_buffer

    def test_buffer_init(self, replay_buffer: ReplayBuffer) -> None:
        assert replay_buffer.capacity == 100
        assert replay_buffer.state_dim == 2
        assert replay_buffer.action_dim == 1
        assert isinstance(replay_buffer.states, torch.Tensor)
        assert isinstance(replay_buffer.actions, torch.Tensor)
        assert isinstance(replay_buffer.rewards, torch.Tensor)
        assert isinstance(replay_buffer.next_states, torch.Tensor)
        assert isinstance(replay_buffer.dones, torch.Tensor)
        assert len(replay_buffer) == 0

    def test_config(self, replay_buffer: ReplayBuffer):
        config = replay_buffer.config()
        assert config == BufferConfig(
            type="ReplayBuffer",
            capacity=100,
            state_dim=2,
            action_dim=1,
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

        # Check that buffer fields are tensors with correct shapes
        assert state_dict["states"].shape == (100, 2)
        assert state_dict["actions"].shape == (100, 1)
        assert state_dict["rewards"].shape == (100, 1)
        assert state_dict["next_states"].shape == (100, 2)
        assert state_dict["dones"].shape == (100, 1)

        # Check metadata
        metadata = replay_buffer.metadata()
        assert metadata["capacity"] == 100
        assert metadata["state_dim"] == 2
        assert metadata["action_dim"] == 1
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

        # Get metadata to verify buffer state
        metadata = filled_buffer.metadata()
        assert metadata["capacity"] == 100
        assert metadata["state_dim"] == 2
        assert metadata["action_dim"] == 1
        assert metadata["device"] == "cpu"
        assert metadata["size"] == 10
        assert metadata["position"] == 10

        # Validate first few items in the buffer
        assert torch.allclose(state_dict["states"][0], torch.tensor([0.0, 1.0]))
        assert torch.allclose(state_dict["actions"][1], torch.tensor([1.0]))
        assert torch.allclose(state_dict["rewards"][2], torch.tensor([1.0]))
        assert torch.allclose(state_dict["next_states"][3], torch.tensor([4.0, 5.0]))
        assert torch.allclose(state_dict["dones"][9], torch.tensor([1.0]))

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
        buffer = ReplayBuffer(
            capacity=100, state_dim=state_dim, action_dim=action_dim, device=device
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

        # Verify batch shapes
        assert batch.states.shape[0] == n_samples
        assert batch.actions.shape[0] == n_samples
        assert batch.rewards.shape[0] == n_samples
        assert batch.next_states.shape[0] == n_samples
        assert batch.dones.shape[0] == n_samples


class TestRolloutBuffer:
    @pytest.fixture
    def rollout_buffer(self) -> RolloutBuffer:
        return RolloutBuffer(capacity=5, state_dim=2, action_dim=1, device="cpu")

    @pytest.fixture
    def sample_experience(self) -> Tuple:
        return (
            torch.tensor([1.0, 2.0]),
            torch.tensor([1.0]),
            1.0,
            torch.tensor([2.0, 3.0]),
            False,
        )

    @pytest.fixture
    def filled_buffer(self, rollout_buffer: RolloutBuffer) -> RolloutBuffer:
        """Fixture that returns a filled rollout buffer with 3 experiences."""
        for i in range(3):
            rollout_buffer.add(
                state=torch.tensor([float(i), float(i + 1)]),
                action=torch.tensor([i]),
                reward=float(i * 0.5),
                next_state=torch.tensor([float(i + 1), float(i + 2)]),
                done=(i == 2),
            )
        return rollout_buffer

    def test_buffer_init(self, rollout_buffer: RolloutBuffer) -> None:
        assert rollout_buffer.capacity == 5
        assert rollout_buffer.state_dim == 2
        assert rollout_buffer.action_dim == 1
        assert isinstance(rollout_buffer.states, torch.Tensor)
        assert isinstance(rollout_buffer.actions, torch.Tensor)
        assert isinstance(rollout_buffer.rewards, torch.Tensor)
        assert isinstance(rollout_buffer.next_states, torch.Tensor)
        assert isinstance(rollout_buffer.dones, torch.Tensor)
        assert len(rollout_buffer) == 0

    def test_config(self, rollout_buffer: RolloutBuffer):
        config = rollout_buffer.config()
        assert config == BufferConfig(
            type="RolloutBuffer",
            capacity=5,
            state_dim=2,
            action_dim=1,
        )

    def test_push_experience(
        self, rollout_buffer: RolloutBuffer, sample_experience: Tuple
    ) -> None:
        rollout_buffer.add(*sample_experience)
        assert len(rollout_buffer) == 1
        assert isinstance(rollout_buffer.states, torch.Tensor)

    def test_buffer_capacity_error(
        self, rollout_buffer: RolloutBuffer, sample_experience: Tuple
    ) -> None:
        # Fill buffer to capacity
        for _ in range(5):
            rollout_buffer.add(*sample_experience)

        # Attempt to push when buffer is full
        with pytest.raises(BufferError):
            rollout_buffer.add(*sample_experience)

    def test_sample_empty_buffer(self, rollout_buffer: RolloutBuffer) -> None:
        with pytest.raises(BufferError) as exc_info:
            rollout_buffer.sample()
        assert str(exc_info.value) == "Buffer is empty!"

    def test_sample_buffer(
        self, rollout_buffer: RolloutBuffer, sample_experience: Tuple
    ) -> None:
        # Fill buffer with experiences
        num_experiences = 3
        for _ in range(num_experiences):
            rollout_buffer.add(*sample_experience)

        batch = rollout_buffer.sample()
        cap = rollout_buffer.capacity

        # Verify batch properties
        assert isinstance(batch, BatchExperience)
        assert batch.states.shape[0] == cap
        assert batch.states.shape[1] == 2
        assert batch.actions.shape[0] == cap
        assert batch.rewards.shape[0] == cap
        assert batch.next_states.shape[0] == cap
        assert batch.next_states.shape[1] == 2
        assert batch.dones.shape[0] == cap

    def test_clear_buffer(
        self, rollout_buffer: RolloutBuffer, sample_experience: Tuple
    ) -> None:
        # Add some experiences
        for _ in range(3):
            rollout_buffer.add(*sample_experience)
        assert len(rollout_buffer) == 3

        # Clear buffer
        rollout_buffer.empty()
        assert len(rollout_buffer) == 0

    def test_len_method(
        self, rollout_buffer: RolloutBuffer, sample_experience: Tuple
    ) -> None:
        assert len(rollout_buffer) == 0
        rollout_buffer.add(*sample_experience)
        assert len(rollout_buffer) == 1
        rollout_buffer.add(*sample_experience)
        assert len(rollout_buffer) == 2
        rollout_buffer.empty()
        assert len(rollout_buffer) == 0

    def test_state_dict_empty_buffer(self, rollout_buffer: RolloutBuffer) -> None:
        state_dict = rollout_buffer.state_dict()

        # Check that state_dict returns the correct tensors
        assert "states" in state_dict
        assert "actions" in state_dict
        assert "rewards" in state_dict
        assert "next_states" in state_dict
        assert "dones" in state_dict

        # Check that buffer tensors are initialized with correct shapes
        assert state_dict["states"].shape == (5, 2)
        assert state_dict["actions"].shape == (5, 1)
        assert state_dict["rewards"].shape == (5, 1)
        assert state_dict["next_states"].shape == (5, 2)
        assert state_dict["dones"].shape == (5, 1)

        # Check metadata
        metadata = rollout_buffer.metadata()
        assert metadata["capacity"] == 5
        assert metadata["state_dim"] == 2
        assert metadata["action_dim"] == 1
        assert metadata["device"] == "cpu"
        assert metadata["position"] == 0
        assert metadata["size"] == 0

    def test_state_dict_filled_buffer(self, filled_buffer: RolloutBuffer) -> None:
        state_dict = filled_buffer.state_dict()

        # Check that state_dict returns the correct tensors
        assert "states" in state_dict
        assert "actions" in state_dict
        assert "rewards" in state_dict
        assert "next_states" in state_dict
        assert "dones" in state_dict

        # Get metadata to verify buffer state
        metadata = filled_buffer.metadata()
        assert metadata["capacity"] == 5
        assert metadata["state_dim"] == 2
        assert metadata["action_dim"] == 1
        assert metadata["device"] == "cpu"
        assert metadata["size"] == 3  # We added 3 experiences
        assert metadata["position"] == 3

        # Test first experience
        assert torch.allclose(state_dict["states"][0], torch.tensor([0.0, 1.0]))
        assert torch.allclose(state_dict["actions"][0], torch.tensor([0.0]))
        assert torch.allclose(state_dict["rewards"][0], torch.tensor([0.0]))
        assert torch.allclose(state_dict["next_states"][0], torch.tensor([1.0, 2.0]))
        assert torch.allclose(state_dict["dones"][0], torch.tensor([0.0]))

        # Test last experience (position 2)
        assert torch.allclose(state_dict["states"][2], torch.tensor([2.0, 3.0]))
        assert torch.allclose(state_dict["actions"][2], torch.tensor([2.0]))
        assert torch.allclose(state_dict["rewards"][2], torch.tensor([1.0]))
        assert torch.allclose(state_dict["next_states"][2], torch.tensor([3.0, 4.0]))
        assert torch.allclose(state_dict["dones"][2], torch.tensor([1.0]))

    def test_save_load(self, filled_buffer: RolloutBuffer, tmp_path) -> None:
        # Create save path and ensure directory exists
        save_dir = tmp_path / "buffer_dir"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save the buffer to a file within the directory
        buffer_file = save_dir / "buffer_file"
        buffer_file.mkdir(parents=True, exist_ok=True)

        filled_buffer.save(buffer_file)

        # Check the files exist
        buffer_metadata_path = buffer_file / "buffer_metadata.json"
        buffer_state_path = buffer_file / "buffer_state.safetensors"

        assert buffer_metadata_path.exists(), "buffer_metadata.json doesn't exist"
        assert buffer_state_path.exists(), "buffer_state.safetensors doesn't exist"

        # Load the metadata
        with open(buffer_metadata_path, "r") as f:
            metadata = json.load(f)

        # Load the buffer
        loaded_buffer = RolloutBuffer.load(buffer_state_path, metadata)

        # Check properties
        assert loaded_buffer.capacity == filled_buffer.capacity
        assert str(loaded_buffer.device) == str(filled_buffer.device)
        assert loaded_buffer.position == filled_buffer.position
        assert loaded_buffer.size == filled_buffer.size
        assert len(loaded_buffer) == len(filled_buffer)

        # Check experiences by sampling
        original_batch = filled_buffer.sample()
        loaded_batch = loaded_buffer.sample()

        # With RolloutBuffer, we can compare directly as sampling isn't random
        assert torch.allclose(original_batch.states, loaded_batch.states)
        assert torch.allclose(original_batch.actions, loaded_batch.actions)
        assert torch.allclose(original_batch.rewards, loaded_batch.rewards)
        assert torch.allclose(original_batch.next_states, loaded_batch.next_states)
        assert torch.allclose(original_batch.dones, loaded_batch.dones)

    def test_save_directory_creation(
        self, filled_buffer: RolloutBuffer, tmp_path
    ) -> None:
        # Create nested directory path and ensure it exists
        nested_dir = tmp_path / "new_subdir" / "nested"
        nested_dir.mkdir(parents=True, exist_ok=True)

        # Create save path in the directory
        save_path = nested_dir / "buffer_save"
        save_path.mkdir(parents=True, exist_ok=True)

        # Save should work now
        filled_buffer.save(save_path)

        # Check files were created
        assert (save_path / "buffer_metadata.json").exists()
        assert (save_path / "buffer_state.safetensors").exists()

    def test_empty_after_save(self, filled_buffer: RolloutBuffer, tmp_path) -> None:
        # Create a save path
        save_path = tmp_path / "buffer_save"
        save_path.mkdir(parents=True, exist_ok=True)

        # Save the buffer
        filled_buffer.save(save_path)
        buffer_state_path = save_path / "buffer_state.safetensors"
        buffer_metadata_path = save_path / "buffer_metadata.json"

        # Empty buffer
        filled_buffer.empty()
        assert len(filled_buffer) == 0

        # Check metadata reflects empty buffer
        metadata = filled_buffer.metadata()
        assert metadata["size"] == 0
        assert metadata["position"] == 0

        # Buffer tensors still exist but position and size are reset
        state_dict = filled_buffer.state_dict()
        assert state_dict["states"].shape == (5, 2)
        assert state_dict["actions"].shape == (5, 1)

        # Load the metadata from file
        with open(buffer_metadata_path, "r") as f:
            saved_metadata = json.load(f)

        # Should be able to reload from file
        loaded_buffer = RolloutBuffer.load(buffer_state_path, saved_metadata)
        assert len(loaded_buffer) == 3  # Original size before emptying

        # Add more experiences to emptied buffer
        filled_buffer.add(
            state=torch.tensor([10.0, 11.0]),
            action=torch.tensor([10.0]),
            reward=5.0,
            next_state=torch.tensor([11.0, 12.0]),
            done=False,
        )
        assert len(filled_buffer) == 1

    def test_load_and_continue_filling(self, tmp_path) -> None:
        # Create and fill a buffer
        buffer = RolloutBuffer(capacity=5, state_dim=2, action_dim=1)
        for i in range(3):
            buffer.add(
                state=torch.tensor([float(i), float(i + 1)]),
                action=torch.tensor([i]),
                reward=float(i * 0.5),
                next_state=torch.tensor([float(i + 1), float(i + 2)]),
                done=(i == 2),
            )

        # Create save path and ensure it exists
        save_path = tmp_path / "buffer_save"
        save_path.mkdir(parents=True, exist_ok=True)

        # Save the buffer
        buffer.save(save_path)

        # Load the metadata
        buffer_metadata_path = save_path / "buffer_metadata.json"
        buffer_state_path = save_path / "buffer_state.safetensors"

        with open(buffer_metadata_path, "r") as f:
            metadata = json.load(f)

        # Load the buffer
        loaded_buffer = RolloutBuffer.load(buffer_state_path, metadata)
        assert len(loaded_buffer) == 3

        # Add more experiences
        loaded_buffer.add(
            state=torch.tensor([10.0, 11.0]),
            action=torch.tensor([10.0]),
            reward=5.0,
            next_state=torch.tensor([11.0, 12.0]),
            done=False,
        )

        assert len(loaded_buffer) == 4

        # Try to add experiences up to capacity
        loaded_buffer.add(
            state=torch.tensor([11.0, 12.0]),
            action=torch.tensor([11.0]),
            reward=5.5,
            next_state=torch.tensor([12.0, 13.0]),
            done=True,
        )

        assert len(loaded_buffer) == 5

        # Should raise error on next push
        with pytest.raises(BufferError, match="Buffer full!"):
            loaded_buffer.add(
                state=torch.tensor([12.0, 13.0]),
                action=torch.tensor([12.0]),
                reward=6.0,
                next_state=torch.tensor([13.0, 14.0]),
                done=False,
            )
