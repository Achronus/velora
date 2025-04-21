import json
from typing import Tuple
import pytest

import torch

from velora.buffer.experience import BatchExperience
from velora.buffer.replay import ReplayBuffer
from velora.models.config import BufferConfig
from velora.models.nf.agent import NeuroFlowCT


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

    def test_buffer_warm(self, replay_buffer: ReplayBuffer):
        model = NeuroFlowCT("InvertedPendulum-v5", 8, 16, device=torch.device("cpu"))
        assert len(replay_buffer) == 0

        n_samples = 10
        model.buffer.warm(model, n_samples, 2)

        assert len(model.buffer) >= n_samples

    def test_add_multi(self, replay_buffer: ReplayBuffer):
        """Test adding multiple experiences at once using add_multi."""
        # Create test data: batch of 5 experiences
        batch_size = 5
        state_dim = replay_buffer.state_dim
        action_dim = replay_buffer.action_dim
        hidden_dim = replay_buffer.hidden_dim

        # Create batch tensors
        states = torch.rand(batch_size, state_dim)
        actions = torch.rand(batch_size, action_dim)
        rewards = torch.rand(batch_size, 1)
        next_states = torch.rand(batch_size, state_dim)
        dones = torch.zeros(batch_size, 1)  # All false
        dones[-1] = 1  # Make the last one done
        hiddens = torch.rand(batch_size, hidden_dim)

        # Store initial position and size
        initial_position = replay_buffer.position
        initial_size = replay_buffer.size

        # Add the batch of experiences
        replay_buffer.add_multi(states, actions, rewards, next_states, dones, hiddens)

        # Verify size increased correctly
        assert replay_buffer.size == min(
            initial_size + batch_size, replay_buffer.capacity
        )

        # Verify position updated correctly
        expected_position = (initial_position + batch_size) % replay_buffer.capacity
        assert replay_buffer.position == expected_position

        # Compute the indices where the experiences should have been stored
        indices = (
            torch.arange(initial_position, initial_position + batch_size)
            % replay_buffer.capacity
        )

        # Verify all experiences were stored correctly
        for i, idx in enumerate(indices):
            assert torch.allclose(
                replay_buffer.states[idx], states[i].to(torch.float32)
            )
            assert torch.allclose(replay_buffer.actions[idx], actions[i])
            assert torch.allclose(replay_buffer.rewards[idx], rewards[i])
            assert torch.allclose(
                replay_buffer.next_states[idx], next_states[i].to(torch.float32)
            )
            assert torch.allclose(replay_buffer.dones[idx], dones[i])
            assert torch.allclose(replay_buffer.hiddens[idx], hiddens[i])

    def test_add_multi_wrapping(self, replay_buffer: ReplayBuffer):
        """Test that add_multi correctly wraps around when it exceeds capacity."""
        # Fill the buffer close to capacity
        capacity = replay_buffer.capacity
        remaining_space = 5  # Leave a small amount of space
        fill_size = capacity - remaining_space

        # Create large batch to fill most of the buffer
        states = torch.zeros(fill_size, replay_buffer.state_dim)
        actions = torch.zeros(fill_size, replay_buffer.action_dim)
        rewards = torch.ones(fill_size, 1)  # Use ones to distinguish
        next_states = torch.zeros(fill_size, replay_buffer.state_dim)
        dones = torch.zeros(fill_size, 1)
        hiddens = torch.zeros(fill_size, replay_buffer.hidden_dim)

        # Fill the buffer
        replay_buffer.add_multi(states, actions, rewards, next_states, dones, hiddens)
        assert replay_buffer.size == fill_size
        assert replay_buffer.position == fill_size

        # Now add a batch that will wrap around
        wrap_size = remaining_space + 3  # 3 more than remaining space

        # Create batch that will wrap around
        wrap_states = torch.ones(
            wrap_size, replay_buffer.state_dim
        )  # Use ones to distinguish
        wrap_actions = torch.ones(wrap_size, replay_buffer.action_dim)
        wrap_rewards = torch.zeros(wrap_size, 1)  # Use zeros to distinguish
        wrap_next_states = torch.ones(wrap_size, replay_buffer.state_dim)
        wrap_dones = torch.zeros(wrap_size, 1)
        wrap_hiddens = torch.ones(wrap_size, replay_buffer.hidden_dim)

        # Add batch that should wrap
        replay_buffer.add_multi(
            wrap_states,
            wrap_actions,
            wrap_rewards,
            wrap_next_states,
            wrap_dones,
            wrap_hiddens,
        )

        # Verify size is now at capacity
        assert replay_buffer.size == capacity

        # Verify position wrapped around correctly
        expected_position = (fill_size + wrap_size) % capacity
        assert replay_buffer.position == expected_position

        # Check the last 'remaining_space' elements of the buffer should have ones from the wrap batch
        for i in range(capacity - remaining_space, capacity):
            wrap_idx = i - (capacity - remaining_space)
            assert torch.allclose(
                replay_buffer.states[i], wrap_states[wrap_idx].to(torch.float32)
            )
            assert torch.allclose(
                replay_buffer.rewards[i], wrap_rewards[wrap_idx]
            )  # Should be zeros

        # Check the first few elements should also have been overwritten with ones from wrap batch
        for i in range(expected_position):
            wrap_idx = remaining_space + i
            assert torch.allclose(
                replay_buffer.states[i], wrap_states[wrap_idx].to(torch.float32)
            )
            assert torch.allclose(
                replay_buffer.rewards[i], wrap_rewards[wrap_idx]
            )  # Should be zeros

    def test_add_multi_exceeds_capacity(self, replay_buffer: ReplayBuffer):
        """Test adding a batch larger than the buffer capacity."""
        capacity = replay_buffer.capacity

        # Create a batch twice the size of the buffer
        batch_size = capacity * 2

        states = torch.rand(batch_size, replay_buffer.state_dim)
        actions = torch.rand(batch_size, replay_buffer.action_dim)
        rewards = torch.rand(batch_size, 1)
        next_states = torch.rand(batch_size, replay_buffer.state_dim)
        dones = torch.zeros(batch_size, 1)
        hiddens = torch.rand(batch_size, replay_buffer.hidden_dim)

        # Add the oversized batch
        replay_buffer.add_multi(states, actions, rewards, next_states, dones, hiddens)

        # Verify size is capped at capacity
        assert replay_buffer.size == capacity

        # Verify position wrapped around correctly
        expected_position = batch_size % capacity
        assert replay_buffer.position == expected_position

        # The buffer should contain the last 'capacity' items from the batch
        start_idx = batch_size - capacity
        for i in range(capacity):
            buffer_idx = (i + expected_position) % capacity
            batch_idx = start_idx + i

            # Make sure to account for wrapping in the buffer
            if batch_idx >= batch_size:
                batch_idx = batch_idx - batch_size

            assert torch.allclose(
                replay_buffer.states[buffer_idx], states[batch_idx].to(torch.float32)
            )
            assert torch.allclose(replay_buffer.actions[buffer_idx], actions[batch_idx])
            assert torch.allclose(replay_buffer.rewards[buffer_idx], rewards[batch_idx])

    def test_add_then_add_multi(self, replay_buffer: ReplayBuffer):
        """Test adding single experiences followed by a batch."""
        # Add a few single experiences
        for i in range(3):
            state = torch.full((replay_buffer.state_dim,), float(i))
            action = torch.full((replay_buffer.action_dim,), float(i))
            reward = float(i)
            next_state = torch.full((replay_buffer.state_dim,), float(i + 1))
            done = False
            hidden = torch.full((replay_buffer.hidden_dim,), float(i))

            replay_buffer.add(state, action, reward, next_state, done, hidden)

        # Verify initial state
        assert replay_buffer.size == 3
        assert replay_buffer.position == 3

        # Create a batch to add
        batch_size = 4
        states = torch.full((batch_size, replay_buffer.state_dim), 10.0)
        actions = torch.full((batch_size, replay_buffer.action_dim), 10.0)
        rewards = torch.full((batch_size, 1), 10.0)
        next_states = torch.full((batch_size, replay_buffer.state_dim), 11.0)
        dones = torch.zeros(batch_size, 1)
        hiddens = torch.full((batch_size, replay_buffer.hidden_dim), 10.0)

        # Add the batch
        replay_buffer.add_multi(states, actions, rewards, next_states, dones, hiddens)

        # Verify updated state
        assert replay_buffer.size == 7
        assert replay_buffer.position == 7

        # Check individual experiences are still there
        for i in range(3):
            assert torch.allclose(
                replay_buffer.states[i],
                torch.full((replay_buffer.state_dim,), float(i)).to(torch.float32),
            )
            assert torch.allclose(replay_buffer.rewards[i], torch.tensor([[float(i)]]))

        # Check batch experiences were added correctly
        for i in range(4):
            assert torch.allclose(
                replay_buffer.states[i + 3], states[i].to(torch.float32)
            )
            assert torch.allclose(replay_buffer.rewards[i + 3], rewards[i])
