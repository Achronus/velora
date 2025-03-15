import pytest
import os
import tempfile
from collections import deque
from pathlib import Path

import gymnasium as gym
import torch

from velora.buffer.experience import BatchExperience, Experience
from velora.buffer.replay import ReplayBuffer
from velora.buffer.rollout import RolloutBuffer
from velora.models.config import BufferConfig
from velora.models.ddpg import LiquidDDPG


class TestExperience:
    @pytest.fixture
    def experience(self) -> Experience:
        return Experience(
            state=torch.tensor([1.0, 2.0]),
            action=1.0,
            reward=1.0,
            next_state=torch.tensor([2.0, 3.0]),
            done=False,
        )

    def test_experience_creation(self, experience: Experience) -> None:
        assert isinstance(experience.state, torch.Tensor)
        assert isinstance(experience.action, float)
        assert isinstance(experience.reward, float)
        assert isinstance(experience.next_state, torch.Tensor)
        assert isinstance(experience.done, bool)

    def test_experience_iteration(self, experience: Experience) -> None:
        state, action, reward, next_state, done = experience
        assert torch.equal(state, torch.tensor([1.0, 2.0]))
        assert action == 1.0
        assert reward == 1.0
        assert torch.equal(next_state, torch.tensor([2.0, 3.0]))
        assert done is False


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
        return ReplayBuffer(capacity=100)

    @pytest.fixture
    def sample_experience(self) -> Experience:
        return Experience(
            state=torch.tensor([1.0, 2.0]),
            action=torch.tensor([1.0]),
            reward=1.0,
            next_state=torch.tensor([2.0, 3.0]),
            done=False,
        )

    @pytest.fixture
    def filled_buffer(self, replay_buffer: ReplayBuffer) -> ReplayBuffer:
        """Fixture that returns a replay buffer with 10 experiences."""
        for i in range(10):
            replay_buffer.push(
                Experience(
                    state=torch.tensor([float(i), float(i + 1)]),
                    action=torch.tensor([i]),
                    reward=float(i * 0.5),
                    next_state=torch.tensor([float(i + 1), float(i + 2)]),
                    done=(i == 9),
                )
            )
        return replay_buffer

    def test_buffer_init(self, replay_buffer: ReplayBuffer) -> None:
        assert replay_buffer.capacity == 100
        assert isinstance(replay_buffer.buffer, deque)
        assert len(replay_buffer.buffer) == 0

    def test_config(self, replay_buffer: ReplayBuffer):
        config = replay_buffer.config
        assert config == BufferConfig(type="ReplayBuffer", capacity=100)

    def test_push_experience(
        self, replay_buffer: ReplayBuffer, sample_experience: Experience
    ) -> None:
        replay_buffer.push(sample_experience)
        assert len(replay_buffer) == 1
        assert isinstance(replay_buffer.buffer[0], Experience)

    def test_buffer_capacity(
        self, replay_buffer: ReplayBuffer, sample_experience: Experience
    ) -> None:
        # Fill buffer beyond capacity
        for _ in range(150):
            replay_buffer.push(sample_experience)
        assert len(replay_buffer) == 100  # Should not exceed capacity

    def test_sample_insufficient_experiences(
        self, replay_buffer: ReplayBuffer, sample_experience: Experience
    ) -> None:
        replay_buffer.push(sample_experience)
        with pytest.raises(ValueError):
            replay_buffer.sample(batch_size=2)

    def test_sample_batch(
        self, replay_buffer: ReplayBuffer, sample_experience: Experience
    ) -> None:
        # Fill buffer with multiple experiences
        for _ in range(10):
            replay_buffer.push(sample_experience)

        batch_size = 5
        batch = replay_buffer.sample(batch_size)

        assert isinstance(batch, BatchExperience)
        assert batch.states.shape[0] == batch_size
        assert batch.actions.shape[0] == batch_size
        assert batch.rewards.shape[0] == batch_size
        assert batch.next_states.shape[0] == batch_size
        assert batch.dones.shape[0] == batch_size

    def test_len_method(
        self, replay_buffer: ReplayBuffer, sample_experience: Experience
    ) -> None:
        assert len(replay_buffer) == 0
        replay_buffer.push(sample_experience)
        assert len(replay_buffer) == 1

    def test_state_dict_empty_buffer(self, replay_buffer: ReplayBuffer) -> None:
        state_dict = replay_buffer.state_dict()

        assert state_dict["capacity"] == 100
        assert state_dict["device"] is None

        # Check that buffer fields are empty lists
        for key in ["states", "actions", "rewards", "next_states", "dones"]:
            assert state_dict["buffer"][key] == []

    def test_state_dict_filled_buffer(self, filled_buffer: ReplayBuffer) -> None:
        state_dict = filled_buffer.state_dict()

        assert state_dict["capacity"] == 100
        assert state_dict["device"] is None

        # Check buffer contents
        buffer_data = state_dict["buffer"]
        assert len(buffer_data["states"]) == 10
        assert len(buffer_data["actions"]) == 10
        assert len(buffer_data["rewards"]) == 10
        assert len(buffer_data["next_states"]) == 10
        assert len(buffer_data["dones"]) == 10

        # Check specific values
        assert buffer_data["states"][0] == [0.0, 1.0]
        assert buffer_data["actions"][1] == [1.0]
        assert buffer_data["rewards"][2] == 1.0
        assert buffer_data["next_states"][3] == [4.0, 5.0]
        assert buffer_data["dones"][9] == 1.0

    def test_save_load(self, filled_buffer: ReplayBuffer) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_file:
            filepath = temp_file.name

        try:
            # Save the buffer
            filled_buffer.save(filepath)
            assert os.path.exists(filepath)

            # Load the buffer
            loaded_buffer = ReplayBuffer.load(filepath)

            # Check properties
            assert loaded_buffer.capacity == filled_buffer.capacity
            assert loaded_buffer.device == filled_buffer.device
            assert len(loaded_buffer) == len(filled_buffer)

            # Check experiences by sampling
            original_batch = filled_buffer.sample(batch_size=5)
            loaded_batch = loaded_buffer.sample(batch_size=5)

            # Due to random sampling, we can't directly compare the batches
            # But we can check they have the same shape and general characteristics
            assert original_batch.states.shape == loaded_batch.states.shape
            assert original_batch.actions.shape == loaded_batch.actions.shape
            assert original_batch.rewards.shape == loaded_batch.rewards.shape
            assert original_batch.next_states.shape == loaded_batch.next_states.shape
            assert original_batch.dones.shape == loaded_batch.dones.shape

        finally:
            # Clean up
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_create_filepath(self) -> None:
        # Test with string path
        string_path = "models/checkpoint.pt"
        buffer_path = ReplayBuffer.create_filepath(string_path)
        assert buffer_path == Path("models", "checkpoint.buffer.pt")

        # Test with Path object
        path_obj = Path("models/checkpoint.pth")
        buffer_path = ReplayBuffer.create_filepath(path_obj)
        assert buffer_path == Path("models", "checkpoint.buffer.pth")

        # Test with multiple extensions
        complex_path = "models/run1.model.pt"
        buffer_path = ReplayBuffer.create_filepath(complex_path)
        assert buffer_path == Path("models", "run1.model.buffer.pt")

    def test_save_directory_creation(self, filled_buffer: ReplayBuffer) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = os.path.join(temp_dir, "new_subdir", "nested")
            filepath = os.path.join(new_dir, "buffer.pt")

            # Directory shouldn't exist yet
            assert not os.path.exists(new_dir)

            # Save should create the directory
            filled_buffer.save(filepath)

            # Check directory and file were created
            assert os.path.exists(new_dir)
            assert os.path.exists(filepath)

    def test_buffer_warm(self):
        device = torch.device("cpu")
        env = gym.make("InvertedPendulum-v5", render_mode="rgb_array")

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        agent = LiquidDDPG(state_dim, 8, action_dim, device=device)
        buffer = ReplayBuffer(capacity=100, device=device)

        # Verify initial empty state
        assert len(buffer) == 0

        # Test warming the buffer
        n_samples = 15
        buffer.warm(agent, env.spec.id, n_samples)

        # Verify buffer has been filled with experiences
        assert len(buffer) == n_samples

        # Add another experience manually
        exp = Experience(
            state=torch.zeros(state_dim, device=device),
            action=torch.tensor([0.5]),
            reward=1.0,
            next_state=torch.zeros(state_dim, device=device),
            done=False,
        )
        buffer.push(exp)

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
        return RolloutBuffer(capacity=5)

    @pytest.fixture
    def sample_experience(self) -> Experience:
        return Experience(
            state=torch.tensor([1.0, 2.0]),
            action=torch.tensor([1.0]),
            reward=1.0,
            next_state=torch.tensor([2.0, 3.0]),
            done=False,
        )

    @pytest.fixture
    def filled_buffer(self, rollout_buffer: RolloutBuffer) -> RolloutBuffer:
        """Fixture that returns a filled rollout buffer with 3 experiences."""
        for i in range(3):
            rollout_buffer.push(
                Experience(
                    state=torch.tensor([float(i), float(i + 1)]),
                    action=torch.tensor([i]),
                    reward=float(i * 0.5),
                    next_state=torch.tensor([float(i + 1), float(i + 2)]),
                    done=(i == 2),
                )
            )
        return rollout_buffer

    def test_buffer_init(self, rollout_buffer: RolloutBuffer) -> None:
        assert rollout_buffer.capacity == 5
        assert isinstance(rollout_buffer.buffer, deque)
        assert len(rollout_buffer.buffer) == 0

    def test_config(self, rollout_buffer: RolloutBuffer):
        config = rollout_buffer.config
        assert config == BufferConfig(type="RolloutBuffer", capacity=5)

    def test_push_experience(
        self, rollout_buffer: RolloutBuffer, sample_experience: Experience
    ) -> None:
        rollout_buffer.push(sample_experience)
        assert len(rollout_buffer) == 1
        assert isinstance(rollout_buffer.buffer[0], Experience)

    def test_buffer_capacity_error(
        self, rollout_buffer: RolloutBuffer, sample_experience: Experience
    ) -> None:
        # Fill buffer to capacity
        for _ in range(5):
            rollout_buffer.push(sample_experience)

        # Attempt to push when buffer is full
        with pytest.raises(BufferError):
            rollout_buffer.push(sample_experience)

    def test_sample_empty_buffer(self, rollout_buffer: RolloutBuffer) -> None:
        with pytest.raises(BufferError) as exc_info:
            rollout_buffer.sample()
        assert str(exc_info.value) == "Buffer is empty!"

    def test_sample_buffer(
        self, rollout_buffer: RolloutBuffer, sample_experience: Experience
    ) -> None:
        # Fill buffer with experiences
        num_experiences = 3
        for _ in range(num_experiences):
            rollout_buffer.push(sample_experience)

        batch = rollout_buffer.sample()

        # Verify batch properties
        assert isinstance(batch, BatchExperience)
        assert batch.states.shape == (num_experiences, 2)
        assert batch.actions.shape == (num_experiences, 1)
        assert batch.rewards.shape == (num_experiences, 1)
        assert batch.next_states.shape == (num_experiences, 2)
        assert batch.dones.shape == (num_experiences, 1)

    def test_clear_buffer(
        self, rollout_buffer: RolloutBuffer, sample_experience: Experience
    ) -> None:
        # Add some experiences
        for _ in range(3):
            rollout_buffer.push(sample_experience)
        assert len(rollout_buffer) == 3

        # Clear buffer
        rollout_buffer.empty()
        assert len(rollout_buffer) == 0

    def test_len_method(
        self, rollout_buffer: RolloutBuffer, sample_experience: Experience
    ) -> None:
        assert len(rollout_buffer) == 0
        rollout_buffer.push(sample_experience)
        assert len(rollout_buffer) == 1
        rollout_buffer.push(sample_experience)
        assert len(rollout_buffer) == 2
        rollout_buffer.empty()
        assert len(rollout_buffer) == 0

    def test_state_dict_empty_buffer(self, rollout_buffer: RolloutBuffer) -> None:
        state_dict = rollout_buffer.state_dict()

        assert state_dict["capacity"] == 5
        assert state_dict["device"] is None

        # Check that buffer fields are empty lists
        for key in ["states", "actions", "rewards", "next_states", "dones"]:
            assert state_dict["buffer"][key] == []

    def test_state_dict_filled_buffer(self, filled_buffer: RolloutBuffer) -> None:
        state_dict = filled_buffer.state_dict()

        assert state_dict["capacity"] == 5
        assert state_dict["device"] is None

        # Check buffer contents
        buffer_data = state_dict["buffer"]
        assert len(buffer_data["states"]) == 3
        assert len(buffer_data["actions"]) == 3
        assert len(buffer_data["rewards"]) == 3
        assert len(buffer_data["next_states"]) == 3
        assert len(buffer_data["dones"]) == 3

        # Check specific values
        assert buffer_data["states"][0] == [0.0, 1.0]
        assert buffer_data["actions"][1] == [1.0]
        assert buffer_data["rewards"][2] == 1.0
        assert buffer_data["next_states"][1] == [2.0, 3.0]
        assert buffer_data["dones"][2] == 1.0

    def test_save_load(self, filled_buffer: RolloutBuffer) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_file:
            filepath = temp_file.name

        try:
            # Save the buffer
            filled_buffer.save(filepath)
            assert os.path.exists(filepath)

            # Load the buffer
            loaded_buffer = RolloutBuffer.load(filepath)

            # Check properties
            assert loaded_buffer.capacity == filled_buffer.capacity
            assert loaded_buffer.device == filled_buffer.device
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

        finally:
            # Clean up
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_create_filepath(self) -> None:
        # Test with string path
        string_path = "models/checkpoint.pt"
        buffer_path = RolloutBuffer.create_filepath(string_path)
        assert buffer_path == Path("models", "checkpoint.buffer.pt")

        # Test with Path
        path_obj = Path("models/checkpoint.pth")
        buffer_path = RolloutBuffer.create_filepath(path_obj)
        assert buffer_path == Path("models", "checkpoint.buffer.pth")

    def test_save_directory_creation(self, filled_buffer: RolloutBuffer) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = os.path.join(temp_dir, "new_subdir", "nested")
            filepath = os.path.join(new_dir, "buffer.pt")

            # Directory shouldn't exist yet
            assert not os.path.exists(new_dir)

            # Save should create the directory
            filled_buffer.save(filepath)

            # Check directory and file were created
            assert os.path.exists(new_dir)
            assert os.path.exists(filepath)

    def test_empty_after_save(self, filled_buffer: RolloutBuffer) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_file:
            filepath = temp_file.name

        try:
            # Save the buffer
            filled_buffer.save(filepath)

            # Empty buffer
            filled_buffer.empty()
            assert len(filled_buffer) == 0

            # Check state dict reflects empty buffer
            state_dict = filled_buffer.state_dict()
            for key in ["states", "actions", "rewards", "next_states", "dones"]:
                assert state_dict["buffer"][key] == []

            # Should be able to reload from file
            loaded_buffer = RolloutBuffer.load(filepath)
            assert len(loaded_buffer) == 3  # Original size before emptying

            # Add more experiences to emptied buffer
            filled_buffer.push(
                Experience(
                    state=torch.tensor([10.0, 11.0]),
                    action=10.0,
                    reward=5.0,
                    next_state=torch.tensor([11.0, 12.0]),
                    done=False,
                )
            )
            assert len(filled_buffer) == 1

        finally:
            # Clean up
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_load_and_continue_filling(self) -> None:
        # Create and fill a buffer
        buffer = RolloutBuffer(capacity=5)
        for i in range(3):
            buffer.push(
                Experience(
                    state=torch.tensor([float(i), float(i + 1)]),
                    action=torch.tensor([i]),
                    reward=float(i * 0.5),
                    next_state=torch.tensor([float(i + 1), float(i + 2)]),
                    done=(i == 2),
                )
            )

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_file:
            filepath = temp_file.name

        try:
            # Save the buffer
            buffer.save(filepath)

            # Load the buffer
            loaded_buffer = RolloutBuffer.load(filepath)
            assert len(loaded_buffer) == 3

            # Add more experiences
            loaded_buffer.push(
                Experience(
                    state=torch.tensor([10.0, 11.0]),
                    action=torch.tensor([10.0]),
                    reward=5.0,
                    next_state=torch.tensor([11.0, 12.0]),
                    done=False,
                )
            )

            assert len(loaded_buffer) == 4

            # Try to add experiences up to capacity
            loaded_buffer.push(
                Experience(
                    state=torch.tensor([11.0, 12.0]),
                    action=torch.tensor([11.0]),
                    reward=5.5,
                    next_state=torch.tensor([12.0, 13.0]),
                    done=True,
                )
            )

            assert len(loaded_buffer) == 5

            # Should raise error on next push
            with pytest.raises(BufferError, match="Buffer full!"):
                loaded_buffer.push(
                    Experience(
                        state=torch.tensor([12.0, 13.0]),
                        action=torch.tensor([12.0]),
                        reward=6.0,
                        next_state=torch.tensor([13.0, 14.0]),
                        done=False,
                    )
                )

        finally:
            # Clean up
            if os.path.exists(filepath):
                os.unlink(filepath)
