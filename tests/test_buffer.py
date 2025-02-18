import pytest
from collections import deque

import torch

from velora.buffer import Experience, BatchExperience, ReplayBuffer, RolloutBuffer


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
            action=1.0,
            reward=1.0,
            next_state=torch.tensor([2.0, 3.0]),
            done=False,
        )

    def test_buffer_initialization(self, replay_buffer: ReplayBuffer) -> None:
        assert replay_buffer.capacity == 100
        assert isinstance(replay_buffer.buffer, deque)
        assert len(replay_buffer.buffer) == 0

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


class TestRolloutBuffer:
    @pytest.fixture
    def rollout_buffer(self) -> RolloutBuffer:
        return RolloutBuffer(capacity=5)

    @pytest.fixture
    def sample_experience(self) -> Experience:
        return Experience(
            state=torch.tensor([1.0, 2.0]),
            action=1.0,
            reward=1.0,
            next_state=torch.tensor([2.0, 3.0]),
            done=False,
        )

    def test_buffer_initialization(self, rollout_buffer: RolloutBuffer) -> None:
        assert rollout_buffer.capacity == 5
        assert isinstance(rollout_buffer.buffer, deque)
        assert len(rollout_buffer.buffer) == 0

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
        with pytest.raises(BufferError) as exc_info:
            rollout_buffer.push(sample_experience)
        assert str(exc_info.value) == "Buffer full! Use the 'clear()' method first."

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
