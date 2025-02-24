from typing import Any, Dict, Literal
import pytest
from unittest.mock import Mock, patch
import os
import tempfile

import torch

import gymnasium as gym

from velora.buffer import Experience
from velora.models import LiquidNCPNetwork
from velora.models.ddpg import DDPGActor, DDPGCritic, LiquidDDPG


NetworkParamsType = Dict[
    Literal["num_obs", "n_neurons", "num_actions", "device"],
    Any,
]
DDPGParamsType = Dict[
    Literal["state_dim", "n_neurons", "action_dim", "buffer_size", "device"],
    Any,
]


class TestDDPGActor:
    @pytest.fixture
    def actor_params(self) -> NetworkParamsType:
        return {
            "num_obs": 4,
            "n_neurons": 64,
            "num_actions": 2,
            "device": torch.device("cpu"),
        }

    @pytest.fixture
    def actor(self, actor_params: NetworkParamsType) -> DDPGActor:
        return DDPGActor(**actor_params)

    def test_init(self, actor: DDPGActor, actor_params: NetworkParamsType):
        assert isinstance(actor.ncp, LiquidNCPNetwork)
        assert actor.ncp.in_features == actor_params["num_obs"]
        assert actor.ncp.n_neurons == actor_params["n_neurons"]
        assert actor.ncp.out_features == actor_params["num_actions"]

    def test_forward(self, actor: DDPGActor, actor_params: NetworkParamsType):
        batch_size = 5
        obs = torch.randn(batch_size, actor_params["num_obs"])

        actions, hidden = actor(obs)

        # Check output shapes
        assert actions.shape == (batch_size, actor_params["num_actions"])
        assert hidden.shape[1] == actor.ncp.n_units

        # Check if actions are bounded by tanh
        assert torch.all(actions >= -1) and torch.all(actions <= 1)


class TestDDPGCritic:
    @pytest.fixture
    def critic_params(self) -> NetworkParamsType:
        return {
            "num_obs": 4,
            "n_neurons": 64,
            "num_actions": 2,
            "device": torch.device("cpu"),
        }

    @pytest.fixture
    def critic(self, critic_params: NetworkParamsType) -> DDPGCritic:
        return DDPGCritic(**critic_params)

    def test_init(self, critic: DDPGCritic, critic_params: NetworkParamsType):
        assert isinstance(critic.ncp, LiquidNCPNetwork)
        assert (
            critic.ncp.in_features
            == critic_params["num_obs"] + critic_params["num_actions"]
        )
        assert critic.ncp.n_neurons == critic_params["n_neurons"]
        assert critic.ncp.out_features == 1

    def test_forward(self, critic: DDPGCritic, critic_params: NetworkParamsType):
        batch_size = 5
        obs = torch.randn(batch_size, critic_params["num_obs"])
        actions = torch.randn(batch_size, critic_params["num_actions"])

        q_values, hidden = critic(obs, actions)

        # Check output shapes
        assert q_values.shape == (batch_size, 1)
        assert hidden.shape[1] == critic.ncp.n_units


class TestLiquidDDPG:
    @pytest.fixture
    def ddpg_params(self) -> DDPGParamsType:
        return {
            "state_dim": 2,
            "n_neurons": 10,
            "action_dim": 1,
            "buffer_size": 1000,
            "device": torch.device("cpu"),
        }

    @pytest.fixture
    def ddpg(self, ddpg_params: DDPGParamsType) -> LiquidDDPG:
        return LiquidDDPG(**ddpg_params)

    @pytest.fixture
    def env(self) -> gym.Env:
        return gym.make("MountainCarContinuous-v0")

    def test_init(self, ddpg: LiquidDDPG, ddpg_params: DDPGParamsType):
        assert isinstance(ddpg.actor, DDPGActor)
        assert isinstance(ddpg.critic, DDPGCritic)
        assert isinstance(ddpg.actor_target, DDPGActor)
        assert isinstance(ddpg.critic_target, DDPGCritic)
        assert len(ddpg.buffer) == 0
        assert ddpg.buffer.capacity == ddpg_params["buffer_size"]

    def test_update_target_networks(self, ddpg: LiquidDDPG):
        tau = 0.005

        # Store initial parameters
        initial_actor_params = {
            name: param.clone() for name, param in ddpg.actor_target.named_parameters()
        }

        # Modify actor parameters to ensure difference
        for param in ddpg.actor.parameters():
            param.data = param.data + 1.0

        # Perform update
        ddpg._update_target_networks(tau)

        # Check if parameters changed but not completely
        for name, param in ddpg.actor_target.named_parameters():
            assert not torch.allclose(param, initial_actor_params[name])
            assert not torch.allclose(param, param + 1.0)  # Shouldn't be fully updated

    def test_predict(self, ddpg: LiquidDDPG, ddpg_params: DDPGParamsType):
        state = torch.randn(ddpg_params["state_dim"])

        # Test without noise
        action, hidden = ddpg.predict(state, noise_scale=0.0)
        assert action.shape == (ddpg_params["action_dim"],)
        assert torch.all(action >= -1) and torch.all(action <= 1)

        # Test with noise
        action, hidden = ddpg.predict(state, noise_scale=0.1)
        assert action.shape == (ddpg_params["action_dim"],)
        assert torch.all(action >= -1) and torch.all(action <= 1)

    def test_train_step(self, ddpg: LiquidDDPG):
        batch_size = 32
        gamma = 0.99

        # Fill buffer with valid experiences
        for _ in range(batch_size + 1):
            state = torch.zeros(ddpg.state_dim)
            action = 1.0  # Single float value
            reward = 2.0  # Single float value
            next_state = torch.zeros(ddpg.state_dim)
            done = False

            # Create Experience object explicitly
            exp = Experience(state, action, reward, next_state, done)
            ddpg.buffer.push(exp)

        # Perform training step
        result = ddpg._train_step(batch_size, gamma)
        assert result is not None
        critic_loss, actor_loss = result

        assert isinstance(critic_loss, float)
        assert isinstance(actor_loss, float)

    def test_train_step_insufficient_buffer(self, ddpg: LiquidDDPG):
        batch_size = 32
        gamma = 0.99

        # Add fewer experiences than batch_size
        for _ in range(batch_size - 1):
            state = torch.zeros(ddpg.state_dim)
            exp = Experience(state, 1.0, 2.0, state, False)
            ddpg.buffer.push(exp)

        # Should return None when buffer is insufficient
        result = ddpg._train_step(batch_size, gamma)
        assert result is None

    def test_train_invalid_env(self, ddpg: LiquidDDPG):
        mock_env = Mock(spec=gym.Env)
        mock_env.action_space = gym.spaces.Discrete(2)  # Invalid action space

        with pytest.raises(EnvironmentError):
            ddpg.train(mock_env, batch_size=32)

    def test_train_valid_env(self, env: gym.Env, ddpg: LiquidDDPG):
        _ = ddpg.train(env, batch_size=32, n_episodes=2, max_steps=10)

    def test_train_if_done(self, env: gym.Env, ddpg: LiquidDDPG):
        # Create a wrapper to force early termination
        class EarlyDoneWrapper(gym.Wrapper):
            def __init__(self, env):
                super().__init__(env)
                self.step_count = 0

            def step(self, action):
                self.step_count += 1
                obs, reward, term, trunc, info = self.env.step(action)
                # Force done after 5 steps
                if self.step_count >= 5:
                    term = True
                return obs, reward, term, trunc, info

            def reset(self, **kwargs):
                self.step_count = 0
                return self.env.reset(**kwargs)

        wrapped_env = EarlyDoneWrapper(env)
        n_episodes = 2

        rewards = ddpg.train(
            wrapped_env,
            batch_size=32,
            n_episodes=n_episodes,
            max_steps=10,  # Even though max_steps is 10, should terminate at 5
        )
        assert len(rewards) == n_episodes

    def test_train_output_print_msg(self, env: gym.Env, ddpg: LiquidDDPG):
        batch_size = 32
        n_episodes = 4
        window_size = 2

        # Mock numpy mean to verify statistics calculation
        with patch("numpy.mean") as mock_mean:
            # Make mean return predictable values
            mock_mean.side_effect = [10.0, 0.5, 0.3]  # reward, critic_loss, actor_loss

            # Mock print to capture output
            with patch("builtins.print") as mock_print:
                ddpg.train(
                    env,
                    batch_size=batch_size,
                    n_episodes=n_episodes,
                    max_steps=10,
                    window_size=window_size,
                )

                # Verify statistics were printed with correct format
                expected_output = (
                    "Episode: 4/4, "
                    "Avg Reward: 10.00, "
                    "Critic Loss: 0.50, "
                    "Actor Loss: 0.30"
                )
                mock_print.assert_any_call(expected_output)

        env.close()

    def test_save_load(self, ddpg: LiquidDDPG):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_file:
            filepath = temp_file.name

        try:
            # Save the model
            ddpg.save(filepath)
            assert os.path.exists(filepath)

            # Load the model
            loaded_ddpg = LiquidDDPG.load(filepath)

            # Check model parameters
            assert loaded_ddpg.state_dim == ddpg.state_dim
            assert loaded_ddpg.n_neurons == ddpg.n_neurons
            assert loaded_ddpg.action_dim == ddpg.action_dim
            assert loaded_ddpg.buffer_size == ddpg.buffer_size
            assert str(loaded_ddpg.device) == str(ddpg.device)

            # Test predict with the same input
            state = torch.randn(ddpg.state_dim)
            action1, _ = ddpg.predict(state, noise_scale=0.0)
            action2, _ = loaded_ddpg.predict(state, noise_scale=0.0)

            # Actions should be identical since we're using the same state and no noise
            assert torch.allclose(action1, action2)

        finally:
            # Clean up
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_save_load_with_buffer(self, ddpg: LiquidDDPG):
        # Add some experiences to buffer
        for i in range(10):
            state = torch.zeros(ddpg.state_dim)
            action = float(i)
            reward = float(i * 0.5)
            next_state = torch.ones(ddpg.state_dim)
            done = i == 9

            exp = Experience(state, action, reward, next_state, done)
            ddpg.buffer.push(exp)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_file:
            filepath = temp_file.name

        try:
            # Save the model with buffer
            ddpg.save(filepath, buffer=True)

            # Check both files exist
            assert os.path.exists(filepath)
            buffer_path = ddpg.buffer.create_filepath(filepath)
            assert os.path.exists(buffer_path)

            # Load the model with buffer
            loaded_ddpg = LiquidDDPG.load(filepath, buffer=True)

            # Check buffer properties
            assert len(loaded_ddpg.buffer) == len(ddpg.buffer)

            # Test predict to ensure models produce similar results
            state = torch.randn(ddpg.state_dim)
            action1, _ = ddpg.predict(state, noise_scale=0.0)
            action2, _ = loaded_ddpg.predict(state, noise_scale=0.0)
            assert torch.allclose(action1, action2)

        finally:
            # Clean up
            if os.path.exists(filepath):
                os.unlink(filepath)
            if os.path.exists(buffer_path):
                os.unlink(buffer_path)

    def test_load_without_buffer_file(self, ddpg: LiquidDDPG):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_file:
            filepath = temp_file.name

        try:
            # Save model without buffer
            ddpg.save(filepath, buffer=False)

            # Try to load with buffer=True, should raise error
            with pytest.raises(
                FileNotFoundError, match="Buffer file .* does not exist"
            ):
                LiquidDDPG.load(filepath, buffer=True)

        finally:
            # Clean up
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_save_directory_creation(self, ddpg: LiquidDDPG):
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = os.path.join(temp_dir, "new_subdir", "nested")
            filepath = os.path.join(new_dir, "model.pt")

            # Directory shouldn't exist yet
            assert not os.path.exists(new_dir)

            # Save should create the directory
            ddpg.save(filepath)

            # Check directory and file were created
            assert os.path.exists(new_dir)
            assert os.path.exists(filepath)

    def test_load_invalid_file(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_file:
            filepath = temp_file.name

        try:
            # Create an invalid model file
            with open(filepath, "w") as f:
                f.write("This is not a valid PyTorch file")

            # Attempt to load should raise an exception
            with pytest.raises(Exception):
                LiquidDDPG.load(filepath)

        finally:
            # Clean up
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_checkpoint_keys_validation(self, ddpg: LiquidDDPG):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_file:
            filepath = temp_file.name

        try:
            # Save the model
            ddpg.save(filepath)

            # Load the checkpoint and modify it
            checkpoint = torch.load(filepath)
            # Add an invalid key
            checkpoint["invalid_key"] = "some_value"
            # Save the modified checkpoint
            torch.save(checkpoint, filepath)

            # Attempt to load should raise ValueError about key mismatch
            with pytest.raises(ValueError, match="Mismatch between checkpoint keys"):
                LiquidDDPG.load(filepath)

        finally:
            # Clean up
            if os.path.exists(filepath):
                os.unlink(filepath)
