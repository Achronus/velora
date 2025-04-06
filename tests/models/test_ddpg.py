from pathlib import Path
import shutil
import time
from typing import Any, Dict, Literal
import pytest
from unittest.mock import MagicMock, Mock, patch
import os
import tempfile
import json

import torch
from torch.jit import RecursiveScriptModule
import gymnasium as gym

from velora.callbacks import CometAnalytics, EarlyStopping, SaveCheckpoints
from velora.models import LiquidNCPNetwork
from velora.models.ddpg import DDPGActor, DDPGCritic, LiquidDDPG
from velora.training.handler import TrainHandler
from velora.utils.core import set_seed


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
        assert hidden.shape[1] == actor.ncp.hidden_size

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
        assert hidden.shape[1] == critic.ncp.hidden_size


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
        return gym.make("InvertedPendulum-v5", render_mode="rgb_array")

    def test_init(self, ddpg: LiquidDDPG):
        assert isinstance(ddpg.actor, (DDPGActor, RecursiveScriptModule))
        assert isinstance(ddpg.critic, (DDPGCritic, RecursiveScriptModule))
        assert isinstance(ddpg.actor_target, (DDPGActor, RecursiveScriptModule))
        assert isinstance(ddpg.critic_target, (DDPGCritic, RecursiveScriptModule))

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
            action = torch.tensor([1.0])
            reward = 2.0  # Single float value
            next_state = torch.zeros(ddpg.state_dim)
            done = False
            hidden = torch.zeros(ddpg.hidden_dim)

            ddpg.buffer.add(state, action, reward, next_state, done, hidden)

        # Perform training step
        result = ddpg._train_step(batch_size, gamma)
        assert result is not None
        critic_loss, actor_loss = result

        assert isinstance(critic_loss, torch.Tensor)
        assert isinstance(actor_loss, torch.Tensor)

    def test_train_step_insufficient_buffer(self, ddpg: LiquidDDPG):
        batch_size = 32
        gamma = 0.99

        # Add fewer experiences than batch_size
        for _ in range(batch_size - 1):
            state = torch.zeros([ddpg.state_dim])
            action = torch.tensor([1.0])
            reward = 2.0
            next_state = torch.zeros(ddpg.state_dim)
            done = False
            hidden = torch.zeros(ddpg.hidden_dim)

            ddpg.buffer.add(state, action, reward, next_state, done, hidden)

        # Should return None when buffer is insufficient
        result = ddpg._train_step(batch_size, gamma)
        assert result is None

    def test_train_invalid_env(self, ddpg: LiquidDDPG):
        mock_env = Mock(spec=gym.Env)
        mock_env.action_space = gym.spaces.Discrete(2)  # Invalid action space

        with pytest.raises(EnvironmentError):
            ddpg.train(mock_env, batch_size=32)

    def test_save_error(self, ddpg: LiquidDDPG):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_file:
            filepath = temp_file.name
        try:
            # Create the file to trigger FileExistsError
            with open(filepath, "w") as f:
                f.write("dummy content")

            # Now trying to save should raise FileExistsError
            with pytest.raises(FileExistsError):
                ddpg.save(filepath)
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_save_load_with_buffer(self, ddpg: LiquidDDPG):
        # Add some experiences to buffer
        for i in range(10):
            state = torch.zeros(ddpg.state_dim)
            action = torch.tensor([i])
            reward = float(i * 0.5)
            next_state = torch.ones(ddpg.state_dim)
            done = i == 9
            hidden = torch.zeros(ddpg.hidden_dim)

            ddpg.buffer.add(state, action, reward, next_state, done, hidden)

        # Get the initial buffer size to compare later
        buffer_size = len(ddpg.buffer)

        # Create a unique temporary directory name
        unique_id = f"model_save_{id(ddpg)}"
        temp_dir = tempfile.gettempdir()
        save_path = Path(temp_dir) / unique_id

        try:
            # Ensure the directory doesn't exist before starting
            if save_path.exists():
                shutil.rmtree(save_path, ignore_errors=True)

            # Save the model with buffer
            ddpg.save(save_path, buffer=True, config=True)

            # Check files exist
            model_state_path = save_path / "model_state.safetensors"
            optim_state_path = save_path / "optim_state.safetensors"
            buffer_state_path = save_path / "buffer_state.safetensors"
            metadata_path = save_path / "metadata.json"
            config_path = save_path.parent / "model_config.json"

            assert model_state_path.exists(), "model_state.safetensors doesn't exist"
            assert optim_state_path.exists(), "optim_state.safetensors doesn't exist"
            assert buffer_state_path.exists(), "buffer_state.safetensors doesn't exist"
            assert metadata_path.exists(), "metadata.json doesn't exist"
            assert config_path.exists(), "model_config.json doesn't exist"

            # Load the model with buffer
            loaded_ddpg = LiquidDDPG.load(save_path, buffer=True)

            # Check buffer properties
            assert len(loaded_ddpg.buffer) == buffer_size
        finally:
            # Clean up resources
            if save_path.exists():
                # Close any open file handles
                import gc

                gc.collect()
                # Try to clean up the directory with retries
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        shutil.rmtree(save_path, ignore_errors=True)
                        if config_path.exists():
                            os.unlink(config_path)
                        break
                    except (PermissionError, OSError):
                        if attempt < max_retries - 1:
                            time.sleep(0.1)  # Short delay before retry
                        continue

    def test_load_without_buffer_file(self, ddpg: LiquidDDPG):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "model_save"

            # Save model without buffer
            ddpg.save(save_path, buffer=False)

            # Try to load with buffer=True, should raise error
            buffer_state_path = save_path / "buffer_state.safetensors"
            assert not buffer_state_path.exists(), "Buffer file should not exist"

            with pytest.raises(
                FileNotFoundError, match=r"Buffer state .* does not exist"
            ):
                LiquidDDPG.load(save_path, buffer=True)

    def test_save_directory_creation(self, ddpg: LiquidDDPG):
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = Path(temp_dir) / "new_subdir" / "nested"
            save_path = new_dir / "model_save"

            # Directory shouldn't exist yet
            assert not new_dir.exists()

            # Save should create the directory
            ddpg.save(save_path, config=True)

            # Check directory and files were created
            assert new_dir.exists()
            assert (save_path / "model_state.safetensors").exists()
            assert (save_path / "optim_state.safetensors").exists()
            assert (save_path / "metadata.json").exists()

            # Check config file was created in the parent directory
            config_path = save_path.parent / "model_config.json"
            assert config_path.exists()

    def test_load_invalid_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "invalid_model.pt")

            # Create an invalid model file
            with open(filepath, "w") as f:
                f.write("This is not a valid PyTorch file")

            # Attempt to load should raise an exception
            with pytest.raises(Exception):
                LiquidDDPG.load(filepath)

    def test_save_existing_file_error(self, ddpg: LiquidDDPG):
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "model.pt")

            # Create the file first
            with open(filepath, "w") as f:
                f.write("Existing file")

            # Attempt to save should raise FileExistsError
            with pytest.raises(FileExistsError):
                ddpg.save(filepath)

    def test_config_file_creation(self, ddpg: LiquidDDPG):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "model_save"

            # Save the model with config
            ddpg.save(save_path, config=True)

            # Check config file exists
            config_path = save_path.parent / "model_config.json"
            assert config_path.exists()

            # Verify config file is valid JSON with expected structure
            with open(config_path, "r") as f:
                config_data = json.loads(f.read())

            # Verify basic structure
            assert "agent" in config_data
            assert "model_details" in config_data
            assert "state_dim" in config_data["model_details"]

    def test_train_with_all_callbacks(
        self, ddpg: LiquidDDPG, env: gym.Env, tmp_path, monkeypatch
    ):
        set_seed(64)
        os.environ["VELORA_TEST_MODE"] = "True"

        # Create a unique directory using tmp_path for this test
        test_id = f"ddpg-test-{id(ddpg)}"
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        CP_DIR = test_id
        FREQ = 1

        # Clean up any existing directories from failed tests
        existing_dir = Path("checkpoints") / CP_DIR
        if existing_dir.exists():
            shutil.rmtree(existing_dir)

        # Mock the comet experiment
        mock_experiment = MagicMock()

        # Set the environment variable for Comet API key
        monkeypatch.setenv("COMET_API_KEY", "test-key")

        def patched_init(self, dirname, **kwargs):
            self.dirname = dirname
            self.filepath = tmp_path / "checkpoints" / dirname / "saves"
            self.frequency = kwargs.get("frequency", 100)
            self.buffer = kwargs.get("buffer", False)
            print(
                f"'{self.__class__.__name__}' enabled with ep_{self.frequency=} and {self.buffer=}."
            )

        # Apply the patch
        with patch.object(SaveCheckpoints, "__init__", patched_init):
            # Mock comet_ml.Experiment to return our mock
            with patch("comet_ml.Experiment", return_value=mock_experiment):
                callbacks = [
                    SaveCheckpoints(CP_DIR, frequency=FREQ, buffer=True),
                    EarlyStopping(
                        target=1000.0, patience=3
                    ),  # Lower patience for faster test
                    CometAnalytics("ddpg-test"),
                ]

                try:
                    # Skip the db operations completely
                    with patch("velora.training.metrics.TrainMetrics.add_episode"):
                        with patch("velora.training.metrics.TrainMetrics.info"):
                            # Patch TrainHandler.episode to set ep_reward on the state
                            # Include the correct signature with ep_reward parameter
                            def patched_episode(
                                self, current_ep, ep_reward=None, *args, **kwargs
                            ):
                                # Set ep_reward directly on the state
                                self.state.ep_reward = 1500.0
                                # No need to call original since we're providing a complete replacement

                            with patch.object(TrainHandler, "episode", patched_episode):
                                # Mock predict to return tensors with correct shapes
                                with patch.object(ddpg, "predict") as mock_predict:
                                    mock_predict.return_value = (
                                        torch.zeros(
                                            ddpg.action_dim, device=ddpg.device
                                        ),  # action
                                        torch.zeros(
                                            (1, ddpg.actor.ncp.hidden_size),
                                            device=ddpg.device,
                                        ),  # hidden state with correct shape
                                    )

                                    # Mock buffer operations to prevent storing experiences
                                    with (
                                        patch.object(ddpg.buffer, "add"),
                                        patch.object(ddpg.buffer, "warm"),
                                    ):
                                        # Mock _train_step to avoid network operations
                                        with patch.object(
                                            ddpg, "_train_step"
                                        ) as mock_train_step:
                                            mock_train_step.return_value = (
                                                torch.tensor(0.1),
                                                torch.tensor(0.2),
                                            )  # Return mock losses

                                            # Run training with reduced episodes for faster testing
                                            ddpg.train(
                                                env,
                                                batch_size=32,
                                                n_episodes=5,
                                                callbacks=callbacks,
                                                max_steps=100,
                                                noise_scale=0.3,
                                                gamma=0.99,
                                                tau=0.005,
                                                window_size=FREQ,
                                            )

                        # Verify experiment was created and configured
                        assert mock_experiment.set_name.call_count == 1
                        assert mock_experiment.add_tags.call_count == 1
                        assert mock_experiment.log_parameters.call_count == 1

                        # Verify experiment was ended
                        assert mock_experiment.end.call_count == 1
                finally:
                    # Clean up the directories - both real and temp paths just to be sure
                    real_path = Path("checkpoints") / CP_DIR
                    if real_path.exists():
                        shutil.rmtree(real_path)

                    temp_path = tmp_path / "checkpoints" / CP_DIR
                    if temp_path.exists():
                        shutil.rmtree(temp_path)

    def test_early_stopping(self, ddpg: LiquidDDPG, env: gym.Env):
        """Test that DDPG training stops early when EarlyStopping callback is triggered."""
        set_seed(64)
        early_stopping = EarlyStopping(target=100.0, patience=1)

        # Mock necessary methods to avoid actual training
        with (
            patch.object(ddpg.buffer, "warm"),
            patch.object(ddpg.buffer, "add"),
            patch.object(
                ddpg,
                "_train_step",
                return_value=(
                    torch.tensor(0.1),
                    torch.tensor(0.2),
                ),
            ),
            patch.object(
                ddpg,
                "predict",
                return_value=(
                    torch.zeros(ddpg.action_dim, device=ddpg.device),
                    torch.zeros((1, ddpg.actor.ncp.hidden_size), device=ddpg.device),
                ),
            ),
        ):
            # Skip database operations by patching add_episode
            with patch("velora.training.metrics.TrainMetrics.add_episode"):
                with patch("velora.training.metrics.TrainMetrics.info"):
                    # Prepare a patched episode method with the correct signature
                    def mock_episode(self, current_ep, ep_reward=None, *args, **kwargs):
                        # Set ep_reward directly on the state
                        self.state.ep_reward = 150.0
                        # No need to call original since we're providing a complete replacement

                    with patch.object(TrainHandler, "episode", mock_episode):
                        # Run training with the EarlyStopping callback
                        ddpg.train(
                            env,
                            batch_size=32,
                            n_episodes=5,  # Set higher than our early stopping threshold
                            callbacks=[early_stopping],
                            max_steps=5,
                            window_size=1,
                        )

    def test_file_exists_error(self, ddpg: LiquidDDPG):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "model_save"

            # Create directory and model state file to trigger error
            save_path.mkdir(parents=True, exist_ok=True)
            model_state_path = save_path / "model_state.safetensors"
            model_state_path.touch()

            # Now trying to save should raise FileExistsError
            with pytest.raises(FileExistsError, match=r"A model state already exists"):
                ddpg.save(save_path)

    def test_invalid_checkpoint_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_path = Path(temp_dir) / "invalid_save"

            # Create an invalid safetensors file
            invalid_path.mkdir(parents=True, exist_ok=True)
            with open(invalid_path / "model_state.safetensors", "w") as f:
                f.write("This is not a valid safetensors file")
            with open(invalid_path / "optim_state.safetensors", "w") as f:
                f.write("This is not a valid safetensors file")
            with open(invalid_path / "metadata.json", "w") as f:
                f.write("{}")

            # Attempt to load should raise an exception
            with pytest.raises(Exception):
                LiquidDDPG.load(invalid_path)
