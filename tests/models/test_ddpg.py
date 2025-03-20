from pathlib import Path
import shutil
from typing import Any, Dict, Literal
import pytest
from unittest.mock import MagicMock, Mock, patch
import os
import tempfile
import json

import torch
import gymnasium as gym

from velora.buffer.experience import Experience
from velora.callbacks import CometAnalytics, EarlyStopping, SaveCheckpoints
from velora.metrics.models import Episode
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
            action = torch.tensor([1.0])
            reward = 2.0  # Single float value
            next_state = torch.zeros(ddpg.state_dim)
            done = False

            # Create Experience object explicitly
            exp = Experience(state, action, reward, next_state, done)
            ddpg.buffer.add(exp)

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
            ddpg.buffer.add(exp)

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
            exp = Experience(state, action, reward, next_state, done)
            ddpg.buffer.add(exp)

        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "model.pt")
            # Save the model with buffer
            ddpg.save(filepath, buffer=True)

            # Check both files exist
            assert os.path.exists(filepath)
            buffer_path = ddpg.buffer.create_filepath(filepath)
            assert os.path.exists(buffer_path)
            config_path = os.path.join(os.path.dirname(filepath), "model_config.json")
            assert os.path.exists(config_path)

            # Load the model with buffer
            loaded_ddpg = LiquidDDPG.load(filepath, buffer=True)

            # Check buffer properties
            assert len(loaded_ddpg.buffer) == len(ddpg.buffer)

    def test_load_without_buffer_file(self, ddpg: LiquidDDPG):
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "model.pt")

            # Save model without buffer
            ddpg.save(filepath, buffer=False)

            # Try to load with buffer=True, should raise error
            with pytest.raises(
                FileNotFoundError, match="Buffer file .* does not exist"
            ):
                LiquidDDPG.load(filepath, buffer=True)

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

            # Check config file was created
            config_path = os.path.join(new_dir, "model_config.json")
            assert os.path.exists(config_path)

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
            filepath = os.path.join(temp_dir, "model.pt")

            # Save the model
            ddpg.save(filepath)

            # Check config file exists
            config_path = os.path.join(os.path.dirname(filepath), "model_config.json")
            assert os.path.exists(config_path)

            # Verify config file is valid JSON with expected structure
            with open(config_path, "r") as f:
                config_data = json.loads(f.read())

            # Verify basic structure
            assert "agent" in config_data
            assert "state_dim" in config_data["model_details"]

    def test_checkpoint_keys_validation(self, ddpg: LiquidDDPG):
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "model.pt")

            # Save the model
            ddpg.save(filepath)

            # Load the checkpoint and modify it
            checkpoint = torch.load(filepath)

            # Add an invalid key
            checkpoint["invalid_key"] = "some_value"

            # Save the modified checkpoint
            modified_filepath = os.path.join(temp_dir, "modified_model.pt")
            torch.save(checkpoint, modified_filepath)

            # Attempt to load should raise ValueError about key mismatch
            with pytest.raises(ValueError, match="Mismatch between checkpoint keys"):
                LiquidDDPG.load(modified_filepath)

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
                    # Mock early stopping to trigger after a few episodes
                    with patch(
                        "velora.metrics.db.get_current_episode"
                    ) as mock_get_episode:
                        # Create a mock episode with high reward to trigger early stopping
                        mock_episode = MagicMock()
                        mock_episode.reward = (
                            1500.0  # High enough to trigger early stopping
                        )
                        mock_get_episode.return_value = [mock_episode]

                        # Mock predict to return tensors with correct shapes
                        with patch.object(ddpg, "predict") as mock_predict:
                            mock_predict.return_value = (
                                torch.zeros(
                                    ddpg.action_dim, device=ddpg.device
                                ),  # action
                                torch.zeros(
                                    (1, ddpg.n_neurons + ddpg.action_dim),
                                    device=ddpg.device,
                                ),  # hidden state with correct shape
                            )

                            # Mock buffer.push to prevent storing experiences
                            with (
                                patch.object(ddpg.buffer, "add"),
                                patch.object(ddpg.buffer, "warm"),
                            ):
                                # Mock _train_step to avoid network operations
                                with patch.object(
                                    ddpg, "_train_step"
                                ) as mock_train_step:
                                    mock_train_step.return_value = (
                                        0.1,
                                        0.2,
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

                        # Verify metrics were logged
                        assert mock_experiment.log_metrics.call_count > 0

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
            patch.object(ddpg, "_train_step", return_value=(0.1, 0.2)),
            patch.object(
                ddpg,
                "predict",
                return_value=(
                    torch.zeros(ddpg.action_dim, device=ddpg.device),
                    torch.zeros(
                        (1, ddpg.n_neurons + ddpg.action_dim), device=ddpg.device
                    ),
                ),
            ),
        ):
            # Prepare TrainHandler.episode to track episodes and add database entries
            original_episode = TrainHandler.episode

            def mock_episode(self, current_ep):
                # Add a database entry for this episode
                test_episode = Episode(
                    experiment_id=self.state.experiment_id,
                    episode_num=current_ep,
                    reward=150.0,  # Higher than our target of 100.0
                    length=100,
                    reward_moving_avg=150.0,
                    reward_moving_std=0.0,
                    actor_loss=0.1,
                    critic_loss=0.1,
                )
                self.state.session.add(test_episode)
                self.state.session.commit()

                # Call original episode method
                original_episode(self, current_ep)

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
