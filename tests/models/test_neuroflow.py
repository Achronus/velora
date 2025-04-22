from pathlib import Path
import shutil
import time
from typing import Any, Dict, Literal
import pytest
from unittest.mock import MagicMock, patch
import os
import tempfile

import torch

from velora.callbacks import CometAnalytics, EarlyStopping, SaveCheckpoints
from velora.models.nf.agent import NeuroFlowCT, NeuroFlow
from velora.models.nf.modules import (
    ActorModule,
    ActorModuleDiscrete,
    CriticModule,
    CriticModuleDiscrete,
    EntropyModule,
    EntropyModuleDiscrete,
)
from velora.buffer.replay import ReplayBuffer
from velora.training.handler import TrainHandler


NetworkParamsType = Dict[
    Literal["state_dim", "n_neurons", "action_dim", "device"],
    Any,
]
NeuroFlowParamsType = Dict[
    Literal[
        "env_id",
        "actor_neurons",
        "critic_neurons",
        "buffer_size",
        "actor_lr",
        "critic_lr",
        "alpha_lr",
        "initial_alpha",
        "log_std",
        "tau",
        "gamma",
        "device",
    ],
    Any,
]


class TestNeuroFlow:
    @pytest.fixture
    def neuroflow_params(self) -> NeuroFlowParamsType:
        return {
            "env_id": "InvertedPendulum-v5",
            "actor_neurons": 16,
            "critic_neurons": 32,
            "buffer_size": 1000,
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "alpha_lr": 3e-4,
            "initial_alpha": 1.0,
            "log_std": (-5, 2),
            "tau": 0.005,
            "gamma": 0.99,
            "device": torch.device("cpu"),
            "seed": 64,
        }

    @pytest.fixture
    def neuroflow(self, neuroflow_params: NeuroFlowParamsType) -> NeuroFlowCT:
        os.environ["VELORA_TEST_MODE"] = "True"
        return NeuroFlowCT(**neuroflow_params)

    def test_init(self, neuroflow: NeuroFlowCT, neuroflow_params: NeuroFlowParamsType):
        assert isinstance(neuroflow.actor, ActorModule)
        assert isinstance(neuroflow.critic, CriticModule)
        assert isinstance(neuroflow.entropy, EntropyModule)
        assert isinstance(neuroflow.buffer, ReplayBuffer)

        assert neuroflow.initial_alpha == neuroflow_params["initial_alpha"]
        assert neuroflow.log_std == neuroflow_params["log_std"]
        assert neuroflow.gamma == neuroflow_params["gamma"]
        assert neuroflow.tau == neuroflow_params["tau"]

        assert neuroflow.state_dim > 0
        assert neuroflow.action_dim > 0
        assert neuroflow.hidden_dim > 0

        assert hasattr(neuroflow, "config")
        assert hasattr(neuroflow, "metadata")

    def test_invalid_env(self, neuroflow_params: NeuroFlowParamsType):
        with pytest.raises(EnvironmentError):
            neuroflow_params["env_id"] = "CartPole-v1"
            NeuroFlowCT(**neuroflow_params)

    def test_predict(self, neuroflow: NeuroFlowCT):
        state = torch.randn(neuroflow.state_dim)

        # Test deterministic prediction (train_mode=False)
        action, hidden = neuroflow.predict(state, None, train_mode=False)

        assert action.shape == (neuroflow.action_dim,)
        assert hidden.shape[1] == neuroflow.actor.hidden_size

        # Test stochastic prediction (train_mode=True)
        action, hidden = neuroflow.predict(state, None, train_mode=True)

        assert action.shape == (neuroflow.action_dim,)
        assert hidden.shape[1] == neuroflow.actor.hidden_size

    def test_update_critics(self, neuroflow: NeuroFlowCT):
        # Create a mock BatchExperience
        batch = MagicMock()
        batch.states = torch.randn(10, neuroflow.state_dim)
        batch.actions = torch.randn(10, neuroflow.action_dim)
        batch.rewards = torch.randn(10, 1)
        batch.next_states = torch.randn(10, neuroflow.state_dim)
        batch.dones = torch.zeros(10, 1)
        batch.hiddens = torch.randn(10, neuroflow.hidden_dim)

        # Test _update_critics method
        with patch.object(neuroflow.actor, "forward") as mock_forward:
            # Mock the forward method to return compatible tensors
            mock_forward.return_value = (
                torch.randn(10, neuroflow.action_dim),
                torch.randn(10, 1),
                torch.randn(10, neuroflow.hidden_dim),
            )

            critic_loss = neuroflow._update_critics(batch)

            assert isinstance(critic_loss, torch.Tensor)
            assert critic_loss.shape == torch.Size([])  # Should be a scalar

    def test_train_step(self, neuroflow: NeuroFlowCT):
        # Fill buffer with valid experiences for sampling
        for i in range(32):
            state = torch.zeros(neuroflow.state_dim)
            action = torch.zeros(neuroflow.action_dim)
            reward = 0.0
            next_state = torch.zeros(neuroflow.state_dim)
            done = False
            hidden = torch.zeros(neuroflow.hidden_dim)

            neuroflow.buffer.add(state, action, reward, next_state, done, hidden)

        # Test _train_step method with mocked components
        with (
            patch.object(neuroflow, "_update_critics", return_value=torch.tensor(0.5)),
            patch.object(neuroflow.actor, "forward") as mock_forward,
            patch.object(neuroflow.critic, "predict") as mock_predict,
            patch.object(
                neuroflow.entropy, "compute_loss", return_value=torch.tensor(0.3)
            ),
            patch.object(neuroflow.actor, "gradient_step"),
            patch.object(neuroflow.entropy, "gradient_step"),
            patch.object(neuroflow.critic, "update_targets"),
        ):
            # Configure mocks
            mock_forward.return_value = (
                torch.randn(32, neuroflow.action_dim),
                torch.randn(32, 1),
                torch.randn(32, neuroflow.hidden_dim),
            )
            mock_predict.return_value = (torch.randn(32, 1), torch.randn(32, 1))

            # Execute train step
            losses = neuroflow._train_step(32)

            # Verify results
            assert isinstance(losses, dict)
            assert "critic" in losses
            assert "actor" in losses
            assert "entropy" in losses

            # Check shapes
            assert losses["critic"].shape == torch.Size([])
            assert losses["actor"].shape == torch.Size([])
            assert losses["entropy"].shape == torch.Size([])

    def test_save_load_with_buffer(self, neuroflow: NeuroFlowCT):
        # Add some experiences to buffer
        for i in range(10):
            state = torch.zeros(neuroflow.state_dim)
            action = torch.zeros(neuroflow.action_dim)
            reward = float(i * 0.5)
            next_state = torch.ones(neuroflow.state_dim)
            done = i == 9
            hidden = torch.zeros(neuroflow.hidden_dim)

            neuroflow.buffer.add(state, action, reward, next_state, done, hidden)

        # Get the initial buffer size to compare later
        buffer_size = len(neuroflow.buffer)

        # Create a unique temporary directory name
        unique_id = f"neuroflow_save_{id(neuroflow)}"
        temp_dir = tempfile.gettempdir()
        save_path = Path(temp_dir) / unique_id

        try:
            # Ensure the directory doesn't exist before starting
            if save_path.exists():
                shutil.rmtree(save_path, ignore_errors=True)

            # Save the model with buffer
            neuroflow.save(save_path, buffer=True, config=True)

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
            loaded_neuroflow = NeuroFlowCT.load(save_path, buffer=True)

            # Check buffer properties
            assert len(loaded_neuroflow.buffer) == buffer_size

            # Verify key attributes match
            assert loaded_neuroflow.state_dim == neuroflow.state_dim
            assert loaded_neuroflow.action_dim == neuroflow.action_dim
            assert loaded_neuroflow.hidden_dim == neuroflow.hidden_dim
            assert loaded_neuroflow.gamma == neuroflow.gamma
            assert loaded_neuroflow.tau == neuroflow.tau

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

    def test_load_without_buffer_file(self, neuroflow: NeuroFlowCT):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "model_save"

            # Save model without buffer
            neuroflow.save(save_path, buffer=False)

            # Try to load with buffer=True, should raise error
            buffer_state_path = save_path / "buffer_state.safetensors"
            assert not buffer_state_path.exists(), "Buffer file should not exist"

            with pytest.raises(
                FileNotFoundError, match=r"Buffer state .* does not exist"
            ):
                NeuroFlowCT.load(save_path, buffer=True)

    def test_save_directory_creation(self, neuroflow: NeuroFlowCT):
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = Path(temp_dir) / "new_subdir" / "nested"
            save_path = new_dir / "model_save"

            # Directory shouldn't exist yet
            assert not new_dir.exists()

            # Save should create the directory
            neuroflow.save(save_path, config=True)

            # Check directory and files were created
            assert new_dir.exists()
            assert (save_path / "model_state.safetensors").exists()
            assert (save_path / "optim_state.safetensors").exists()
            assert (save_path / "metadata.json").exists()

            # Check config file was created in the parent directory
            config_path = save_path.parent / "model_config.json"
            assert config_path.exists()

    def test_file_exists_error(self, neuroflow: NeuroFlowCT):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "model_save"

            # Create directory and model state file to trigger error
            save_path.mkdir(parents=True, exist_ok=True)
            model_state_path = save_path / "model_state.safetensors"
            model_state_path.touch()

            # Now trying to save should raise FileExistsError
            with pytest.raises(FileExistsError, match=r"A model state already exists"):
                neuroflow.save(save_path)

    def test_train_with_callbacks(self, neuroflow: NeuroFlowCT, tmp_path, monkeypatch):
        os.environ["VELORA_TEST_MODE"] = "True"

        # Create a unique directory using tmp_path for this test
        test_id = f"neuroflow-test-{id(neuroflow)}"
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
                        target=1000.0, patience=1
                    ),  # Low patience for faster test
                    CometAnalytics("neuroflow-test"),
                ]

                try:
                    # Skip the db operations completely
                    with patch("velora.training.metrics.TrainMetrics.add_episode"):
                        with patch("velora.training.metrics.TrainMetrics.info"):
                            # Patch TrainHandler.episode to set ep_reward on the state
                            def patched_episode(
                                self, current_ep, ep_reward=None, *args, **kwargs
                            ):
                                # Set ep_reward directly on the state
                                self.state.ep_reward = 1500.0

                            with patch.object(TrainHandler, "episode", patched_episode):
                                # Mock predict to return tensors with correct shapes
                                with patch.object(neuroflow, "predict") as mock_predict:
                                    mock_predict.return_value = (
                                        torch.zeros(
                                            neuroflow.action_dim,
                                            device=neuroflow.device,
                                        ),  # action
                                        torch.zeros(
                                            (1, neuroflow.actor.hidden_size),
                                            device=neuroflow.device,
                                        ),  # hidden
                                    )

                                    # Mock buffer operations to prevent storing experiences
                                    with (
                                        patch.object(neuroflow.buffer, "add"),
                                        patch.object(neuroflow.buffer, "warm"),
                                    ):
                                        # Mock _train_step to avoid network operations
                                        with patch.object(
                                            neuroflow, "_train_step"
                                        ) as mock_train_step:
                                            mock_train_step.return_value = {
                                                "critic": torch.tensor(0.1),
                                                "actor": torch.tensor(0.2),
                                                "entropy": torch.tensor(0.3),
                                            }

                                            # Run training with reduced episodes for faster testing
                                            neuroflow.train(
                                                batch_size=32,
                                                n_episodes=3,
                                                callbacks=callbacks,
                                                max_steps=10,
                                                window_size=FREQ,
                                                warmup_steps=32 * 2,
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

    def test_early_stopping(self, neuroflow: NeuroFlowCT):
        early_stopping = EarlyStopping(target=100.0, patience=2)

        # Mock necessary methods to avoid actual training
        with (
            patch.object(neuroflow.buffer, "warm"),
            patch.object(neuroflow.buffer, "add"),
            patch.object(
                neuroflow,
                "_train_step",
                return_value={
                    "critic": torch.tensor(0.1),
                    "actor": torch.tensor(0.2),
                    "entropy": torch.tensor(0.3),
                },
            ),
            patch.object(
                neuroflow,
                "predict",
                return_value=(
                    torch.zeros(neuroflow.action_dim, device=neuroflow.device),
                    torch.zeros(
                        (1, neuroflow.actor.hidden_size), device=neuroflow.device
                    ),
                ),
            ),
        ):
            # Skip database operations by patching add_episode
            with patch("velora.training.metrics.TrainMetrics.add_episode"):
                with patch("velora.training.metrics.TrainMetrics.info"):
                    # Prepare a patched episode method
                    def mock_episode(self, current_ep, ep_reward=None, *args, **kwargs):
                        # Set ep_reward directly on the state
                        self.state.ep_reward = 150.0

                    with patch.object(TrainHandler, "episode", mock_episode):
                        # Run training with the EarlyStopping callback
                        neuroflow.train(
                            batch_size=32,
                            n_episodes=5,  # Set higher than our early stopping threshold
                            callbacks=[early_stopping],
                            max_steps=5,
                            window_size=1,
                            warmup_steps=32 * 2,
                        )

    def test_state_dict(self, neuroflow: NeuroFlowCT):
        """Test that state_dict method returns the expected structure."""
        state_dict = neuroflow.state_dict()

        # Check main sections
        assert "modules" in state_dict
        assert "optimizers" in state_dict

        # Check key modules are included
        assert (
            "actor" in state_dict["modules"] or "actor.network" in state_dict["modules"]
        )
        assert (
            "critic" in state_dict["modules"]
            or "critic.network1" in state_dict["modules"]
        )

        # Check optimizers
        assert any("actor_optim" in key for key in state_dict["optimizers"])
        assert any("critic_optim" in key for key in state_dict["optimizers"])

    def test_train_cycle(self, neuroflow: NeuroFlowCT):
        neuroflow.train(5, n_episodes=2, warmup_steps=10)


class TestNeuroFlowDiscrete:
    @pytest.fixture
    def neuroflow_params(self) -> NeuroFlowParamsType:
        return {
            "env_id": "CartPole-v1",
            "actor_neurons": 16,
            "critic_neurons": 32,
            "buffer_size": 1000,
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "alpha_lr": 3e-4,
            "initial_alpha": 1.0,
            "tau": 0.005,
            "gamma": 0.99,
            "device": torch.device("cpu"),
            "seed": 64,
        }

    @pytest.fixture
    def neuroflow(self, neuroflow_params: NeuroFlowParamsType) -> NeuroFlow:
        os.environ["VELORA_TEST_MODE"] = "True"
        return NeuroFlow(**neuroflow_params)

    def test_init(self, neuroflow: NeuroFlow, neuroflow_params: NeuroFlowParamsType):
        assert isinstance(neuroflow.actor, ActorModuleDiscrete)
        assert isinstance(neuroflow.critic, CriticModuleDiscrete)
        assert isinstance(neuroflow.entropy, EntropyModuleDiscrete)
        assert isinstance(neuroflow.buffer, ReplayBuffer)

        assert neuroflow.initial_alpha == neuroflow_params["initial_alpha"]
        assert neuroflow.gamma == neuroflow_params["gamma"]
        assert neuroflow.tau == neuroflow_params["tau"]

        assert neuroflow.state_dim > 0
        assert neuroflow.action_dim > 0
        assert neuroflow.hidden_dim > 0

        assert hasattr(neuroflow, "config")
        assert hasattr(neuroflow, "metadata")

    def test_invalid_env(self, neuroflow_params: NeuroFlowParamsType):
        with pytest.raises(EnvironmentError):
            neuroflow_params["env_id"] = "InvertedPendulum-v5"
            NeuroFlow(**neuroflow_params)

    def test_predict(self, neuroflow: NeuroFlow):
        state = torch.randn(neuroflow.state_dim)

        # Test deterministic prediction (train_mode=False)
        action, hidden = neuroflow.predict(state, None, train_mode=False)

        assert action.shape == torch.Size([])
        assert hidden.shape[1] == neuroflow.actor.hidden_size

        # Test stochastic prediction (train_mode=True)
        action, hidden = neuroflow.predict(state, None, train_mode=True)

        assert action.shape == torch.Size([])
        assert hidden.shape[1] == neuroflow.actor.hidden_size

    def test_update_critics(self, neuroflow: NeuroFlow):
        # Create a mock BatchExperience
        batch = MagicMock()
        batch.states = torch.randn(10, neuroflow.state_dim)
        batch.actions = torch.randint(0, neuroflow.action_dim, (10, 1))
        batch.rewards = torch.randn(10, 1)
        batch.next_states = torch.randn(10, neuroflow.state_dim)
        batch.dones = torch.zeros(10, 1)
        batch.hiddens = torch.randn(10, neuroflow.hidden_dim)

        # Test _update_critics method
        with patch.object(neuroflow.actor, "forward") as mock_forward:
            # Mock the forward method to return compatible tensors
            mock_forward.return_value = (
                torch.randint(0, neuroflow.action_dim, (10,)),
                torch.randn(10, neuroflow.action_dim),
                torch.randn(10, 1),
                torch.randn(10, neuroflow.hidden_dim),
            )

            critic_loss = neuroflow._update_critics(batch)

            assert isinstance(critic_loss, torch.Tensor)
            assert critic_loss.shape == torch.Size([])  # Should be a scalar

    def test_train_step(self, neuroflow: NeuroFlow):
        # Fill buffer with valid experiences for sampling
        for i in range(32):
            state = torch.zeros(neuroflow.state_dim)
            action = torch.zeros(1)
            reward = 0.0
            next_state = torch.zeros(neuroflow.state_dim)
            done = False
            hidden = torch.zeros(neuroflow.hidden_dim)

            neuroflow.buffer.add(state, action, reward, next_state, done, hidden)

        # Test _train_step method with mocked components
        with (
            patch.object(neuroflow, "_update_critics", return_value=torch.tensor(0.5)),
            patch.object(neuroflow.actor, "forward") as mock_forward,
            patch.object(neuroflow.critic, "predict") as mock_predict,
            patch.object(
                neuroflow.entropy, "compute_loss", return_value=torch.tensor(0.3)
            ),
            patch.object(neuroflow.actor, "gradient_step"),
            patch.object(neuroflow.entropy, "gradient_step"),
            patch.object(neuroflow.critic, "update_targets"),
        ):
            # Configure mocks
            mock_forward.return_value = (
                torch.randn(32, neuroflow.action_dim),
                torch.randn(32, neuroflow.action_dim),
                torch.randn(32, 1),
                torch.randn(32, neuroflow.hidden_dim),
            )
            mock_predict.return_value = (torch.randn(32, 1), torch.randn(32, 1))

            # Execute train step
            losses = neuroflow._train_step(32)

            # Verify results
            assert isinstance(losses, dict)
            assert "critic" in losses
            assert "actor" in losses
            assert "entropy" in losses

            # Check shapes
            assert losses["critic"].shape == torch.Size([])
            assert losses["actor"].shape == torch.Size([])
            assert losses["entropy"].shape == torch.Size([])

    def test_save_load_with_buffer(self, neuroflow: NeuroFlow):
        # Add some experiences to buffer
        for i in range(10):
            state = torch.zeros(neuroflow.state_dim)
            action = torch.zeros(1)
            reward = float(i * 0.5)
            next_state = torch.ones(neuroflow.state_dim)
            done = i == 9
            hidden = torch.zeros(neuroflow.hidden_dim)

            neuroflow.buffer.add(state, action, reward, next_state, done, hidden)

        # Get the initial buffer size to compare later
        buffer_size = len(neuroflow.buffer)

        # Create a unique temporary directory name
        unique_id = f"neuroflow_save_{id(neuroflow)}"
        temp_dir = tempfile.gettempdir()
        save_path = Path(temp_dir) / unique_id

        try:
            # Ensure the directory doesn't exist before starting
            if save_path.exists():
                shutil.rmtree(save_path, ignore_errors=True)

            # Save the model with buffer
            neuroflow.save(save_path, buffer=True, config=True)

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
            loaded_neuroflow = NeuroFlow.load(save_path, buffer=True)

            # Check buffer properties
            assert len(loaded_neuroflow.buffer) == buffer_size

            # Verify key attributes match
            assert loaded_neuroflow.state_dim == neuroflow.state_dim
            assert loaded_neuroflow.action_dim == neuroflow.action_dim
            assert loaded_neuroflow.hidden_dim == neuroflow.hidden_dim
            assert loaded_neuroflow.gamma == neuroflow.gamma
            assert loaded_neuroflow.tau == neuroflow.tau

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

    def test_load_without_buffer_file(self, neuroflow: NeuroFlow):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "model_save"

            # Save model without buffer
            neuroflow.save(save_path, buffer=False)

            # Try to load with buffer=True, should raise error
            buffer_state_path = save_path / "buffer_state.safetensors"
            assert not buffer_state_path.exists(), "Buffer file should not exist"

            with pytest.raises(
                FileNotFoundError, match=r"Buffer state .* does not exist"
            ):
                NeuroFlow.load(save_path, buffer=True)

    def test_save_directory_creation(self, neuroflow: NeuroFlow):
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = Path(temp_dir) / "new_subdir" / "nested"
            save_path = new_dir / "model_save"

            # Directory shouldn't exist yet
            assert not new_dir.exists()

            # Save should create the directory
            neuroflow.save(save_path, config=True)

            # Check directory and files were created
            assert new_dir.exists()
            assert (save_path / "model_state.safetensors").exists()
            assert (save_path / "optim_state.safetensors").exists()
            assert (save_path / "metadata.json").exists()

            # Check config file was created in the parent directory
            config_path = save_path.parent / "model_config.json"
            assert config_path.exists()

    def test_file_exists_error(self, neuroflow: NeuroFlow):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "model_save"

            # Create directory and model state file to trigger error
            save_path.mkdir(parents=True, exist_ok=True)
            model_state_path = save_path / "model_state.safetensors"
            model_state_path.touch()

            # Now trying to save should raise FileExistsError
            with pytest.raises(FileExistsError, match=r"A model state already exists"):
                neuroflow.save(save_path)

    def test_train_with_callbacks(self, neuroflow: NeuroFlow, tmp_path, monkeypatch):
        os.environ["VELORA_TEST_MODE"] = "True"

        # Create a unique directory using tmp_path for this test
        test_id = f"neuroflow-test-{id(neuroflow)}"
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
                        target=1000.0, patience=1
                    ),  # Low patience for faster test
                    CometAnalytics("neuroflow-test"),
                ]

                try:
                    # Skip the db operations completely
                    with patch("velora.training.metrics.TrainMetrics.add_episode"):
                        with patch("velora.training.metrics.TrainMetrics.info"):
                            # Patch TrainHandler.episode to set ep_reward on the state
                            def patched_episode(
                                self, current_ep, ep_reward=None, *args, **kwargs
                            ):
                                # Set ep_reward directly on the state
                                self.state.ep_reward = 1500.0

                            with patch.object(TrainHandler, "episode", patched_episode):
                                # Mock predict to return tensors with correct shapes
                                with patch.object(neuroflow, "predict") as mock_predict:
                                    mock_predict.return_value = (
                                        torch.tensor(
                                            0,
                                            device=neuroflow.device,
                                        ),  # action
                                        torch.zeros(
                                            (1, neuroflow.actor.hidden_size),
                                            device=neuroflow.device,
                                        ),  # hidden
                                    )

                                    # Mock buffer operations to prevent storing experiences
                                    with (
                                        patch.object(neuroflow.buffer, "add"),
                                        patch.object(neuroflow.buffer, "warm"),
                                    ):
                                        # Mock _train_step to avoid network operations
                                        with patch.object(
                                            neuroflow, "_train_step"
                                        ) as mock_train_step:
                                            mock_train_step.return_value = {
                                                "critic": torch.tensor(0.1),
                                                "actor": torch.tensor(0.2),
                                                "entropy": torch.tensor(0.3),
                                            }

                                            # Run training with reduced episodes for faster testing
                                            neuroflow.train(
                                                batch_size=32,
                                                n_episodes=3,
                                                callbacks=callbacks,
                                                max_steps=10,
                                                window_size=FREQ,
                                                warmup_steps=32 * 2,
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

    def test_early_stopping(self, neuroflow: NeuroFlow):
        early_stopping = EarlyStopping(target=100.0, patience=2)

        # Mock necessary methods to avoid actual training
        with (
            patch.object(neuroflow.buffer, "warm"),
            patch.object(neuroflow.buffer, "add"),
            patch.object(
                neuroflow,
                "_train_step",
                return_value={
                    "critic": torch.tensor(0.1),
                    "actor": torch.tensor(0.2),
                    "entropy": torch.tensor(0.3),
                },
            ),
            patch.object(
                neuroflow,
                "predict",
                return_value=(
                    torch.tensor(0, device=neuroflow.device),
                    torch.zeros(
                        (1, neuroflow.actor.hidden_size), device=neuroflow.device
                    ),
                ),
            ),
        ):
            # Skip database operations by patching add_episode
            with patch("velora.training.metrics.TrainMetrics.add_episode"):
                with patch("velora.training.metrics.TrainMetrics.info"):
                    # Prepare a patched episode method
                    def mock_episode(self, current_ep, ep_reward=None, *args, **kwargs):
                        # Set ep_reward directly on the state
                        self.state.ep_reward = 150.0

                    with patch.object(TrainHandler, "episode", mock_episode):
                        # Run training with the EarlyStopping callback
                        neuroflow.train(
                            batch_size=32,
                            n_episodes=5,  # Set higher than our early stopping threshold
                            callbacks=[early_stopping],
                            max_steps=5,
                            window_size=1,
                            warmup_steps=32 * 2,
                        )

    def test_state_dict(self, neuroflow: NeuroFlow):
        """Test that state_dict method returns the expected structure."""
        state_dict = neuroflow.state_dict()

        # Check main sections
        assert "modules" in state_dict
        assert "optimizers" in state_dict

        # Check key modules are included
        assert (
            "actor" in state_dict["modules"] or "actor.network" in state_dict["modules"]
        )
        assert (
            "critic" in state_dict["modules"]
            or "critic.network1" in state_dict["modules"]
        )

        # Check optimizers
        assert any("actor_optim" in key for key in state_dict["optimizers"])
        assert any("critic_optim" in key for key in state_dict["optimizers"])

    def test_train_cycle(self, neuroflow: NeuroFlow):
        neuroflow.train(5, n_episodes=2, warmup_steps=10)
