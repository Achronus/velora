from pathlib import Path
import tempfile
import pytest
import torch

from velora.models.nf.agent import NeuroFlow
from velora.utils.restore import (
    load_model,
    optim_to_tensor,
    optim_from_tensor,
    save_model,
)


def test_optim_to_tensor():
    """Test converting optimizer state dictionary to tensor dictionary."""
    import torch.optim as optim

    # Create a simple model and optimizer
    model = torch.nn.Linear(10, 5)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Generate some gradients to populate optimizer state
    dummy_input = torch.randn(2, 10)
    output = model(dummy_input)
    loss = output.sum()
    loss.backward()
    optimizer.step()

    # Get optimizer state dict
    optim_state = optimizer.state_dict()

    # Convert to tensor dict
    tensor_dict = optim_to_tensor("test_optim", optim_state)

    # Verify conversion
    assert isinstance(tensor_dict, dict)
    assert len(tensor_dict) > 0

    # Check that all values are tensors
    for key, value in tensor_dict.items():
        assert isinstance(key, str)
        assert isinstance(value, torch.Tensor)
        assert key.startswith("test_optim.")


def test_optim_from_tensor():
    """Test converting tensor dictionary back to optimizer state."""
    import torch.optim as optim

    # Create a simple model and optimizer
    model = torch.nn.Linear(10, 5)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Generate some gradients
    dummy_input = torch.randn(2, 10)
    output = model(dummy_input)
    loss = output.sum()
    loss.backward()
    optimizer.step()

    # Get optimizer state dict and convert to tensor dict
    original_state = optimizer.state_dict()
    tensor_dict = optim_to_tensor("actor_optim", original_state)

    # Convert back to optimizer state format
    reconstructed_state = optim_from_tensor(tensor_dict)

    # Verify reconstruction
    assert "actor_optim" in reconstructed_state
    assert isinstance(reconstructed_state["actor_optim"], dict)

    # Test with a second optimizer to verify both keys work
    critic_tensor_dict = optim_to_tensor("critic_optim", original_state)
    combined_dict = {**tensor_dict, **critic_tensor_dict}

    both_states = optim_from_tensor(combined_dict)
    assert "actor_optim" in both_states
    assert "critic_optim" in both_states


class TestLoadModel:
    def test_missing_optim_state(self):
        # Create a temporary directory structure with missing optim_state
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "model_save"
            save_path.mkdir()

            # Create model_state.safetensors and metadata.json but not optim_state.safetensors
            model_state_path = save_path / "model_state.safetensors"
            metadata_path = save_path / "metadata.json"

            # Create empty files
            model_state_path.touch()
            metadata_path.write_text("{}")

            # Verify optim_state.safetensors doesn't exist
            optim_state_path = save_path / "optim_state.safetensors"
            assert not optim_state_path.exists()

            # Attempt to load should raise FileNotFoundError
            with pytest.raises(
                FileNotFoundError, match="Optimizer state .* does not exist"
            ):
                load_model(NeuroFlow, save_path)

    def test_missing_metadata(self):
        # Create a temporary directory structure with missing metadata
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "model_save"
            save_path.mkdir()

            # Create model_state.safetensors and optim_state.safetensors but not metadata.json
            model_state_path = save_path / "model_state.safetensors"
            optim_state_path = save_path / "optim_state.safetensors"

            # Create empty files
            model_state_path.touch()
            optim_state_path.touch()

            # Verify metadata.json doesn't exist
            metadata_path = save_path / "metadata.json"
            assert not metadata_path.exists()

            # Attempt to load should raise FileNotFoundError
            with pytest.raises(FileNotFoundError, match="Metadata .* does not exist"):
                load_model(NeuroFlow, save_path)


class TestSaveModel:
    @pytest.fixture
    def test_agent(self) -> NeuroFlow:
        return NeuroFlow(
            "InvertedPendulum-v5",
            8,
            16,
            device="cpu",
        )

    def test_force_flag(self, tmp_path: Path, test_agent: NeuroFlow):
        # Define a checkpoint path in the temporary directory
        checkpoint_path = tmp_path / "test_model"

        # First save should work without force
        save_model(test_agent, checkpoint_path, buffer=True, config=True)

        # Verify files were created
        assert (checkpoint_path / "model_state.safetensors").exists()
        assert (checkpoint_path / "metadata.json").exists()
        assert (checkpoint_path / "buffer_state.safetensors").exists()
        assert (checkpoint_path.parent / "model_config.json").exists()

        # Without force, trying to save again should raise an error
        with pytest.raises(FileExistsError):
            save_model(test_agent, checkpoint_path, buffer=True, config=True)

        # Change something in the agent to verify the save is actually updated
        test_agent.actor.optim.zero_grad()  # Change optimizer state

        # With force=True, saving should succeed and overwrite
        save_model(test_agent, checkpoint_path, buffer=True, config=True, force=True)

        # Load the agent back to verify it saved properly
        loaded_agent = NeuroFlow.load(checkpoint_path, buffer=True)

        # Verify the agent was loaded successfully
        assert loaded_agent.state_dim == test_agent.state_dim
        assert loaded_agent.action_dim == test_agent.action_dim

        # Compare some parameters to ensure they match
        test_params = list(test_agent.actor.network.parameters())[0]
        loaded_params = list(loaded_agent.actor.network.parameters())[0]
        assert torch.allclose(test_params, loaded_params)
