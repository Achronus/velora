import torch
import pytest
from unittest.mock import patch

from velora.models.lnn.ncp import LiquidNCPNetwork, NCPNetwork
from velora.models.sac.continuous import SACActor, SACCritic, SACCriticNCP
from velora.models.sac.discrete import SACActorDiscrete, SACCriticDiscrete


class TestSACActor:
    @pytest.fixture
    def actor_params(self):
        return {
            "num_obs": 4,
            "n_neurons": 16,
            "num_actions": 2,
            "action_scale": torch.tensor([1.0, 1.0]),
            "action_bias": torch.tensor([0.0, 0.0]),
            "log_std_min": -5,
            "log_std_max": 2,
            "device": torch.device("cpu"),
        }

    @pytest.fixture
    def actor(self, actor_params) -> SACActor:
        with patch("torch.jit.script", lambda x: x):  # Skip JIT compilation
            return SACActor(**actor_params)

    def test_init(self, actor: SACActor, actor_params):
        assert actor.log_std_min == actor_params["log_std_min"]
        assert actor.log_std_max == actor_params["log_std_max"]
        assert torch.equal(actor.action_scale, actor_params["action_scale"])
        assert torch.equal(actor.action_bias, actor_params["action_bias"])
        assert isinstance(actor.ncp, LiquidNCPNetwork)

    def test_predict(self, actor: SACActor):
        obs = torch.zeros(1, 4)  # Batch of 1 observation
        hidden = None  # Test with None to use default initialization

        # Call the method
        actions, new_hidden = actor.predict(obs, hidden)

        # Check output shapes
        assert actions.shape == torch.Size([2])
        assert new_hidden is not None
        assert new_hidden.dim() > 0  # Should be a non-empty tensor

        # Check actions are in valid range
        assert torch.all(actions <= actor.action_scale + actor.action_bias)
        assert torch.all(actions >= -actor.action_scale + actor.action_bias)

    def test_forward(self, actor: SACActor):
        obs = torch.zeros(1, 4)  # Batch of 1 observation
        hidden = None  # Test with None to use default initialization

        # Call the method
        actions, log_prob, new_hidden = actor.forward(obs, hidden)

        # Check output shapes
        assert actions.shape == torch.Size([2])
        assert log_prob.shape == torch.Size([1])
        assert new_hidden is not None
        assert new_hidden.dim() > 0  # Should be a non-empty tensor

        # Check actions are in valid range
        assert torch.all(actions <= actor.action_scale + actor.action_bias)
        assert torch.all(actions >= -actor.action_scale + actor.action_bias)


class TestSACCritic:
    @pytest.fixture
    def critic_params(self):
        return {
            "num_obs": 4,
            "n_neurons": 16,
            "num_actions": 2,
            "device": torch.device("cpu"),
        }

    @pytest.fixture
    def critic(self, critic_params) -> SACCritic:
        with patch("torch.jit.script", lambda x: x):  # Skip JIT compilation
            return SACCritic(**critic_params)

    def test_init(self, critic: SACCritic, critic_params):
        assert isinstance(critic, SACCritic)
        assert isinstance(critic.ncp, LiquidNCPNetwork)
        assert (
            critic.ncp.in_features
            == critic_params["num_obs"] + critic_params["num_actions"]
        )
        assert critic.ncp.out_features == 1

    def test_forward(self, critic: SACCritic):
        obs = torch.zeros(1, 4)
        actions = torch.zeros(1, 2)
        hidden = None  # Test with None to use default initialization

        # Call the method
        q_values, new_hidden = critic.forward(obs, actions, hidden)

        # Check output shapes
        assert q_values.shape == torch.Size([1])
        assert new_hidden is not None
        assert new_hidden.dim() > 0  # Should be a non-empty tensor


class TestSACCriticNCP:
    @pytest.fixture
    def critic_params(self):
        return {
            "num_obs": 4,
            "n_neurons": 16,
            "num_actions": 2,
            "device": torch.device("cpu"),
        }

    @pytest.fixture
    def critic(self, critic_params) -> SACCriticNCP:
        with patch("torch.jit.script", lambda x: x):  # Skip JIT compilation
            return SACCriticNCP(**critic_params)

    def test_init(self, critic: SACCriticNCP, critic_params):
        assert isinstance(critic, SACCriticNCP)
        assert isinstance(critic.ncp, NCPNetwork)
        assert (
            critic.ncp.in_features
            == critic_params["num_obs"] + critic_params["num_actions"]
        )
        assert critic.ncp.out_features == 1

    def test_forward(self, critic: SACCriticNCP):
        obs = torch.zeros(1, 4)
        actions = torch.zeros(1, 2)

        # Call the method
        q_values = critic.forward(obs, actions)

        # Check output shape
        assert q_values.shape == torch.Size([1])


class TestSACActorDiscrete:
    @pytest.fixture
    def actor_params(self):
        return {
            "num_obs": 4,
            "n_neurons": 16,
            "num_actions": 3,  # Discrete action space with 3 possible actions
            "device": torch.device("cpu"),
        }

    @pytest.fixture
    def actor(self, actor_params) -> SACActorDiscrete:
        with patch("torch.jit.script", lambda x: x):  # Skip JIT compilation
            return SACActorDiscrete(**actor_params)

    def test_init(self, actor: SACActorDiscrete, actor_params):
        assert actor.num_actions == actor_params["num_actions"]
        assert hasattr(actor, "softmax")
        assert isinstance(actor.ncp, LiquidNCPNetwork)

    def test_predict(self, actor: SACActorDiscrete):
        obs = torch.zeros(1, 4)  # Batch of 1 observation
        hidden = None  # Test with None to use default initialization

        # Call the method
        actions, new_hidden = actor.predict(obs, hidden)

        # Check output shapes and values
        assert actions.shape == torch.Size([])
        assert actions.shape == torch.Size([])
        assert 0 <= actions.item() < 3  # Action should be valid

        # Check hidden state
        assert new_hidden is not None
        assert new_hidden.dim() > 0  # Should be a non-empty tensor

    def test_forward(self, actor: SACActorDiscrete):
        obs = torch.zeros(1, 4)  # Batch of 1 observation
        hidden = None  # Test with None to use default initialization

        # Call the method
        actions, probs, log_prob, new_hidden = actor.forward(obs, hidden)

        # Check output shapes
        assert actions.shape == torch.Size([])
        assert probs.shape == torch.Size([3])
        assert log_prob.shape == torch.Size([1])

        # Check action is valid
        assert 0 <= actions.item() < 3


class TestSACCriticDiscrete:
    @pytest.fixture
    def critic_params(self):
        return {
            "num_obs": 4,
            "n_neurons": 16,
            "num_actions": 3,  # Discrete action space with 3 possible actions
            "device": torch.device("cpu"),
        }

    @pytest.fixture
    def critic(self, critic_params) -> SACCriticDiscrete:
        with patch("torch.jit.script", lambda x: x):  # Skip JIT compilation
            return SACCriticDiscrete(**critic_params)

    def test_init(self, critic: SACCriticDiscrete, critic_params):
        assert isinstance(critic, SACCriticDiscrete)
        assert isinstance(critic.ncp, LiquidNCPNetwork)
        assert critic.ncp.in_features == critic_params["num_obs"]
        assert critic.ncp.out_features == critic_params["num_actions"]

    def test_forward(self, critic: SACCriticDiscrete):
        obs = torch.zeros(1, 4)
        hidden = None  # Test with None to use default initialization

        # Call the method
        q_values, new_hidden = critic.forward(obs, hidden)

        # Check output shapes
        assert q_values.shape == torch.Size([3])  # Should be (batch_size, num_actions)
        assert new_hidden is not None
        assert new_hidden.dim() > 0  # Should be a non-empty tensor
