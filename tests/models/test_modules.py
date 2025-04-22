from typing import Any, Dict, Literal
import pytest
from unittest.mock import patch

import torch
from torch.jit import RecursiveScriptModule

from velora.models.nf.modules import (
    ActorModule,
    ActorModuleDiscrete,
    CriticModule,
    CriticModuleDiscrete,
    EntropyModule,
    EntropyModuleDiscrete,
)


NetworkParamsType = Dict[
    Literal["state_dim", "n_neurons", "action_dim", "device"],
    Any,
]


class TestActorModule:
    @pytest.fixture
    def actor_params(self) -> NetworkParamsType:
        return {
            "state_dim": 4,
            "n_neurons": 16,
            "action_dim": 2,
            "action_scale": torch.tensor([1.0, 1.0]),
            "action_bias": torch.tensor([0.0, 0.0]),
            "log_std_min": -5,
            "log_std_max": 2,
            "optim": torch.optim.Adam,
            "lr": 3e-4,
            "device": torch.device("cpu"),
        }

    @pytest.fixture
    def actor_module(self, actor_params) -> ActorModule:
        return ActorModule(**actor_params)

    def test_init(self, actor_module: ActorModule, actor_params: Dict[str, Any]):
        assert isinstance(actor_module.network, RecursiveScriptModule)
        assert actor_module.state_dim == actor_params["state_dim"]
        assert actor_module.action_dim == actor_params["action_dim"]
        assert actor_module.hidden_size > 0
        assert hasattr(actor_module, "optim")

    def test_predict(self, actor_params: Dict[str, Any]):
        with patch("torch.jit.script") as mock_script:
            mock_script.side_effect = lambda module: module

            actor_module = ActorModule(**actor_params)

            # Add batch dimension to state
            state = torch.randn(
                1,
                actor_params["state_dim"],
                device=actor_params["device"],
            )

            # Test predict (deterministic) with explicit None for hidden
            action, hidden = actor_module.predict(state, None)

            # Check shape with batch dimension
            assert action.shape == torch.Size([actor_params["action_dim"]])
            assert hidden is not None
            assert hidden.dim() >= 1  # Should be at least 1D tensor

    def test_forward(self, actor_params: Dict[str, Any]):
        with patch("torch.jit.script") as mock_script:
            mock_script.side_effect = lambda module: module

            actor_module = ActorModule(**actor_params)

            # Add batch dimension to state
            state = torch.randn(
                1, actor_params["state_dim"], device=actor_params["device"]
            )

            # Test forward (stochastic) with explicit None for hidden
            action, log_prob, hidden = actor_module.forward(state, None)

            # Check shapes with batch dimension
            assert action.shape == torch.Size([actor_params["action_dim"]])
            assert log_prob.numel() == 1  # Should have 1 element
            assert hidden is not None

    def test_gradient_steps(self, actor_module: ActorModule):
        loss = torch.tensor(1.0, requires_grad=True)
        actor_module.gradient_step(loss)

    def test_repr(self, actor_module: ActorModule):
        assert (
            repr(actor_module)
            == "ActorModule(state_dim=4, n_neurons=16, action_dim=2, action_scale=tensor([1., 1.]), action_bias=tensor([0., 0.]), optim=Adam, log_std=(-5, 2), lr=0.0003, device=cpu)"
        )


class TestActorModuleDiscrete:
    @pytest.fixture
    def actor_params(self) -> NetworkParamsType:
        return {
            "state_dim": 4,
            "n_neurons": 16,
            "action_dim": 2,
            "optim": torch.optim.Adam,
            "lr": 3e-4,
            "device": torch.device("cpu"),
        }

    @pytest.fixture
    def actor_module(self, actor_params) -> ActorModuleDiscrete:
        return ActorModuleDiscrete(**actor_params)

    def test_init(
        self, actor_module: ActorModuleDiscrete, actor_params: Dict[str, Any]
    ):
        assert isinstance(actor_module.network, RecursiveScriptModule)
        assert actor_module.state_dim == actor_params["state_dim"]
        assert actor_module.action_dim == actor_params["action_dim"]
        assert actor_module.hidden_size > 0
        assert hasattr(actor_module, "optim")

    def test_predict(self, actor_params: Dict[str, Any]):
        with patch("torch.jit.script") as mock_script:
            mock_script.side_effect = lambda module: module

            actor_module = ActorModuleDiscrete(**actor_params)

            # Add batch dimension to state
            state = torch.randn(
                1,
                actor_params["state_dim"],
                device=actor_params["device"],
            )

            # Test predict (deterministic) with explicit None for hidden
            action, hidden = actor_module.predict(state, None)

            # Check shape with batch dimension
            assert action.shape == torch.Size([])
            assert hidden is not None
            assert hidden.dim() >= 1  # Should be at least 1D tensor

    def test_forward(self, actor_params: Dict[str, Any]):
        with patch("torch.jit.script") as mock_script:
            mock_script.side_effect = lambda module: module

            actor_module = ActorModuleDiscrete(**actor_params)

            # Add batch dimension to state
            state = torch.randn(
                1, actor_params["state_dim"], device=actor_params["device"]
            )

            # Test forward (stochastic) with explicit None for hidden
            action, probs, log_prob, hidden = actor_module.forward(state, None)

            # Check shapes with batch dimension
            assert action.shape == torch.Size([])
            assert probs.numel() == 2
            assert log_prob.numel() == 1  # Should have 1 element
            assert hidden is not None

    def test_gradient_steps(self, actor_module: ActorModuleDiscrete):
        loss = torch.tensor(1.0, requires_grad=True)
        actor_module.gradient_step(loss)

    def test_repr(self, actor_module: ActorModuleDiscrete):
        assert (
            repr(actor_module)
            == "ActorModuleDiscrete(state_dim=4, n_neurons=16, action_dim=2, optim=Adam, lr=0.0003, device=cpu)"
        )


class TestCriticModule:
    @pytest.fixture
    def critic_params(self) -> NetworkParamsType:
        return {
            "state_dim": 4,
            "n_neurons": 16,
            "action_dim": 2,
            "optim": torch.optim.Adam,
            "lr": 3e-4,
            "tau": 0.005,
            "device": torch.device("cpu"),
        }

    @pytest.fixture
    def critic_module(self, critic_params) -> CriticModule:
        return CriticModule(**critic_params)

    def test_init(self, critic_module: CriticModule, critic_params: Dict[str, Any]):
        assert isinstance(critic_module.network1, RecursiveScriptModule)
        assert isinstance(critic_module.network2, RecursiveScriptModule)
        assert isinstance(critic_module.target1, RecursiveScriptModule)
        assert isinstance(critic_module.target2, RecursiveScriptModule)
        assert critic_module.state_dim == critic_params["state_dim"]
        assert critic_module.action_dim == critic_params["action_dim"]
        assert critic_module.tau == critic_params["tau"]
        assert hasattr(critic_module, "optim1")
        assert hasattr(critic_module, "optim2")

    def test_predict(self, critic_params: Dict[str, Any]):
        with patch("torch.jit.script") as mock_script:
            mock_script.side_effect = lambda module: module

            critic_module = CriticModule(**critic_params)

            # Use batch dimension for inputs
            state = torch.randn(
                1, critic_params["state_dim"], device=critic_params["device"]
            )
            action = torch.randn(
                1, critic_params["action_dim"], device=critic_params["device"]
            )

            # Test predict
            q1, q2 = critic_module.predict(state, action)

            # Check shapes - should be (batch_size, 1)
            assert q1.shape == torch.Size([1])
            assert q2.shape == torch.Size([1])

    def test_target_predict(self, critic_params: Dict[str, Any]):
        with patch("torch.jit.script") as mock_script:
            mock_script.side_effect = lambda module: module

            critic_module = CriticModule(**critic_params)

            # Use batch dimension for inputs
            state = torch.randn(
                1, critic_params["state_dim"], device=critic_params["device"]
            )
            action = torch.randn(
                1, critic_params["action_dim"], device=critic_params["device"]
            )

            # Test target predict
            q_target = critic_module.target_predict(state, action)

            # Check shape - should be (batch_size, 1)
            assert q_target.shape == torch.Size([1])

    def test_update_targets(self, critic_module: CriticModule):
        initial_target1_params = {
            name: param.clone()
            for name, param in critic_module.target1.named_parameters()
        }

        # Modify network parameters
        for param in critic_module.network1.parameters():
            param.data = param.data + 1.0

        # Perform update
        critic_module.update_targets()

        # Check if parameters changed but not completely
        for name, param in critic_module.target1.named_parameters():
            assert not torch.allclose(param, initial_target1_params[name])
            assert not torch.allclose(param, param + 1.0)  # Shouldn't be fully updated

    def test_gradient_steps(self, critic_module: CriticModule):
        c1_loss = torch.tensor(1.0, requires_grad=True)
        c2_loss = torch.tensor(1.0, requires_grad=True)
        critic_module.gradient_step(c1_loss, c2_loss)

    def test_repr(self, critic_module: CriticModule):
        assert (
            repr(critic_module)
            == "CriticModule(state_dim=4, n_neurons=16, action_dim=2, optim=Adam, lr=0.0003, tau=0.005, device=cpu)"
        )


class TestCriticModuleDiscrete:
    @pytest.fixture
    def critic_params(self) -> NetworkParamsType:
        return {
            "state_dim": 4,
            "n_neurons": 16,
            "action_dim": 2,
            "optim": torch.optim.Adam,
            "lr": 3e-4,
            "tau": 0.005,
            "device": torch.device("cpu"),
        }

    @pytest.fixture
    def critic_module(self, critic_params) -> CriticModuleDiscrete:
        return CriticModuleDiscrete(**critic_params)

    def test_init(
        self, critic_module: CriticModuleDiscrete, critic_params: Dict[str, Any]
    ):
        assert isinstance(critic_module.network1, RecursiveScriptModule)
        assert isinstance(critic_module.network2, RecursiveScriptModule)
        assert isinstance(critic_module.target1, RecursiveScriptModule)
        assert isinstance(critic_module.target2, RecursiveScriptModule)
        assert critic_module.state_dim == critic_params["state_dim"]
        assert critic_module.action_dim == critic_params["action_dim"]
        assert critic_module.tau == critic_params["tau"]
        assert hasattr(critic_module, "optim1")
        assert hasattr(critic_module, "optim2")

    def test_predict(self, critic_params: Dict[str, Any]):
        with patch("torch.jit.script") as mock_script:
            mock_script.side_effect = lambda module: module

            critic_module = CriticModuleDiscrete(**critic_params)

            # Use batch dimension for inputs
            state = torch.randn(
                1, critic_params["state_dim"], device=critic_params["device"]
            )

            # Test predict
            q1, q2 = critic_module.predict(state)

            # Check shapes - should be (batch_size, 2)
            assert q1.shape == torch.Size([2])
            assert q2.shape == torch.Size([2])

    def test_target_predict(self, critic_params: Dict[str, Any]):
        with patch("torch.jit.script") as mock_script:
            mock_script.side_effect = lambda module: module

            critic_module = CriticModuleDiscrete(**critic_params)

            # Use batch dimension for inputs
            state = torch.randn(
                1, critic_params["state_dim"], device=critic_params["device"]
            )

            # Test target predict
            q_target = critic_module.target_predict(state)

            # Check shape - should be (batch_size, 2)
            assert q_target.shape == torch.Size([2])

    def test_update_targets(self, critic_module: CriticModuleDiscrete):
        initial_target1_params = {
            name: param.clone()
            for name, param in critic_module.target1.named_parameters()
        }

        # Modify network parameters
        for param in critic_module.network1.parameters():
            param.data = param.data + 1.0

        # Perform update
        critic_module.update_targets()

        # Check if parameters changed but not completely
        for name, param in critic_module.target1.named_parameters():
            assert not torch.allclose(param, initial_target1_params[name])
            assert not torch.allclose(param, param + 1.0)  # Shouldn't be fully updated

    def test_gradient_steps(self, critic_module: CriticModuleDiscrete):
        c1_loss = torch.tensor(1.0, requires_grad=True)
        c2_loss = torch.tensor(1.0, requires_grad=True)
        critic_module.gradient_step(c1_loss, c2_loss)

    def test_repr(self, critic_module: CriticModuleDiscrete):
        assert (
            repr(critic_module)
            == "CriticModuleDiscrete(state_dim=4, n_neurons=16, action_dim=2, optim=Adam, lr=0.0003, tau=0.005, device=cpu)"
        )


class TestEntropyModule:
    @pytest.fixture
    def entropy_params(self) -> Dict[str, Any]:
        return {
            "action_dim": 2,
            "initial_alpha": 1.0,
            "optim": torch.optim.Adam,
            "lr": 3e-4,
            "device": torch.device("cpu"),
        }

    @pytest.fixture
    def entropy_module(self, entropy_params) -> EntropyModule:
        return EntropyModule(**entropy_params)

    def test_init(self, entropy_module: EntropyModule, entropy_params: Dict[str, Any]):
        assert entropy_module.action_dim == entropy_params["action_dim"]
        assert entropy_module.initial_alpha == entropy_params["initial_alpha"]
        assert hasattr(entropy_module, "log_alpha")
        assert hasattr(entropy_module, "optim")

    def test_alpha(self, entropy_module: EntropyModule):
        # Test alpha property
        alpha = entropy_module.alpha
        assert isinstance(alpha, torch.Tensor)
        assert alpha.item() > 0  # Alpha should be positive

    def test_compute_loss(self, entropy_module: EntropyModule):
        log_probs = torch.randn(10, 1)

        # Test compute loss
        loss = entropy_module.compute_loss(log_probs)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])  # Should be a scalar

    def test_gradient_steps(self, entropy_module: EntropyModule):
        entropy_loss = torch.tensor(1.0, requires_grad=True)
        entropy_module.gradient_step(entropy_loss)

    def test_repr(self, entropy_module: EntropyModule):
        assert (
            repr(entropy_module)
            == "EntropyModule(action_dim=2, initial_alpha=1.0, optim=Adam, lr=0.0003, device=cpu)"
        )


class TestEntropyModuleDiscrete:
    @pytest.fixture
    def entropy_params(self) -> Dict[str, Any]:
        return {
            "action_dim": 2,
            "initial_alpha": 1.0,
            "optim": torch.optim.Adam,
            "lr": 3e-4,
            "device": torch.device("cpu"),
        }

    @pytest.fixture
    def entropy_module(self, entropy_params) -> EntropyModuleDiscrete:
        return EntropyModuleDiscrete(**entropy_params)

    def test_init(
        self, entropy_module: EntropyModuleDiscrete, entropy_params: Dict[str, Any]
    ):
        assert entropy_module.action_dim == entropy_params["action_dim"]
        assert entropy_module.initial_alpha == entropy_params["initial_alpha"]
        assert hasattr(entropy_module, "log_alpha")
        assert hasattr(entropy_module, "optim")

    def test_alpha(self, entropy_module: EntropyModuleDiscrete):
        # Test alpha property
        alpha = entropy_module.alpha
        assert isinstance(alpha, torch.Tensor)
        assert alpha.item() > 0  # Alpha should be positive

    def test_compute_loss(self, entropy_module: EntropyModuleDiscrete):
        probs = torch.randn(10, 1)
        log_probs = torch.randn(10, 1)

        # Test compute loss
        loss = entropy_module.compute_loss(probs, log_probs)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])  # Should be a scalar

    def test_gradient_steps(self, entropy_module: EntropyModuleDiscrete):
        entropy_loss = torch.tensor(1.0, requires_grad=True)
        entropy_module.gradient_step(entropy_loss)

    def test_repr(self, entropy_module: EntropyModuleDiscrete):
        assert (
            repr(entropy_module)
            == "EntropyModuleDiscrete(action_dim=2, initial_alpha=1.0, optim=Adam, lr=0.0003, device=cpu)"
        )
