from typing import Any, Literal
import pytest
from unittest.mock import patch
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from velora.models.lnn import NCPLiquidCell, LiquidNCPNetwork
from velora.wiring import Wiring


CellParamsType = dict[Literal["in_features", "n_hidden", "mask", "device"], Any]
NetworkParamsType = dict[
    Literal["in_features", "n_neurons", "out_features", "sparsity_level", "device"],
    Any,
]


class TestNCPLiquidCell:
    @pytest.fixture
    def cell_params(self) -> CellParamsType:
        in_features = 8
        n_hidden = 6
        mask = torch.tensor(
            [
                [-1, 1, 0, -1, 0, 1],
                [0, -1, 1, 0, 1, 0],
                [1, 0, -1, 1, 0, 0],
                [0, 1, 0, -1, 1, 0],
                [1, 0, 1, 0, -1, 1],
                [0, 1, 0, 1, 0, -1],
                [1, 0, 0, 1, 0, 1],
                [0, 1, 1, 0, 1, 0],
            ]
        )

        return {
            "in_features": in_features,
            "n_hidden": n_hidden,
            "mask": mask,
            "device": torch.device("cpu"),
        }

    @pytest.fixture
    def cell(self, cell_params: CellParamsType) -> NCPLiquidCell:
        return NCPLiquidCell(**cell_params)

    def test_init(self, cell: NCPLiquidCell, cell_params: CellParamsType):
        assert cell.in_features == cell_params["in_features"]
        assert cell.n_hidden == cell_params["n_hidden"]
        assert cell.head_size == cell_params["n_hidden"] + cell_params["in_features"]
        assert cell.device == cell_params["device"]

        assert isinstance(cell.g_head, nn.Linear)
        assert isinstance(cell.h_head, nn.Linear)
        assert isinstance(cell.f_head_to_g, nn.Linear)
        assert isinstance(cell.f_head_to_h, nn.Linear)
        assert isinstance(cell.proj, nn.Linear)

        # Check activation functions
        assert isinstance(cell.tanh, nn.Tanh)
        assert isinstance(cell.sigmoid, nn.Sigmoid)

        # Check sparsity mask
        assert isinstance(cell.sparsity_mask, nn.Parameter)
        assert not cell.sparsity_mask.requires_grad

    def test_prep_mask(self, cell: NCPLiquidCell, cell_params: CellParamsType):
        original_mask: torch.Tensor = cell_params["mask"]
        processed_mask = cell._prep_mask(original_mask)

        n_extras = original_mask.shape[1]
        expected_shape = (original_mask.shape[1], original_mask.shape[0] + n_extras)
        assert processed_mask.shape == expected_shape

        # Check if negatives are converted to positives
        assert torch.all(processed_mask >= 0)
        assert torch.all(processed_mask <= 1)

        # Check if extra nodes section is all ones
        extra_section = processed_mask[:, original_mask.shape[0] :]
        assert torch.all(extra_section == 1)

    def test_sparse_head(self, cell: NCPLiquidCell):
        batch_size = 4
        x = torch.rand(batch_size, cell.head_size)

        # Create a mock head with known weights and bias
        mock_head = nn.Linear(cell.head_size, cell.n_hidden)

        # Test with actual sparsity mask
        result = cell._sparse_head(x, mock_head)

        # Shape check
        assert result.shape == (batch_size, cell.n_hidden)

        # Verify that the sparsity mask is applied
        with patch.object(F, "linear") as mock_linear:
            cell._sparse_head(x, mock_head)
            args, _ = mock_linear.call_args

            # Check if weight is modified by sparsity mask
            input_tensor, weight, bias = args
            assert torch.allclose(weight, mock_head.weight * cell.sparsity_mask)
            assert torch.allclose(bias, mock_head.bias)

    def test_new_hidden(self, cell: NCPLiquidCell):
        batch_size = 3
        x = torch.rand(batch_size, cell.head_size)
        g_out = torch.rand(batch_size, cell.n_hidden)
        h_out = torch.rand(batch_size, cell.n_hidden)

        # Mock the sparse_head to return predictable values
        with patch.object(cell, "_sparse_head") as mock_sparse_head:
            # Configure mock to return different values for different heads
            def side_effect(x, head):
                if head == cell.f_head_to_g:
                    return torch.ones(batch_size, cell.n_hidden) * 0.5
                elif head == cell.f_head_to_h:
                    return torch.ones(batch_size, cell.n_hidden) * 0.3
                return None

            mock_sparse_head.side_effect = side_effect

            result = cell._new_hidden(x, g_out, h_out)

            # Shape check
            assert result.shape == (batch_size, cell.n_hidden)

            # Verify calls to _sparse_head
            assert mock_sparse_head.call_count == 2

            # First call should be with f_head_to_g
            args1, _ = mock_sparse_head.call_args_list[0]
            assert args1[1] == cell.f_head_to_g

            # Second call should be with f_head_to_h
            args2, _ = mock_sparse_head.call_args_list[1]
            assert args2[1] == cell.f_head_to_h

    def test_forward(self, cell: NCPLiquidCell):
        batch_size = 5
        x = torch.rand(batch_size, cell.in_features)
        hidden = torch.rand(batch_size, cell.n_hidden)

        # Run forward pass
        y_pred, new_hidden = cell(x, hidden)

        # Check output shapes
        assert y_pred.shape == (batch_size, cell.n_hidden)
        assert new_hidden.shape == (batch_size, cell.n_hidden)

        # Ensure hidden state is updated (different from input)
        assert not torch.allclose(hidden, new_hidden)

    def test_forward_flow(self, cell: NCPLiquidCell):
        batch_size = 2
        x = torch.rand(batch_size, cell.in_features)
        hidden = torch.rand(batch_size, cell.n_hidden)

        # Create mock outputs
        mock_g_out = torch.rand(batch_size, cell.n_hidden)
        mock_h_out = torch.rand(batch_size, cell.n_hidden)
        mock_new_hidden = torch.rand(batch_size, cell.n_hidden)
        mock_proj_out = torch.rand(batch_size, cell.n_hidden)

        # Set up mocks for internal methods
        with (
            patch.object(cell, "_sparse_head") as mock_sparse_head,
            patch.object(
                cell, "_new_hidden", return_value=mock_new_hidden
            ) as mock_new_hidden_fn,
        ):
            # Configure sparse_head to return different values for different heads
            def side_effect(x, head):
                if head == cell.g_head:
                    return mock_g_out
                elif head == cell.h_head:
                    return mock_h_out
                elif head == cell.proj:
                    return mock_proj_out
                return None

            mock_sparse_head.side_effect = side_effect

            # Run forward pass
            y_pred, new_hidden = cell(x, hidden)

            # Verify calls to _sparse_head for g_head and h_head
            assert mock_sparse_head.call_count >= 3

            # Check call to _new_hidden
            mock_new_hidden_fn.assert_called_once()
            args, _ = mock_new_hidden_fn.call_args
            assert torch.allclose(args[1], mock_g_out)
            assert torch.allclose(args[2], mock_h_out)

            # Check output calculation
            assert torch.allclose(new_hidden, mock_new_hidden)
            assert torch.allclose(y_pred, mock_proj_out + mock_new_hidden)

    def test_device_handling(self, cell_params: CellParamsType):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            cell_params["device"] = device
            cell = NCPLiquidCell(**cell_params)

            # Create input on CPU
            batch_size = 3
            x = torch.rand(batch_size, cell.in_features)
            hidden = torch.rand(batch_size, cell.n_hidden)

            # Run forward pass
            y_pred, new_hidden = cell(x, hidden)

            # Check if outputs are on the specified device
            assert y_pred.device == device
            assert new_hidden.device == device

    def test_input_output_stability(self, cell: NCPLiquidCell):
        batch_size = 4
        x = torch.rand(batch_size, cell.in_features)
        hidden = torch.rand(batch_size, cell.n_hidden)

        # First forward pass
        y_pred1, new_hidden1 = cell(x, hidden)

        # Second forward pass with same inputs
        y_pred2, new_hidden2 = cell(x, hidden)

        # Outputs should be identical for deterministic network
        assert torch.allclose(y_pred1, y_pred2)
        assert torch.allclose(new_hidden1, new_hidden2)

    def test_gradient_flow(self, cell: NCPLiquidCell):
        # Enable gradient computation
        x = torch.rand(1, cell.in_features, requires_grad=True)
        hidden = torch.rand(1, cell.n_hidden, requires_grad=True)

        # Forward pass
        y_pred, _ = cell(x, hidden)

        # Create a loss and backpropagate
        loss = y_pred.sum()
        loss.backward()

        # Check if gradients are computed
        assert x.grad is not None
        assert hidden.grad is not None

        # Check gradients for weights
        assert cell.g_head.weight.grad is not None
        assert cell.h_head.weight.grad is not None
        assert cell.f_head_to_g.weight.grad is not None
        assert cell.f_head_to_h.weight.grad is not None
        assert cell.proj.weight.grad is not None

        # Sparsity mask should not have gradients
        assert cell.sparsity_mask.grad is None

    def test_forward_with_zero_inputs(self, cell: NCPLiquidCell):
        batch_size = 3
        x = torch.zeros(batch_size, cell.in_features)
        hidden = torch.zeros(batch_size, cell.n_hidden)

        # Should run without errors
        y_pred, new_hidden = cell(x, hidden)

        # Check output shapes
        assert y_pred.shape == (batch_size, cell.n_hidden)
        assert new_hidden.shape == (batch_size, cell.n_hidden)


class TestLiquidNCPNetwork:
    @pytest.fixture
    def network_params(self) -> NetworkParamsType:
        return {
            "in_features": 10,
            "n_neurons": 20,
            "out_features": 5,
            "sparsity_level": 0.5,
            "device": torch.device("cpu"),
        }

    @pytest.fixture
    def network(self, network_params: NetworkParamsType) -> LiquidNCPNetwork:
        return LiquidNCPNetwork(**network_params)

    def test_init(self, network: LiquidNCPNetwork, network_params: NetworkParamsType):
        assert network.in_features == network_params["in_features"]
        assert network.n_neurons == network_params["n_neurons"]
        assert network.out_features == network_params["out_features"]
        assert network.device == network_params["device"]
        assert (
            network.n_units
            == network_params["n_neurons"] + network_params["out_features"]
        )

        # Check if layers are created correctly
        assert isinstance(network.layers, OrderedDict)
        assert list(network.layers.keys()) == ["inter", "command", "motor"]
        assert all(
            isinstance(layer, NCPLiquidCell) for layer in network.layers.values()
        )

        # Check sequential model
        assert isinstance(network.ncp, nn.Sequential)
        assert len(network.ncp) == 3

    def test_wiring_init(self, network: LiquidNCPNetwork):
        assert hasattr(network, "_wiring")
        assert isinstance(network._wiring, Wiring)

        # Check if masks and counts exist
        assert hasattr(network, "_masks")
        assert hasattr(network, "_counts")

    def test_forward_shape(
        self, network: LiquidNCPNetwork, network_params: NetworkParamsType
    ):
        batch_size = 8
        x = torch.rand(batch_size, network_params["in_features"])

        y_pred, h_state = network(x)

        # Check output shapes
        assert y_pred.shape == (batch_size, network_params["out_features"])
        assert h_state.shape == (batch_size, network.n_units)

    def test_forward_single_sample(
        self, network: LiquidNCPNetwork, network_params: NetworkParamsType
    ):
        x = torch.rand(1, network_params["in_features"])

        y_pred, h_state = network(x)

        # For batch_size=1, y_pred should be squeezed to (out_features)
        assert y_pred.shape == (network_params["out_features"],)
        assert h_state.shape == (1, network.n_units)

    def test_forward_with_provided_hidden_state(
        self, network: LiquidNCPNetwork, network_params: NetworkParamsType
    ):
        batch_size = 4
        x = torch.rand(batch_size, network_params["in_features"])
        h_state = torch.rand(batch_size, network.n_units)

        y_pred, new_h_state = network(x, h_state)

        # Check output shapes
        assert y_pred.shape == (batch_size, network_params["out_features"])
        assert new_h_state.shape == (batch_size, network.n_units)

        # Hidden state should be updated (different from input)
        assert not torch.allclose(h_state, new_h_state)

    def test_ncp_forward(self, network: LiquidNCPNetwork):
        batch_size = 5
        x = torch.rand(batch_size, network.in_features)
        hidden = torch.rand(batch_size, network.n_units)

        y_pred, new_h_state = network._ncp_forward(x, hidden)

        # Check output shapes
        assert y_pred.shape == (batch_size, network.out_features)
        assert new_h_state.shape == (batch_size, network.n_units)

    def test_forward_invalid_dimensions(self, network: LiquidNCPNetwork):
        # 3D tensor (invalid)
        x = torch.rand(2, 3, network.in_features)

        with pytest.raises(ValueError) as excinfo:
            network(x)

        assert "Unsupported dimensionality" in str(excinfo.value)

    def test_device_handling(self, network_params: NetworkParamsType):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            network_params["device"] = device
            network = LiquidNCPNetwork(**network_params)

            # Create input on CPU
            x = torch.rand(4, network_params["in_features"])

            # Run forward pass
            y_pred, h_state = network(x)

            # Check if outputs are on the specified device
            assert y_pred.device == device
            assert h_state.device == device

    @pytest.mark.parametrize("sparsity_level", [0.1, 0.5, 0.9])
    def test_different_sparsity_levels(
        self, network_params: NetworkParamsType, sparsity_level: float
    ):
        network_params["sparsity_level"] = sparsity_level
        network = LiquidNCPNetwork(**network_params)

        batch_size = 3
        x = torch.rand(batch_size, network_params["in_features"])

        # Should run without errors
        y_pred, h_state = network(x)

        # Basic shape checks
        assert y_pred.shape == (batch_size, network_params["out_features"])
        assert h_state.shape == (batch_size, network.n_units)

    def test_layer_independence(self, network: LiquidNCPNetwork):
        with patch.object(
            network, "_ncp_forward", wraps=network._ncp_forward
        ) as mock_ncp_forward:
            batch_size = 2
            x = torch.rand(batch_size, network.in_features)
            h_state = torch.rand(batch_size, network.n_units)

            network(x, h_state)

            # Check if hidden state was split correctly
            args, _ = mock_ncp_forward.call_args
            passed_x, passed_h = args

            assert torch.allclose(passed_x, x)
            assert torch.allclose(passed_h, h_state)
