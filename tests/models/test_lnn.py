from typing import Any, Literal
import pytest
from unittest.mock import patch

import torch
import torch.nn as nn

from velora.models.lnn import (
    NCPLiquidCell,
    LiquidNCPNetwork,
    SparseLinear,
    SparseParameter,
)
from velora.utils.core import set_seed
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

        # Check that SparseLinear layers are created
        assert isinstance(cell.g_head, SparseLinear)
        assert isinstance(cell.h_head, SparseLinear)
        assert isinstance(cell.f_head_to_g, SparseLinear)
        assert isinstance(cell.f_head_to_h, SparseLinear)
        assert isinstance(cell.proj, SparseLinear)

        # Check activation functions
        assert isinstance(cell.tanh, nn.Tanh)
        assert isinstance(cell.sigmoid, nn.Sigmoid)

        # Check sparsity mask
        assert torch.is_tensor(cell.sparsity_mask)
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

    def test_make_layer(self, cell: NCPLiquidCell):
        layer = cell._make_layer()

        # Check if it's a SparseLinear layer with correct dimensions
        assert isinstance(layer, SparseLinear)
        assert layer.in_features == cell.head_size
        assert layer.out_features == cell.n_hidden

        # Check if mask is applied
        assert torch.is_tensor(layer.weight.mask)
        assert layer.weight.mask.shape == (cell.n_hidden, cell.head_size)
        assert cell.device == layer.weight.device

    def test_new_hidden(self, cell: NCPLiquidCell):
        batch_size = 3
        x = torch.rand(batch_size, cell.head_size, device=cell.device)
        g_out = torch.rand(batch_size, cell.n_hidden, device=cell.device)
        h_out = torch.rand(batch_size, cell.n_hidden, device=cell.device)

        # Mock the forward methods of the layers
        with (
            patch.object(cell.f_head_to_g, "forward") as mock_f_to_g,
            patch.object(cell.f_head_to_h, "forward") as mock_f_to_h,
        ):
            # Configure mocks to return predictable values
            mock_f_to_g.return_value = (
                torch.ones(batch_size, cell.n_hidden, device=cell.device) * 0.5
            )
            mock_f_to_h.return_value = (
                torch.ones(batch_size, cell.n_hidden, device=cell.device) * 0.3
            )

            result = cell._new_hidden(x, g_out, h_out)

            # Shape check
            assert result.shape == (batch_size, cell.n_hidden)

            # Check that layers were called
            mock_f_to_g.assert_called_once_with(x)
            mock_f_to_h.assert_called_once_with(x)

            # Verify the calculation is correct
            # sigmoid(0.5 + 0.3) = sigmoid(0.8) ≈ 0.69
            gate_out = torch.sigmoid(
                torch.ones(batch_size, cell.n_hidden, device=cell.device) * 0.8
            )
            f_head = 1.0 - gate_out
            expected = torch.tanh(g_out) * f_head + gate_out * torch.tanh(h_out)
            assert torch.allclose(result, expected)

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
        x = torch.rand(batch_size, cell.in_features, device=cell.device)
        hidden = torch.rand(batch_size, cell.n_hidden, device=cell.device)

        # Create mock outputs
        mock_g_out = torch.rand(batch_size, cell.n_hidden, device=cell.device)
        mock_h_out = torch.rand(batch_size, cell.n_hidden, device=cell.device)
        mock_new_hidden = torch.rand(batch_size, cell.n_hidden, device=cell.device)
        mock_proj_out = torch.rand(batch_size, cell.n_hidden, device=cell.device)

        # Set up mocks for layers and methods
        with (
            patch.object(cell.g_head, "forward", return_value=mock_g_out),
            patch.object(cell.h_head, "forward", return_value=mock_h_out),
            patch.object(cell.proj, "forward", return_value=mock_proj_out),
            patch.object(cell, "_new_hidden", return_value=mock_new_hidden),
        ):
            # Run forward pass
            y_pred, new_hidden = cell(x, hidden)

            # Check that correct inputs are passed to _new_hidden
            concatenated = torch.cat([x, hidden], dim=1)
            cell._new_hidden.assert_called_once()
            args, _ = cell._new_hidden.call_args
            assert torch.allclose(args[0], concatenated)
            assert torch.allclose(args[1], mock_g_out)
            assert torch.allclose(args[2], mock_h_out)

            # Check output calculation
            assert torch.allclose(new_hidden, mock_new_hidden)
            assert torch.allclose(y_pred, mock_proj_out + mock_new_hidden)

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
        set_seed(64)
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

    def test_param_attrs(self, network: LiquidNCPNetwork):
        total = network.total_params
        active = network.active_params

        assert total == 1017
        assert active == 887

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
        assert h_state.shape == (batch_size, network.hidden_size)

    def test_forward_single_sample(
        self, network: LiquidNCPNetwork, network_params: NetworkParamsType
    ):
        x = torch.rand(1, network_params["in_features"])

        y_pred, h_state = network(x)

        # For batch_size=1, y_pred should be squeezed to (out_features)
        assert y_pred.shape == (network_params["out_features"],)
        assert h_state.shape == (1, network.hidden_size)

    def test_forward_with_provided_hidden_state(
        self, network: LiquidNCPNetwork, network_params: NetworkParamsType
    ):
        batch_size = 4
        x = torch.rand(batch_size, network_params["in_features"])
        h_state = torch.rand(batch_size, network.hidden_size)

        y_pred, new_h_state = network(x, h_state)

        # Check output shapes
        assert y_pred.shape == (batch_size, network_params["out_features"])
        assert new_h_state.shape == (batch_size, network.hidden_size)

        # Hidden state should be updated (different from input)
        assert not torch.allclose(h_state, new_h_state)

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
        assert h_state.shape == (batch_size, network.hidden_size)


class TestSparseParameter:
    def test_init(self):
        data = torch.ones(3, 4)
        mask = torch.ones(3, 4)
        mask[0, 0] = 0  # Set one value to zero in mask

        param = SparseParameter(data, mask)

        # Check that the parameter has been masked correctly
        assert param[0, 0].item() == 0
        assert param[0, 1].item() == 1
        assert param.mask.equal(mask)
        assert param.requires_grad is True

    def test_requires_grad_false(self):
        data = torch.ones(2, 2)
        mask = torch.ones(2, 2)
        param = SparseParameter(data, mask, requires_grad=False)
        assert param.requires_grad is False

    def test_mask_application(self):
        data = torch.ones(3, 3)
        mask = torch.zeros(3, 3)
        mask[1, 1] = 1  # Only one value should be non-zero

        param = SparseParameter(data, mask)

        # Check that only masked values are non-zero
        assert torch.sum(param != 0).item() == 1
        assert param[1, 1].item() == 1

    def test_data_assignment(self):
        data = torch.ones(2, 3)
        mask = torch.zeros(2, 3)
        mask[0, 1] = 1

        param = SparseParameter(data, mask)
        # Assign new data
        new_data = torch.full((2, 3), 5.0)
        param.data = new_data

        # Check that mask was applied to new data
        assert param[0, 0].item() == 0  # Masked value
        assert param[0, 1].item() == 5  # Unmasked value

    def test_deepcopy(self):
        import copy

        data = torch.ones(2, 2)
        mask = torch.ones(2, 2)
        mask[0, 0] = 0

        param = SparseParameter(data, mask)
        param_copy = copy.deepcopy(param)

        # Check that copy has same values but is a different object
        assert id(param) != id(param_copy)
        assert param_copy[0, 0].item() == 0
        assert param_copy[0, 1].item() == 1
        assert id(param.mask) != id(param_copy.mask)

    def test_apply_mask(self):
        data = torch.ones(3, 4)
        mask = torch.ones(3, 4)
        mask[0, 0] = 0
        mask[-1, -1] = 0

        param = SparseParameter(data, mask)
        param.apply_mask()

        assert param.data[0, 0].item() == 0
        assert param.data[-1, -1].item() == 0
        assert param.mask.equal(mask)
        assert any(param.data.not_equal(data).tolist())


class TestSparseLinear:
    def test_init(self):
        in_features = 5
        out_features = 3
        mask = torch.ones(out_features, in_features)
        mask[0, 0] = 0  # Mask one connection

        layer = SparseLinear(in_features, out_features, mask)

        # Check dimensions and properties
        assert layer.in_features == in_features
        assert layer.out_features == out_features
        assert isinstance(layer.weight, SparseParameter)
        assert layer.weight.shape == (out_features, in_features)
        assert layer.weight.mask.equal(mask)
        assert layer.bias is not None
        assert layer.bias.shape == (out_features,)

    def test_no_bias(self):
        in_features = 4
        out_features = 2
        mask = torch.ones(out_features, in_features)

        layer = SparseLinear(in_features, out_features, mask, bias=False)

        # Check that bias is None
        assert layer.bias is None

    def test_forward_pass(self):
        in_features = 3
        out_features = 2
        batch_size = 4

        # Create a mask where only the diagonal elements are non-zero
        mask = torch.zeros(out_features, in_features)
        for i in range(min(out_features, in_features)):
            mask[i, i] = 1

        layer = SparseLinear(in_features, out_features, mask)

        # Set weights to identity-like matrix for easy testing
        with torch.no_grad():
            layer.weight.data = torch.ones_like(layer.weight.data)
            layer.bias.data = torch.zeros_like(layer.bias.data)

        # Input with batch dimension
        x = torch.ones(batch_size, in_features)
        output = layer(x)

        # Check output shape
        assert output.shape == (batch_size, out_features)

        # Check that masked connections have no effect
        # Only the diagonal elements should contribute
        expected = torch.zeros(batch_size, out_features)
        for i in range(min(out_features, in_features)):
            expected[:, i] = 1.0  # Only diagonal elements are active

        assert torch.allclose(output, expected)

    def test_reset_parameters(self):
        in_features = 10
        out_features = 5
        mask = torch.ones(out_features, in_features)

        layer = SparseLinear(in_features, out_features, mask)

        # Store initial parameters
        initial_weight = layer.weight.clone()
        initial_bias = layer.bias.clone()

        # Reset parameters
        layer.reset_parameters()

        # Check that parameters have changed
        assert not torch.allclose(layer.weight, initial_weight)
        assert not torch.allclose(layer.bias, initial_bias)

        # Check that masked values remain zero
        assert torch.sum(layer.weight * (1 - mask)) == 0

    def test_extra_repr(self):
        in_features = 5
        out_features = 3
        mask = torch.ones(out_features, in_features)

        # With bias
        layer = SparseLinear(in_features, out_features, mask)
        repr_str = layer.extra_repr()
        assert f"in_features={in_features}" in repr_str
        assert f"out_features={out_features}" in repr_str
        assert "bias=True" in repr_str

        # Without bias
        layer_no_bias = SparseLinear(in_features, out_features, mask, bias=False)
        repr_str = layer_no_bias.extra_repr()
        assert "bias=False" in repr_str
