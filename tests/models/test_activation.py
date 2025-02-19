import pytest

from velora.models.activation import ActivationEnum, ActivationTypeLiteral, LeCunTanh

import torch
import torch.nn as nn


class TestActivationEnum:
    @pytest.mark.parametrize(
        "activation",
        list(ActivationTypeLiteral.__dict__["__args__"]),
    )
    def test_get_str(self, activation: str):
        activation_layer = ActivationEnum.get(activation)
        assert isinstance(activation_layer, nn.Module)

    def test_enum(self):
        value = ActivationEnum.RELU.value
        assert isinstance(value, nn.Module)

    def test_invalid(self):
        with pytest.raises(ValueError):
            ActivationEnum.get("invalid_activation")


class TestLeCunTanh:
    def test_forward(self):
        activation = LeCunTanh()
        x = torch.tensor([-1.0, 0.0, 1.0])
        output: torch.Tensor = activation(x)
        assert output.shape == x.shape  # Output should match input shape

    def test_gradient(self):
        activation = LeCunTanh()
        x = torch.randn(5, requires_grad=True)
        y: torch.Tensor = activation(x).sum()
        y.backward()
        assert x.grad is not None  # Gradients should be computed
