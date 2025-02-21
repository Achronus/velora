import pytest

from velora.models.activation import ActivationEnum
from velora.models.backbone import MLP, BasicCNN

import torch
import torch.nn as nn


class TestMLP:
    def test_init(self):
        model = MLP(
            in_features=10,
            n_hidden=[20, 30],
            out_features=5,
            activation="relu",
            dropout_p=0.1,
        )
        assert isinstance(model.fc, nn.Sequential)
        assert len(model.fc) > 2

    def test_forward(self):
        model = MLP(in_features=10, n_hidden=[20, 30], out_features=5)
        x = torch.randn(2, 10)
        output: torch.Tensor = model(x)
        assert output.shape == (2, 5)

    @pytest.mark.parametrize(
        "activation", ["relu", "tanh", "sigmoid", "leaky_relu", "lecun_tanh"]
    )
    def test_activations(self, activation: str):
        model = MLP(in_features=10, n_hidden=20, out_features=5, activation=activation)
        assert isinstance(model.fc[1], ActivationEnum.get(activation).__class__)

    def test_dropout(self):
        model = MLP(in_features=10, n_hidden=[20, 30], out_features=5, dropout_p=0.5)
        assert any(isinstance(layer, nn.Dropout) for layer in model.fc)


class TestBasicCNN:
    @pytest.fixture
    def model(self) -> BasicCNN:
        return BasicCNN(in_channels=3)

    def test_init(self, model: BasicCNN):
        assert isinstance(model.conv, nn.Sequential)

    def test_forward(self, model: BasicCNN):
        x = torch.randn(2, 3, 84, 84)  # Batch of 2, 3 channels, 84x84 image size
        output: torch.Tensor = model(x)
        assert len(output.shape) == 2  # Should be a flattened output

    def test_out_size(self, model: BasicCNN):
        output_size = model.out_size((84, 84))
        assert isinstance(output_size, int)
        assert output_size > 0

    def test_out_size_shape_error(self, model: BasicCNN):
        x = torch.randn(1, 64, 64)

        with pytest.raises(ValueError):
            model.out_size(x)

    def test_forward_shape_error(self, model: BasicCNN):
        x = torch.randn(1, 64, 64)

        with pytest.raises(ValueError):
            model(x)
