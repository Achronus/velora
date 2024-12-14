import torch
import torch.nn as nn

from velora.policy.base import FeatureExtractor
from velora.policy.inputs import CNNInputs, MLPInputs


class MlpExtractor(FeatureExtractor):
    """
    A multi-layered perceptron backbone.

    Args:
        inputs (velora.policy.MLPInputs): a model containing the input parameters
    """

    def __init__(self, inputs: MLPInputs) -> None:
        super().__init__(inputs)

        layers = []
        self.out_features = inputs.in_features

        for h_units in inputs.n_hidden:
            layers.extend(
                [
                    nn.Linear(self.out_features, h_units),
                    nn.ReLU(),
                ]
            )
            self.out_features = h_units

        self.fc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class CNNExtractor(FeatureExtractor):
    """
    CNN backbone from DQN Nature paper with a custom MLP:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    Args:
        inputs (velora.policy.CNNInputs): a model containing the input parameters
    """

    def __init__(self, inputs: CNNInputs) -> None:
        super().__init__(inputs)

        self.conv = nn.Sequential(
            nn.Conv2d(inputs.in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        n_flatten = self._calc_n_flatten((inputs.in_channels, *inputs.img_shape))

        self.fc = MlpExtractor(
            inputs=MLPInputs(in_features=n_flatten, n_hidden=inputs.n_hidden)
        )

        self.out_features = self.fc.out_features

    def _calc_n_flatten(self, dim: tuple[int, ...]) -> int:
        """Calculates the size of the convolution output using a dummy input."""
        with torch.no_grad():
            x = torch.zeros(1, *dim)
            return self.conv(x).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))
