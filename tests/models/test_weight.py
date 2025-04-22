import pytest
import torch
import torch.nn as nn
import math

from velora.models.weight import (
    get_init_fn,
    init_linear_xavier,
    init_linear_sonar,
    init_linear_zero,
    init_linear_trunc_normal,
    init_linear_kaiming_uniform,
    init_linear_orthogonal,
    WeightInitType,
)


class TestInitLayerFunctions:
    @pytest.fixture
    def linear_layer(self) -> nn.Linear:
        return nn.Linear(10, 5)

    def test_get(self):
        # Test all defined weight initialization types
        assert get_init_fn("xavier") == init_linear_xavier
        assert get_init_fn("sonar") == init_linear_sonar
        assert get_init_fn("zero") == init_linear_zero
        assert get_init_fn("trunc_normal") == init_linear_trunc_normal
        assert get_init_fn("kaiming_uniform") == init_linear_kaiming_uniform
        assert get_init_fn("orthogonal") == init_linear_orthogonal

        # Test None case
        assert get_init_fn(None) is None

    def test_xavier(self, linear_layer):
        # Save original state for comparison
        original_weight = linear_layer.weight.clone()

        # Apply initialization
        init_linear_xavier(linear_layer)

        # Verify weights were changed
        assert not torch.allclose(linear_layer.weight, original_weight)

        # Verify bias is zero
        assert torch.all(linear_layer.bias == 0)

    def test_sonar(self, linear_layer):
        # Save original state for comparison
        original_weight = linear_layer.weight.clone()

        # Apply initialization with default parameters
        init_linear_sonar(linear_layer)

        # Verify weights were changed
        assert not torch.allclose(linear_layer.weight, original_weight)

        # Verify bias is zero
        assert torch.all(linear_layer.bias == 0)

        # Verify weight bounds
        expected_std = 0.006 * (3 / linear_layer.in_features) ** 0.5
        assert (
            linear_layer.weight.min() >= -expected_std - 1e-5
        )  # Add small epsilon for floating point precision
        assert linear_layer.weight.max() <= expected_std + 1e-5

    def test_zero(self, linear_layer):
        # Apply initialization
        init_linear_zero(linear_layer)

        # Verify weights are zero
        assert torch.all(linear_layer.weight == 0)

        # Verify bias is zero
        assert torch.all(linear_layer.bias == 0)

    def test_trunc_normal(self, linear_layer):
        # Save original state for comparison
        original_weight = linear_layer.weight.clone()

        # Apply initialization
        init_linear_trunc_normal(linear_layer)

        # Verify weights were changed
        assert not torch.allclose(linear_layer.weight, original_weight)

        # Verify bias is zero
        assert torch.all(linear_layer.bias == 0)

        # Check standard deviation is close to specified value (1e-3)
        # Allow some deviation due to truncation
        assert abs(linear_layer.weight.std().item() - 1e-3) < 5e-4

    def test_kaiming_uniform(self, linear_layer):
        # Save original state for comparison
        original_weight = linear_layer.weight.clone()
        original_bias = linear_layer.bias.clone()

        # Apply initialization
        init_linear_kaiming_uniform(linear_layer)

        # Verify weights were changed
        assert not torch.allclose(linear_layer.weight, original_weight)

        # Verify bias was changed
        assert not torch.allclose(linear_layer.bias, original_bias)

        # Verify bias bounds
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear_layer.weight)
        bound = 1 / math.sqrt(fan_in)
        assert linear_layer.bias.min() >= -bound - 1e-5
        assert linear_layer.bias.max() <= bound + 1e-5

    def test_orthogonal(self, linear_layer):
        # Save original state for comparison
        original_weight = linear_layer.weight.clone()

        # Apply initialization
        init_linear_orthogonal(linear_layer)

        # Verify weights were changed
        assert not torch.allclose(linear_layer.weight, original_weight)

        # Verify bias is zero
        assert torch.all(linear_layer.bias == 0)

        # Check orthogonality property (W * W^T should be close to identity for rows)
        # Since we have more rows than columns, we can't have perfect orthogonality
        # We'll check a weaker condition
        if linear_layer.weight.shape[0] <= linear_layer.weight.shape[1]:
            # More columns than rows - can check orthogonality
            product = torch.mm(linear_layer.weight, linear_layer.weight.t())
            identity = torch.eye(
                linear_layer.weight.shape[0], device=linear_layer.weight.device
            )
            assert torch.allclose(product, identity, atol=1e-6)

    def test_init_with_bias_none(self):
        # Create a linear layer without bias
        layer = nn.Linear(10, 5, bias=False)

        # Test each initialization function
        init_funcs = [
            init_linear_xavier,
            init_linear_sonar,
            init_linear_zero,
            init_linear_trunc_normal,
            init_linear_kaiming_uniform,
            init_linear_orthogonal,
        ]

        for init_func in init_funcs:
            # This should not raise an exception
            init_func(layer)
            # No assertion needed - we're just checking that no exception is raised

    def test_weight_init_type_enum(self):
        expected_types = {
            "xavier",
            "sonar",
            "zero",
            "trunc_normal",
            "kaiming_uniform",
            "orthogonal",
        }

        # Check that all expected types are in the WeightInitType
        for init_type in expected_types:
            assert init_type in WeightInitType.__args__

        # Check that WeightInitType doesn't contain unexpected types
        assert set(WeightInitType.__args__) == expected_types

    def test_get_with_all_types(self):
        for init_type in WeightInitType.__args__:
            init_fn = get_init_fn(init_type)
            assert callable(init_fn)  # The returned value should be callable

    def test_sonar_custom_std(self, linear_layer):
        custom_std = 0.01

        # Apply initialization with custom std
        init_linear_sonar(linear_layer, sonar_std=custom_std)

        # Verify weight bounds
        expected_std = custom_std * (3 / linear_layer.in_features) ** 0.5
        assert linear_layer.bias.min() == 0
        assert linear_layer.weight.min() >= -expected_std - 1e-5
        assert linear_layer.weight.max() <= expected_std + 1e-5

    def test_orthogonal_custom_gain(self, linear_layer):
        custom_gain = 2.0

        # Apply initialization with custom gain
        init_linear_orthogonal(linear_layer, gain=custom_gain)

        # Standard orthogonal initialization has expected norm
        # With custom gain, the norm should be scaled
        original_norm = torch.norm(linear_layer.weight).item()

        # Reset and apply with gain=1.0
        linear_layer = nn.Linear(10, 5)
        init_linear_orthogonal(linear_layer, gain=1.0)
        base_norm = torch.norm(linear_layer.weight).item()

        # Check that the norm scales approximately with gain
        # Allow some tolerance due to randomness in initialization
        assert abs(original_norm / base_norm - custom_gain) < 0.5
