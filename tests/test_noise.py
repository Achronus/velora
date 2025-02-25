import pytest
from typing import Generator, Tuple

import torch
import numpy as np

from velora.noise import OUNoise

NoiseParamsType = Tuple[int, float, float, float]


class TestOUNoise:
    @pytest.fixture
    def noise_params(self) -> NoiseParamsType:
        return (
            3,  # size
            0.0,  # mu
            0.15,  # theta
            0.2,  # sigma
        )

    @pytest.fixture
    def ou_noise(self, noise_params: NoiseParamsType) -> OUNoise:
        size, mu, theta, sigma = noise_params
        return OUNoise(size, mu=mu, theta=theta, sigma=sigma)

    @pytest.fixture
    def seed_generator(self) -> Generator[None, None, None]:
        torch.manual_seed(42)
        yield
        torch.manual_seed(torch.initial_seed())

    def test_init(self, noise_params: NoiseParamsType) -> None:
        size, mu, theta, sigma = noise_params
        noise = OUNoise(size, mu=mu, theta=theta, sigma=sigma)

        assert noise.theta == theta, "Incorrect theta value"
        assert noise.sigma == sigma, "Incorrect sigma value"
        assert torch.all(noise.mu == mu), "Incorrect mu value"
        assert noise.mu.shape == (size,), "Incorrect mu shape"
        assert noise.device is None, "Incorrect default device"

    def test_custom_device(self) -> None:
        device = torch.device("cpu")  # Using CPU as it's always available
        noise = OUNoise(3, device=device)

        assert noise.device == device, "Device not set correctly"
        assert noise.mu.device == device, "Mu tensor not on correct device"
        assert noise.state.device == device, "State tensor not on correct device"

    def test_reset(self, ou_noise: OUNoise) -> None:
        # Generate some noise to change the state
        ou_noise.sample()

        # Store the state
        state_before_reset = ou_noise.state.clone()

        # Reset the process
        ou_noise.reset()

        # Check if state is reset to mu
        assert torch.all(ou_noise.state == ou_noise.mu), "Reset didn't return to mu"
        assert not torch.all(ou_noise.state == state_before_reset), (
            "State unchanged after reset"
        )

    def test_sample_shape(self, ou_noise: OUNoise) -> None:
        sample = ou_noise.sample()
        assert sample.shape == ou_noise.mu.shape, "Sample shape doesn't match mu shape"

    def test_multiple_samples(
        self, ou_noise: OUNoise, seed_generator: Generator
    ) -> None:
        sample1 = ou_noise.sample().clone()  # Clone to get a separate copy
        sample2 = ou_noise.sample().clone()  # Clone to get a separate copy
        assert not torch.all(sample1 == sample2), "Consecutive samples are identical"

    def test_mean_reversion(self, seed_generator: Generator) -> None:
        """Test if noise process exhibits mean reversion."""
        # Create noise process with strong mean reversion
        noise = OUNoise(1, mu=0.0, theta=0.9, sigma=0.1)

        # Start from an extreme value
        noise.state = torch.tensor([5.0])

        # Sample multiple times and check if it moves toward mean
        samples = []
        for _ in range(100):
            samples.append(noise.sample().item())

        # Check if final samples are closer to mu than initial samples
        first_10_avg = abs(np.mean(samples[:10]))
        last_10_avg = abs(np.mean(samples[-10:]))

        assert last_10_avg < first_10_avg, "Process didn't show mean reversion"

    def test_noise_scale(self, seed_generator: Generator) -> None:
        # Create two noise processes with different sigma values
        noise_small = OUNoise(1, sigma=0.1)
        noise_large = OUNoise(1, sigma=1.0)

        # Generate samples
        samples_small = [noise_small.sample().item() for _ in range(1000)]
        samples_large = [noise_large.sample().item() for _ in range(1000)]

        # Compare standard deviations
        std_small = np.std(samples_small)
        std_large = np.std(samples_large)

        assert std_large > std_small, "Larger sigma didn't result in larger variance"

    def test_negative_parameters(self) -> None:
        with pytest.raises(ValueError):
            OUNoise(3, theta=-0.15)  # Negative theta should raise error

        with pytest.raises(ValueError):
            OUNoise(3, sigma=-0.2)  # Negative sigma should raise error

    def test_zero_size(self) -> None:
        with pytest.raises(ValueError):
            OUNoise(0)  # Zero size should raise error
