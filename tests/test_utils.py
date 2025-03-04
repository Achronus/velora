import pytest
from unittest.mock import patch
from typing import Generator, List, Tuple

import torch
import torch.nn as nn
import numpy as np

from velora.utils.core import set_seed, set_device
from velora.utils.torch import to_tensor, stack_tensor, soft_update, hard_update


SampleDataType = Tuple[List[float], List[int], List[bool]]


class TestSetSeed:
    @pytest.fixture
    def random_seed(self) -> int:
        return 42

    def test_consistency(self, random_seed: int) -> None:
        set_seed(random_seed)

        # Generate random numbers
        torch_random1 = torch.rand(5)
        numpy_random1 = np.random.rand(5)

        set_seed(random_seed)

        # Generate random numbers again
        torch_random2 = torch.rand(5)
        numpy_random2 = np.random.rand(5)

        # Check if the random numbers are the same
        assert torch.all(torch_random1 == torch_random2), (
            "PyTorch random numbers are not consistent"
        )
        assert np.all(numpy_random1 == numpy_random2), (
            "NumPy random numbers are not consistent"
        )

    def test_different_values(self) -> None:
        # Set first seed
        set_seed(42)
        torch_random1 = torch.rand(5)
        numpy_random1 = np.random.rand(5)

        # Set different seed
        set_seed(43)
        torch_random2 = torch.rand(5)
        numpy_random2 = np.random.rand(5)

        # Check if the random numbers are different
        assert not torch.all(torch_random1 == torch_random2), (
            "PyTorch random numbers should be different"
        )
        assert not np.all(numpy_random1 == numpy_random2), (
            "NumPy random numbers should be different"
        )


class TestSetDevice:
    @pytest.fixture
    def cuda_available(self) -> Generator:
        with patch("torch.cuda.is_available", return_value=True) as mock:
            yield mock

    @pytest.fixture
    def cuda_unavailable(self) -> Generator:
        with patch("torch.cuda.is_available", return_value=False) as mock:
            yield mock

    def test_auto_with_cuda(self, cuda_available: Generator) -> None:
        device = set_device("auto")
        assert device == torch.device("cuda:0"), "Should select CUDA when available"

    def test_auto_without_cuda(self, cuda_unavailable: Generator) -> None:
        device = set_device("auto")
        assert device == torch.device("cpu"), (
            "Should select CPU when CUDA is unavailable"
        )

    def test_specific_cpu(self) -> None:
        device = set_device("cpu")
        assert device == torch.device("cpu"), "Should use CPU when explicitly specified"

    def test_specific_cuda(self) -> None:
        device = set_device("cuda:0")
        assert device == torch.device("cuda:0"), (
            "Should use CUDA when explicitly specified"
        )

    def test_invalid(self) -> None:
        with pytest.raises(RuntimeError):
            set_device("invalid_device")


class TestToTensor:
    @pytest.fixture
    def sample_data(self) -> SampleDataType:
        return (
            [1.0, 2.0, 3.0],
            [1, 2, 3],
            [True, False, True],
        )

    def test_float_list(self, sample_data: SampleDataType) -> None:
        """Test converting float list to tensor."""
        float_list = sample_data[0]
        tensor = to_tensor(float_list)
        assert tensor.dtype == torch.float32
        assert torch.allclose(tensor, torch.tensor(float_list, dtype=torch.float32))

    def test_int_list(self, sample_data: SampleDataType) -> None:
        """Test converting int list to tensor."""
        int_list = sample_data[1]
        tensor = to_tensor(int_list, dtype=torch.int64)
        assert tensor.dtype == torch.int64
        assert torch.all(tensor == torch.tensor(int_list, dtype=torch.int64))

    def test_bool_list(self, sample_data: SampleDataType) -> None:
        """Test converting bool list to tensor."""
        bool_list = sample_data[2]
        tensor = to_tensor(bool_list, dtype=torch.bool)
        assert tensor.dtype == torch.bool
        assert torch.all(tensor == torch.tensor(bool_list, dtype=torch.bool))

    def test_with_device(self, sample_data: SampleDataType) -> None:
        float_list = sample_data[0]
        device = torch.device("cpu")
        tensor = to_tensor(float_list, device=device)
        assert tensor.device == device

    def test_empty_list(self) -> None:
        tensor = to_tensor([])
        assert tensor.numel() == 0


class TestStackTensor:
    @pytest.fixture
    def sample_tensors(self) -> List[torch.Tensor]:
        return [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5, 6]),
        ]

    def test_basic_stacking(self, sample_tensors: List[torch.Tensor]) -> None:
        stacked = stack_tensor(sample_tensors)
        expected = torch.stack(sample_tensors)
        assert torch.all(stacked == expected)
        assert stacked.dtype == torch.float32

    def test_dtype_conversion(self, sample_tensors: List[torch.Tensor]) -> None:
        stacked = stack_tensor(sample_tensors, dtype=torch.int64)
        assert stacked.dtype == torch.int64

    def test_with_device(self, sample_tensors: List[torch.Tensor]) -> None:
        device = torch.device("cpu")
        stacked = stack_tensor(sample_tensors, device=device)
        assert stacked.device == device

    def test_empty_list(self) -> None:
        with pytest.raises(RuntimeError):
            stack_tensor([])


class SimpleNetwork(nn.Module):
    """A simple neural network for testing parameter updates."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestNetworkUpdates:
    """Test suite for network update utility functions."""

    @pytest.fixture
    def networks(self) -> Tuple[nn.Module, nn.Module]:
        """Fixture providing source and target networks."""
        return SimpleNetwork(), SimpleNetwork()

    class TestSoftUpdate:
        def test_basic(self, networks: Tuple[nn.Module, nn.Module]) -> None:
            """Test soft parameter update between networks."""
            source_net, target_net = networks

            # Store initial parameters
            initial_target_params = [param.clone() for param in target_net.parameters()]
            source_params = [param.clone() for param in source_net.parameters()]

            # Perform soft update
            tau = 0.5
            soft_update(source_net, target_net, tau=tau)

            # Check if parameters were updated correctly
            for initial_target, current_target, source in zip(
                initial_target_params, target_net.parameters(), source_params
            ):
                expected = tau * source + (1.0 - tau) * initial_target
                assert torch.allclose(current_target, expected)

        def test_incorrect_nets(self) -> None:
            """Test with different source, target networks."""
            source_net = nn.Linear(5, 3)
            target_net = nn.Linear(4, 2)

            with pytest.raises(RuntimeError):
                soft_update(source_net, target_net)

    class TestHardUpdate:
        def test_basic(self, networks: Tuple[nn.Module, nn.Module]) -> None:
            source_net, target_net = networks

            # Perform hard update
            hard_update(source_net, target_net)

            # Check if parameters match exactly
            for target_param, source_param in zip(
                target_net.parameters(), source_net.parameters()
            ):
                assert torch.all(target_param == source_param)

        def test_incorrect_nets(self) -> None:
            """Test with different source, target networks."""
            source_net = nn.Linear(5, 3)
            target_net = nn.Linear(4, 2)

            with pytest.raises(RuntimeError):
                hard_update(source_net, target_net)
