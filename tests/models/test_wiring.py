from typing import Dict, Literal
import pytest
import torch

from velora.wiring import Wiring, LayerMasks, NeuronCounts

WiringParamsType = Dict[
    Literal["in_features", "n_neurons", "out_features", "sparsity_level"],
    int | float,
]


class TestWiring:
    @pytest.fixture
    def wiring_params(self) -> WiringParamsType:
        return {
            "in_features": 4,
            "n_neurons": 10,
            "out_features": 2,
            "sparsity_level": 0.5,
        }

    @pytest.fixture
    def wiring(self, wiring_params: WiringParamsType) -> Wiring:
        return Wiring(**wiring_params)

    def test_init(self, wiring: Wiring, wiring_params: WiringParamsType):
        # Check basic attributes
        assert wiring.density_level == 1.0 - wiring_params["sparsity_level"]
        assert wiring.n_command == max(int(0.4 * wiring_params["n_neurons"]), 1)
        assert wiring.n_inter == wiring_params["n_neurons"] - wiring.n_command

        # Check neuron counts
        assert isinstance(wiring.counts, NeuronCounts)
        assert wiring.counts.sensory == wiring_params["in_features"]
        assert wiring.counts.inter == wiring.n_inter
        assert wiring.counts.command == wiring.n_command
        assert wiring.counts.motor == wiring_params["out_features"]

        # Check masks initialization
        assert isinstance(wiring.masks, LayerMasks)
        assert wiring.masks.inter.shape == (
            wiring_params["in_features"],
            wiring.n_inter,
        )
        assert wiring.masks.command.shape == (wiring.n_inter, wiring.n_command)
        assert wiring.masks.motor.shape == (
            wiring.n_command,
            wiring_params["out_features"],
        )

    def test_invalid_sparsity_level(self):
        invalid_values = [-0.1, 0.0, 0.05, 0.95, 1.0, 1.5]
        params = {
            "in_features": 4,
            "n_neurons": 10,
            "out_features": 2,
        }

        for sparsity in invalid_values:
            with pytest.raises(ValueError) as excinfo:
                Wiring(**params, sparsity_level=sparsity)
            assert "must be between '[0.1, 0.9]'" in str(excinfo.value)

    def test_synapse_count_correct(self, wiring: Wiring):
        test_cases = [
            (10, 1, max(int(10 * wiring.density_level), 1)),
            (10, 2, max(int(10 * wiring.density_level * 2), 1)),
            (1, 1, 1),  # Should always return at least 1
        ]

        for count, scale, expected in test_cases:
            result = wiring._synapse_count(count, scale)
            assert result == expected
            assert result >= 1  # Should never be less than 1

    def test_polarity(self):
        test_shapes = [(1,), (5,), (3, 4), (2, 3, 4)]

        for shape in test_shapes:
            result = Wiring.polarity(shape)

            # Check shape
            assert result.shape == shape

            # Check values are only -1 or 1
            assert torch.all(torch.abs(result) == 1)

            # Check dtype
            assert result.dtype == torch.int32

    def test_build_connections(self, wiring: Wiring):
        # Test various mask sizes and connection counts
        test_cases = [
            (torch.zeros((3, 5), dtype=torch.int32), 2),
            (torch.zeros((5, 8), dtype=torch.int32), 3),
            (torch.zeros((2, 4), dtype=torch.int32), 1),
        ]

        for mask, count in test_cases:
            result = wiring._build_connections(mask, count)

            # Check shape hasn't changed
            assert result.shape == mask.shape

            # Check values are only -1, 0, or 1
            assert set(result.unique().tolist()).issubset({-1, 0, 1})

            # Check if each column has at least one connection
            assert all(torch.count_nonzero(col) >= 1 for col in result.T)

            # Check each row has at least one connection
            assert all(torch.count_nonzero(row) >= 1 for row in result)

    def test_build_recurrent_connections(self, wiring: Wiring):
        size = 5
        count = 3
        array = torch.zeros((size, size), dtype=torch.int32)

        result = wiring._build_recurrent_connections(array, count)

        # Check shape hasn't changed
        assert result.shape == array.shape

        # Check we have some non-zero connections
        assert torch.count_nonzero(result) > 0

        # Check if values are -1 or 1
        assert set(result.unique().tolist()).issubset({-1, 0, 1})

        # Check that connections are within bounds
        assert torch.count_nonzero(result) <= count

    def test_build_full_network(self, wiring: Wiring):
        # Run build (already called in __init__)
        wiring.build()

        # Check inter layer
        assert torch.count_nonzero(wiring.masks.inter) >= wiring.counts.sensory
        assert all(torch.count_nonzero(col) >= 1 for col in wiring.masks.inter.T)

        # Check command layer
        assert torch.count_nonzero(wiring.masks.command) >= wiring.counts.inter
        assert all(torch.count_nonzero(col) >= 1 for col in wiring.masks.command.T)

        # Check motor layer
        assert torch.count_nonzero(wiring.masks.motor) >= wiring.counts.command
        assert all(torch.count_nonzero(col) >= 1 for col in wiring.masks.motor.T)

    def test_data_method(self, wiring: Wiring):
        masks, counts = wiring.data()

        # Check return types
        assert isinstance(masks, LayerMasks)
        assert isinstance(counts, NeuronCounts)

        # Check if masks match
        assert torch.equal(masks.inter, wiring.masks.inter)
        assert torch.equal(masks.command, wiring.masks.command)
        assert torch.equal(masks.motor, wiring.masks.motor)

        # Check if counts match
        assert counts.sensory == wiring.counts.sensory
        assert counts.inter == wiring.counts.inter
        assert counts.command == wiring.counts.command
        assert counts.motor == wiring.counts.motor

    @pytest.mark.parametrize("sparsity_level", [0.1, 0.5, 0.9])
    def test_different_sparsity_levels(
        self, wiring_params: WiringParamsType, sparsity_level: float
    ):
        wiring_params["sparsity_level"] = sparsity_level
        wiring = Wiring(**wiring_params)

        # Higher sparsity should result in fewer connections
        for row in wiring.masks.inter:
            # Each row should have at least one connection
            assert torch.count_nonzero(row) >= 1

            # With higher sparsity (lower density), should have fewer connections
            if sparsity_level > 0.5:
                assert torch.count_nonzero(row) <= wiring.n_inter

        # Check if all columns have at least one connection
        assert all(torch.count_nonzero(col) >= 1 for col in wiring.masks.inter.T)
