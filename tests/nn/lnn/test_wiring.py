import numpy as np
import pytest
import torch

from velora.nn.lnn.wiring import Wiring, build_layer_mask, build_wiring


SPARSITIES = [0.1, 0.25, 0.5, 0.75, 0.9]


class TestBuildLayerMask:
    def test_shape_is_torch_layout(self) -> None:
        mask = build_layer_mask(4, 16)
        assert mask.shape == (16, 4)

    def test_values_are_binary(self) -> None:
        mask = build_layer_mask(8, 32, sparsity=0.5)
        assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}

    def test_dtype_is_float32(self) -> None:
        assert build_layer_mask(4, 8).dtype == torch.float32

    def test_is_contiguous(self) -> None:
        assert build_layer_mask(4, 8).is_contiguous()

    @pytest.mark.parametrize("sparsity", SPARSITIES)
    def test_every_target_has_incoming(self, sparsity: float) -> None:
        mask = build_layer_mask(6, 24, sparsity=sparsity)
        assert (mask.sum(dim=1) > 0).all()

    @pytest.mark.parametrize("sparsity", SPARSITIES)
    def test_every_source_has_outgoing(self, sparsity: float) -> None:
        mask = build_layer_mask(24, 6, sparsity=sparsity)
        assert (mask.sum(dim=0) > 0).all()

    @pytest.mark.parametrize("sparsity", [0.25, 0.5, 0.75])
    def test_density_close_to_expected(self, sparsity: float) -> None:
        rng = np.random.default_rng(0)
        mask = build_layer_mask(200, 200, sparsity=sparsity, rng=rng)
        assert mask.mean().item() == pytest.approx(1.0 - sparsity, abs=0.02)

    def test_same_seed_is_deterministic(self) -> None:
        a = build_layer_mask(8, 16, seed=7)
        b = build_layer_mask(8, 16, seed=7)
        assert torch.equal(a, b)

    def test_different_seed_differs(self) -> None:
        a = build_layer_mask(16, 32, seed=1)
        b = build_layer_mask(16, 32, seed=2)
        assert not torch.equal(a, b)

    def test_shared_rng_advances_between_calls(self) -> None:
        rng = np.random.default_rng(3)
        a = build_layer_mask(16, 32, rng=rng)
        b = build_layer_mask(16, 32, rng=rng)
        assert not torch.equal(a, b)

    def test_single_neuron_layers(self) -> None:
        mask = build_layer_mask(1, 1, sparsity=0.9)
        assert mask.shape == (1, 1)
        assert mask.item() == 1.0

    @pytest.mark.parametrize("sparsity", [0.0, 0.05, 0.95, 1.0, -0.1])
    def test_invalid_sparsity_raises(self, sparsity: float) -> None:
        with pytest.raises(ValueError, match="sparsity"):
            build_layer_mask(4, 8, sparsity=sparsity)


class TestBuildWiring:
    def test_returns_wiring(self) -> None:
        assert isinstance(build_wiring(4, 16, {"out": 2}), Wiring)

    def test_layer_shapes(self) -> None:
        wiring = build_wiring(4, 16, {"out": 2})
        assert wiring.inter.shape == (10, 4)
        assert wiring.command.shape == (6, 10)
        assert wiring.recurrent.shape == (6, 6)
        assert wiring.heads["out"].shape == (2, 6)

    def test_multiple_heads(self) -> None:
        wiring = build_wiring(4, 16, {"pi": 8, "y": 1})
        assert list(wiring.heads) == ["pi", "y"]
        assert wiring.heads["pi"].shape == (8, 6)
        assert wiring.heads["y"].shape == (1, 6)

    def test_heads_are_independent(self) -> None:
        wiring = build_wiring(6, 32, {"a": 16, "b": 16})
        assert not torch.equal(wiring.heads["a"], wiring.heads["b"])

    @pytest.mark.parametrize("sparsity", SPARSITIES)
    def test_all_masks_binary(self, sparsity: float) -> None:
        wiring = build_wiring(6, 32, {"out": 4}, sparsity=sparsity)
        for mask in (wiring.inter, wiring.command, *wiring.heads.values()):
            assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}

    @pytest.mark.parametrize("sparsity", SPARSITIES)
    def test_no_isolated_neurons(self, sparsity: float) -> None:
        wiring = build_wiring(6, 32, {"out": 4}, sparsity=sparsity)
        for mask in (wiring.inter, wiring.command, *wiring.heads.values()):
            assert (mask.sum(dim=1) > 0).all()
            assert (mask.sum(dim=0) > 0).all()

    def test_head_density_matches_other_layers(self) -> None:
        wiring = build_wiring(64, 400, {"out": 200}, sparsity=0.5)
        assert wiring.heads["out"].mean().item() == pytest.approx(0.5, abs=0.05)

    def test_recurrent_dense_by_default(self) -> None:
        wiring = build_wiring(4, 32, {"out": 2})
        assert wiring.recurrent.sum() == wiring.recurrent.numel()

    def test_recurrent_sparse_when_requested(self) -> None:
        wiring = build_wiring(4, 400, {"out": 2}, recurrent_sparsity=0.9)
        assert wiring.recurrent.mean().item() == pytest.approx(0.1, abs=0.02)

    def test_recurrent_allows_self_loops(self) -> None:
        wiring = build_wiring(4, 400, {"out": 2}, recurrent_sparsity=0.5)
        assert wiring.recurrent.diagonal().sum() > 0

    def test_same_seed_is_deterministic(self) -> None:
        a = build_wiring(4, 16, {"out": 2}, seed=11)
        b = build_wiring(4, 16, {"out": 2}, seed=11)
        assert torch.equal(a.inter, b.inter)
        assert torch.equal(a.command, b.command)
        assert torch.equal(a.recurrent, b.recurrent)
        assert torch.equal(a.heads["out"], b.heads["out"])

    def test_neuron_split_is_forty_percent_command(self) -> None:
        wiring = build_wiring(4, 100, {"out": 2})
        assert wiring.command.shape[0] == 40
        assert wiring.inter.shape[0] == 60

    def test_tiny_network_keeps_one_command_neuron(self) -> None:
        wiring = build_wiring(2, 2, {"out": 1})
        assert wiring.command.shape[0] == 1
        assert wiring.inter.shape[0] == 1

    @pytest.mark.parametrize("sparsity", [0.0, 0.95, 1.0])
    def test_invalid_sparsity_raises(self, sparsity: float) -> None:
        with pytest.raises(ValueError, match="sparsity"):
            build_wiring(4, 16, {"out": 2}, sparsity=sparsity)

    @pytest.mark.parametrize("sparsity", [0.0, 0.95, 1.0])
    def test_invalid_recurrent_sparsity_raises(self, sparsity: float) -> None:
        with pytest.raises(ValueError, match="recurrent_sparsity"):
            build_wiring(4, 16, {"out": 2}, recurrent_sparsity=sparsity)

    def test_empty_heads_raises(self) -> None:
        with pytest.raises(ValueError, match="heads"):
            build_wiring(4, 16, {})
