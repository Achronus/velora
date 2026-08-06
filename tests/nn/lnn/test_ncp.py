import pytest
import torch
from torch import nn

from velora.nn.lnn.cell import (
    AdaptiveLiquidCell,
    DecayLiquidCell,
    DeltaErasureLiquidCell,
    NCPLiquidCell,
)
from velora.nn.lnn.ncp import LNN
from velora.nn.lnn.wiring import build_wiring

CELLS = [NCPLiquidCell, DecayLiquidCell, DeltaErasureLiquidCell, AdaptiveLiquidCell]
DECAY_CELLS = [DecayLiquidCell, AdaptiveLiquidCell]
PLAIN_CELLS = [NCPLiquidCell, DeltaErasureLiquidCell]


def feedforward(cell) -> torch.Tensor:
    return cell.g_head.mask[:, : cell.in_features]


def recurrent(cell) -> torch.Tensor:
    return cell.g_head.mask[:, cell.in_features :]


class TestWiringReachesTheNetwork:
    def test_masks_match_build_wiring(self) -> None:
        net = LNN(4, 32, 2, seed=13)
        wiring = build_wiring(4, 32, {"out": 2}, seed=13)

        assert torch.equal(feedforward(net.inter), wiring.inter)
        assert torch.equal(feedforward(net.command), wiring.command)
        assert torch.equal(feedforward(net.motor), wiring.heads["out"])

    def test_command_gets_the_recurrent_mask(self) -> None:
        net = LNN(4, 32, 2, seed=13, recurrent_sparsity=0.5)
        wiring = build_wiring(4, 32, {"out": 2}, seed=13, recurrent_sparsity=0.5)

        assert torch.equal(recurrent(net.command), wiring.recurrent)

    def test_inter_and_motor_stay_dense(self) -> None:
        net = LNN(4, 32, 2, recurrent_sparsity=0.5)

        for cell in (net.inter, net.motor):
            assert recurrent(cell).sum() == recurrent(cell).numel()

    def test_command_recurrence_dense_by_default(self) -> None:
        net = LNN(4, 32, 2)
        assert recurrent(net.command).sum() == recurrent(net.command).numel()

    def test_no_polarity_reaches_the_layers(self) -> None:
        net = LNN(4, 32, 2)

        for cell in (net.inter, net.command, net.motor):
            values = torch.unique(cell.g_head.mask).tolist()
            assert set(values) <= {0.0, 1.0}


class TestCellType:
    def test_defaults_to_adaptive(self) -> None:
        net = LNN(4, 32, 2)
        assert net.cell_type is AdaptiveLiquidCell

        for cell in (net.inter, net.command, net.motor):
            assert isinstance(cell, AdaptiveLiquidCell)

    @pytest.mark.parametrize("cell_type", CELLS)
    def test_every_layer_uses_the_given_type(self, cell_type) -> None:
        net = LNN(4, 32, 2, cell_type=cell_type)

        for cell in (net.inter, net.command, net.motor):
            assert type(cell) is cell_type

    @pytest.mark.parametrize("cell_type", CELLS)
    def test_forward_works_for_every_type(self, cell_type) -> None:
        net = LNN(4, 32, 2, cell_type=cell_type, recurrent_sparsity=0.5)
        preds, hidden = net(torch.randn(3, 5, 4))

        assert preds.shape == (3, 5, 2)
        assert hidden.shape == (3, net.hidden_size)

    @pytest.mark.parametrize("cell_type", CELLS)
    def test_recurrent_mask_reaches_command_layer(self, cell_type) -> None:
        net = LNN(4, 32, 2, seed=13, cell_type=cell_type, recurrent_sparsity=0.5)
        wiring = build_wiring(4, 32, {"out": 2}, seed=13, recurrent_sparsity=0.5)

        assert torch.equal(recurrent(net.command), wiring.recurrent)

    @pytest.mark.parametrize("cell_type", DECAY_CELLS)
    def test_alpha_rank_is_applied_to_decay_cells(self, cell_type) -> None:
        net = LNN(4, 32, 2, cell_type=cell_type, alpha_rank=3)

        for cell in (net.inter, net.command, net.motor):
            assert cell.alpha_rank == 3
            assert cell.alpha_down.out_features == 3

    @pytest.mark.parametrize("cell_type", PLAIN_CELLS)
    def test_alpha_rank_ignored_by_plain_cells(self, cell_type) -> None:
        net = LNN(4, 32, 2, cell_type=cell_type, alpha_rank=3)

        for cell in (net.inter, net.command, net.motor):
            assert not hasattr(cell, "alpha_rank")

    def test_param_counts_differ_between_types(self) -> None:
        counts = {c: LNN(4, 32, 2, cell_type=c).total_params for c in CELLS}
        assert len(set(counts.values())) == len(CELLS)

    @pytest.mark.parametrize("cell_type", [nn.Linear, "AdaptiveLiquidCell", None, 4])
    def test_invalid_cell_type_raises(self, cell_type) -> None:
        with pytest.raises(TypeError, match="NCPLiquidCell"):
            LNN(4, 32, 2, cell_type=cell_type)


class TestForward:
    @pytest.mark.parametrize("recurrent_sparsity", [None, 0.5, 0.9])
    def test_output_shapes(self, recurrent_sparsity: float | None) -> None:
        net = LNN(4, 32, 2, recurrent_sparsity=recurrent_sparsity)
        preds, hidden = net(torch.randn(3, 5, 4))

        assert preds.shape == (3, 5, 2)
        assert hidden.shape == (3, net.hidden_size)

    def test_unbatched_time_dim_is_expanded(self) -> None:
        net = LNN(4, 32, 2)
        preds, _ = net(torch.randn(3, 4))
        assert preds.shape == (3, 1, 2)

    def test_sparse_recurrence_reduces_active_params(self) -> None:
        dense = LNN(4, 64, 2)
        sparse = LNN(4, 64, 2, recurrent_sparsity=0.9)
        assert sparse.active_params < dense.active_params

    def test_gradients_flow(self) -> None:
        net = LNN(4, 32, 2, recurrent_sparsity=0.5)
        preds, _ = net(torch.randn(2, 3, 4))
        preds.sum().backward()

        grads = [p.grad for p in net.parameters() if p.requires_grad]
        assert all(g is not None for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)