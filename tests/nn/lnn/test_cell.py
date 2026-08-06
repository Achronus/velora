import pytest
import torch

from velora.nn.lnn.cell import (
    AdaptiveLiquidCell,
    DecayLiquidCell,
    DeltaErasureLiquidCell,
    NCPLiquidCell,
)
from velora.nn.lnn.wiring import build_layer_mask

CELLS = [NCPLiquidCell, DecayLiquidCell, DeltaErasureLiquidCell, AdaptiveLiquidCell]


@pytest.fixture
def mask() -> torch.Tensor:
    return build_layer_mask(4, 8, seed=5)


@pytest.mark.parametrize("cell_cls", CELLS)
class TestRecurrentMask:
    def test_dense_by_default(self, cell_cls, mask: torch.Tensor) -> None:
        cell = cell_cls(4, 8, mask)
        recurrent = cell.g_head.mask[:, 4:]
        assert recurrent.sum() == recurrent.numel()

    def test_sparse_mask_is_applied(self, cell_cls, mask: torch.Tensor) -> None:
        recurrent_mask = torch.zeros((8, 8))
        recurrent_mask[0, 0] = 1.0

        cell = cell_cls(4, 8, mask, recurrent_mask=recurrent_mask)
        assert torch.equal(cell.g_head.mask[:, 4:], recurrent_mask)

    def test_feedforward_block_is_preserved(self, cell_cls, mask) -> None:
        cell = cell_cls(4, 8, mask)
        assert torch.equal(cell.g_head.mask[:, :4], mask)

    def test_all_heads_share_the_mask(self, cell_cls, mask: torch.Tensor) -> None:
        recurrent_mask = build_layer_mask(8, 8, seed=9)
        cell = cell_cls(4, 8, mask, recurrent_mask=recurrent_mask)

        heads = [cell.g_head, cell.h_head, cell.f_head_to_g, cell.f_head_to_h]
        for head in heads:
            assert torch.equal(head.mask, cell.g_head.mask)

    def test_wrong_shape_raises(self, cell_cls, mask: torch.Tensor) -> None:
        with pytest.raises(ValueError, match="recurrent_mask"):
            cell_cls(4, 8, mask, recurrent_mask=torch.ones((3, 3)))

    def test_polarity_is_not_applied(self, cell_cls, mask: torch.Tensor) -> None:
        cell = cell_cls(4, 8, mask)
        assert set(torch.unique(cell.g_head.mask).tolist()) <= {0.0, 1.0}

    def test_masked_weights_stay_zero_after_backward(
        self, cell_cls, mask: torch.Tensor
    ) -> None:
        cell = cell_cls(4, 8, mask)
        out, _ = cell(torch.randn(2, 4), torch.zeros(2, 8), torch.ones(1))
        out.sum().backward()

        zeroed = cell.g_head.mask == 0
        assert (cell.g_head.weights.grad[zeroed] == 0).all()


@pytest.mark.parametrize("cell_cls", CELLS)
class TestForward:
    def test_output_shapes(self, cell_cls, mask: torch.Tensor) -> None:
        cell = cell_cls(4, 8, mask)
        out, hidden = cell(torch.randn(3, 4), torch.zeros(3, 8), torch.ones(1))
        assert out.shape == (3, 8)
        assert hidden.shape == (3, 8)

    def test_sparse_recurrence_changes_output(
        self, cell_cls, mask: torch.Tensor
    ) -> None:
        torch.manual_seed(0)
        dense = cell_cls(4, 8, mask)

        torch.manual_seed(0)
        sparse = cell_cls(4, 8, mask, recurrent_mask=torch.zeros((8, 8)))

        x, hidden, ts = torch.randn(2, 4), torch.randn(2, 8), torch.ones(1)
        assert not torch.allclose(dense(x, hidden, ts)[0], sparse(x, hidden, ts)[0])
