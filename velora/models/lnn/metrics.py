from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

from pydantic import BaseModel

if TYPE_CHECKING:
    from velora.models.lnn.ncp import LiquidNCPNetwork

from velora.models.lnn.sparse import SparseLinear


@dataclass
class GradientMetric:
    """
    A storage container for a single set of gradient metrics for an NCP Network.

    Attributes:
        inter (float): average value for the inter layer
        command (float): average value for the command layer
        motor (float): average value for the motor layer
        overall (float): average value for the whole network
    """

    inter: float
    command: float
    motor: float
    overall: float


class NCPGradientMetrics(BaseModel):
    """
    Gradient metric storage for the NCP Network.

    Attributes:
        grad_norm: the raw magnitude (length) of the gradient vector.
            Informs us how large the gradients are in absolute terms.
            Note: doesn't account for the scale of the parameters.

            When -

            - Very large value (e.g., `>10`) - indication of exploding gradients
            - Very small value (e.g., `<0.0001`)- indication of vanishing gradients

        grad_ratio: gradient-to-parameter ratio. Magnitude of the weights.
            Defines the relative size of the gradient compared to the parameter.
            Note: normalizes gradients by parameter size.

            When -

            - Ratios `>>1` (e.g., `>10`) - strong indicator of exploding gradients
            - Ratios `<<1` (e.g., `<0.001`) - strong indicator of vanishing gradients
            - Healthy - Often between `[0.01, 1]`
    """

    grad_norm: GradientMetric
    grad_ratio: GradientMetric

    def to_dict(self, prefix: str) -> Dict[str, float]:
        """
        Converts storage to a dictionary for effective logging.

        Parameters:
            prefix (str): a prefix for metric names (e.g., 'actor' or 'critic')

        Returns:
            metrics (Dict[str, float]): metrics as name-value pairs. 8 in total -

            - `<prefix>.grad_norm.inter`
            - `<prefix>.grad_norm.command`
            - `<prefix>.grad_norm.motor`
            - `<prefix>.grad_norm.overall`
            - `<prefix>.grad_ratio.inter`
            - `<prefix>.grad_ratio.command`
            - `<prefix>.grad_ratio.motor`
            - `<prefix>.grad_ratio.overall`

        """
        return {
            f"{prefix}.grad_norm.inter": self.grad_norm.inter,
            f"{prefix}.grad_norm.command": self.grad_norm.command,
            f"{prefix}.grad_norm.motor": self.grad_norm.motor,
            f"{prefix}.grad_norm.overall": self.grad_norm.overall,
            f"{prefix}.grad_ratio.inter": self.grad_ratio.inter,
            f"{prefix}.grad_ratio.command": self.grad_ratio.command,
            f"{prefix}.grad_ratio.motor": self.grad_ratio.motor,
            f"{prefix}.grad_ratio.overall": self.grad_ratio.overall,
        }


class NCPGradientUtils:
    """
    A helper class for computing the gradient metrics for NCP networks.

    Useful for identifying vanishing/exploding gradients.
    """

    def __init__(self, ncp: "LiquidNCPNetwork") -> None:
        """
        Parameters:
            ncp (LiquidNCPNetwork): the network to use
        """
        self.ncp = ncp
        self.head_names = self._get_head_names()

        self.grad_norms = {
            "inter": {"total": 0.0, "count": 0},
            "command": {"total": 0.0, "count": 0},
            "motor": {"total": 0.0, "count": 0},
        }
        self.grad_ratios = {
            "inter": {"total": 0.0, "count": 0},
            "command": {"total": 0.0, "count": 0},
            "motor": {"total": 0.0, "count": 0},
        }

    def _get_head_names(self) -> List[str]:
        """
        Extracts the head names from a single NCP layer.

        Returns:
            names (List[str]): a list of NCP cell head names.
        """
        return [
            name
            for name, layer in list(self.ncp.layers["inter"].named_children())
            if isinstance(layer, SparseLinear)
        ]

    def _update_grads(self) -> None:
        """
        Updates the gradient dictionaries by computing the gradients for each layer.
        """
        for cell_name, cell in self.ncp.layers.items():
            for head_name in self.head_names:
                head: SparseLinear = getattr(cell, head_name)

                if head.weight.grad is not None:
                    grad_norm = head.weight.grad.norm().item()
                    param_norm = head.weight.data.norm().item()

                    self.grad_norms[cell_name]["total"] += grad_norm
                    self.grad_norms[cell_name]["count"] += 1

                    # Calculate grad-to-param ratio (vanishing/exploding indicator)
                    if param_norm > 1e-8:
                        ratio = grad_norm / param_norm
                        self.grad_ratios[cell_name]["total"] += ratio
                        self.grad_ratios[cell_name]["count"] += 1

    def compute(self) -> NCPGradientMetrics:
        """
        Computes the gradient metrics for the network.

        Returns:
            metrics (NCPGradientMetrics): gradient norms and ratios for the network.
        """
        self._update_grads()

        # Calculate averages for each cell
        norms = {
            cell: stats["total"] / max(stats["count"], 1)
            for cell, stats in self.grad_norms.items()
        }
        ratios = {
            cell: stats["total"] / max(stats["count"], 1)
            for cell, stats in self.grad_ratios.items()
        }

        # Calculate overall averages
        total_norm = sum(data["total"] for data in self.grad_norms.values())
        total_norm_count = sum(data["count"] for data in self.grad_norms.values())
        norms["overall"] = total_norm / max(total_norm_count, 1)

        total_ratio = sum(data["total"] for data in self.grad_ratios.values())
        total_ratio_count = sum(data["count"] for data in self.grad_ratios.values())
        ratios["overall"] = total_ratio / max(total_ratio_count, 1)

        return NCPGradientMetrics(
            grad_norm=GradientMetric(**norms),
            grad_ratio=GradientMetric(**ratios),
        )
