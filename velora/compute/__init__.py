from velora.compute.mixflow import fwdrev_grad, fwdrev_value_and_grad
from velora.compute.rl_ops import (
    compute_importance_weights,
    transform_from_2hot,
    transform_to_2hot,
)
from velora.compute.softmax import Softmax
from velora.compute.utils import (
    batched_index,
    categorical_kl_divergence,
    compute_l2_mean_penalty,
)
from velora.compute.vtrace import (
    VTraceOutput,
    compute_vtrace,
    vtrace_td_error_and_advantage,
)

__all__ = [
    "Softmax",
    "fwdrev_grad",
    "fwdrev_value_and_grad",
    "compute_importance_weights",
    "transform_from_2hot",
    "transform_to_2hot",
    "batched_index",
    "categorical_kl_divergence",
    "compute_l2_mean_penalty",
    "compute_vtrace",
    "vtrace_td_error_and_advantage",
    "VTraceOutput",
]
