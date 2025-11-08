import jax

from flax import nnx

from flax.typing import Initializer

from velora.constants import DEFAULT_BIAS_INIT, DEFAULT_HIDDEN_INIT


class SparseLinear(nnx.Module):
    """A linear layer with sparsely weighted connections.

    Equation:
    $$
    y = x * (w * m) + b
    $$

    Parameters:
        in_features (int): number of input features
        out_features (int): number of output features
        mask (jax.Array): sparsity mask (m) tensor of shape
            `(out_features, in_features)`
        rngs (flax.nnx.Rngs, optional): random number generator key.
            Must have a `params=[value]` attribute
        hidden_init (flax.nnx.nn.initializers, optional): initializer function for the weight
            matrix. Default is `lecun_uniform()`
        bias_init (flax.nnx.nn.initializers, optional): initializer function for the
            bias. Default is `zeros_init()`
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask: jax.Array,
        *,
        rngs: nnx.Rngs = nnx.Rngs(params=0),
        hidden_init: Initializer = DEFAULT_HIDDEN_INIT,
        bias_init: Initializer = DEFAULT_BIAS_INIT,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.mask = mask
        self.kernel_init = hidden_init
        self.bias_init = bias_init

        weight_key = rngs.params()
        weights = hidden_init(weight_key, (in_features, out_features))
        self.weights = nnx.Param(weights * self.mask)

        bias_key = rngs.params()
        self.bias = nnx.Param(bias_init(bias_key, (out_features,)))

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Applies a linear transformation to the inputs along the last dimension.

        Parameters:
            x (jax.Array): the array to transform with shape `(..., in_features)`

        Returns:
            y_pred (jax.Array): the layer prediction with sparsity applied. Has shape `(..., out_features)`
        """
        return x * (self.weights * self.mask) + self.bias
