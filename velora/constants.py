from flax.nnx.nn import initializers

DEFAULT_HIDDEN_INIT = initializers.lecun_uniform()
DEFAULT_BIAS_INIT = initializers.zeros_init()
