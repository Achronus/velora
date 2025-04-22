# Entropy Modules

Entropy modules are an extension to the SAC algorithm that are used for automatic tuning.

The layout of the modules are identical but their underlying functionality differs to handle their respective use cases.

The only method differences are the required parameters for the `compute_loss` method.

Entropy modules are a wrapper over the top of PyTorch functionality and are made up of the following components:

| Attribute | Description | PyTorch Item |
| --------- | ----------- | ------------ |
| `target` | The target entropy value. | `float` or `torch.Tensor` |
| `log_alpha` | A tunable parameter. | `torch.nn.Parameter` |
| `alpha` | The current entropy coefficient. | `torch.Tensor` |
| `optim` | The entropy optimizer. | `torch.optim.Optimizer` |

## Discrete

???+ api "API Docs"

    [`velora.models.nf.modules.EntropyModuleDiscrete`](../../reference/models/modules.md#velora.models.nf.modules.EntropyModuleDiscrete)

For `discrete` action spaces, we use the `EntropyModuleDiscrete` class.

This accepts the following parameters:

| Parameter | Description            | Default            |
| --------- | ---------------------- | ------------------ |
| `action_dim` | The dimension of the action space. | - |
| `initial_alpha` | The starting entropy coefficient value. | `1.0` |
| `optim` | The PyTorch optimizer. | `torch.optim.Adam` |
| `lr` | The optimizer learning rate. | `0.0003` |
| `device` | The device to perform computations on. E.g., `cpu` or `cuda:0`. | `None` |

## Continuous

???+ api "API Docs"

    [`velora.models.nf.modules.EntropyModule`](../../reference/models/modules.md#velora.models.nf.modules.EntropyModule)

For `continuous` action spaces, we use the `EntropyModule` class.

The parameters are the same as the `EntropyModuleDiscrete` class.

## Compute Loss

To compute the module loss, we use the `compute_loss` method:

```python
# EntropyModuleDiscrete
entropy_loss = entropy.compute_loss(actor_probs, actor_log_probs)

# EntropyModule
entropy_loss = entropy.compute_loss(actor_log_probs)
```

## Updating Gradients

To update the gradients, we use the `gradient_step` method:

```python
entropy.gradient_step(entropy_loss)
```

## Config

To quickly get an overview of the modules parameters we can use the `config` method:

```python
config = entropy.config()
```

This provides us with an `EntropyParameters` config model containing details about the module.

Other modules have their own respective config models that are obtained using their attribute `module.config` instead.

---

Next, we'll dive into working with static `backbones` that Velora offers 👋.
