# Critic Modules

Critic modules follow the Critic part of the Actor-Critic architecture. In NF's case, we follow a SAC base with NCP Networks where both variants estimate Q-values using two target networks.

The layout of the modules are identical but their underlying functionality differs to handle their respective use cases.

The only differences are the required parameters for the `predict` and `target_predict` methods.

Critic modules are a wrapper over the top of PyTorch functionality and are made up of the following components:

| Attribute | Description | PyTorch Item |
| --------- | ----------- | ------------ |
| `network1` | The first Critic network. | `torch.nn.Module` |
| `network2` | The second Critic network. | `torch.nn.Module` |
| `target1` | The first Critic's target network. | `torch.nn.Module` |
| `target2` | The second Critic's target network. | `torch.nn.Module` |
| `optim1` | The first Critic network's optimizer. | `torch.optim.Optimizer` |
| `optim1` | The second Critic network's optimizer. | `torch.optim.Optimizer` |

## Discrete

???+ api "API Docs"

    [`velora.models.nf.modules.CriticModuleDiscrete`](../../reference/models/modules.md#velora.models.nf.modules.CriticModuleDiscrete)

    [`velora.models.sac.SACCriticNCPDiscrete`](../../reference/models/sac.md#velora.models.sac.SACCriticNCPDiscrete)

For `discrete` action spaces, we use the `CriticModuleDiscrete` class.

This accepts the following parameters:

| Parameter | Description            | Default            |
| --------- | ---------------------- | ------------------ |
| `state_dim` | The dimension of the state space. | - |
| `n_neurons` | The number of decision/hidden neurons. | - |
| `action_dim` | The dimension of the action space. | - |
| `optim` | The PyTorch optimizer. | `torch.optim.Adam` |
| `lr` | The optimizer learning rate. | `0.0003` |
| `tau` | The soft target network update factor. | `0.0005` |
| `device` | The device to perform computations on. E.g., `cpu` or `cuda:0`. | `None` |

## Continuous

???+ api "API Docs"

    [`velora.models.nf.modules.CriticModule`](../../reference/models/modules.md#velora.models.nf.modules.CriticModule)

    [`velora.models.sac.SACCriticNCP`](../../reference/models/sac.md#velora.models.sac.SACCriticNCP)

For `continuous` action spaces, we use the `CriticModule` class.

The parameters are the same as the `CriticModuleDiscrete` class.

## Target Updates

To update the target networks we use the `update_targets` method:

```python
critic.update_targets()
```

## Updating Gradients

To update the network gradients, we use the `gradient_step` method:

```python
critic.gradient_step(c1_loss, c2_loss)
```

## Prediction

To make a prediction with the Critic networks, we use the `predict` method:

```python
# CriticModuleDiscrete
q1_pred, q2_pred = critic.predict(obs)

# CriticModule
q1_pred, q2_pred = critic.predict(obs, actions)
```

## Target Prediction

To make a prediction with the target networks, we use the `target_predict` method:

```python
# CriticModuleDiscrete
next_q_min = critic.target_predict(obs)

# CriticModule
next_q_min = critic.target_predict(obs, actions)
```

This gives us the smallest next Q-Value prediction between the two target networks (`torch.min(q_values1, q_values2)`).

---

Next, we'll look at the `entropy` modules! 🚀
