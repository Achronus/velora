# Actor Modules

Actor modules follow the Actor part of the Actor-Critic architecture. In NF's case, we follow a SAC base with Liquid NCP Networks, so the `continuous` variant uses a Gaussian policy, and the `discrete` variant a Categorical one.

The layout of the modules are identical but their underlying functionality differs to handle their respective use cases.

The only differences are the required `init` parameters and the number of items returned by the `forward` method.

Actor modules are a wrapper over the top of PyTorch functionality and are made up of the following components:

| Attribute | Description | PyTorch Item |
| --------- | ----------- | ------------ |
| `network` | The Actor network. | `torch.nn.Module` |
| `optim` | The Actor's optimizer. | `torch.optim.Optimizer` |

## Discrete

???+ api "API Docs"

    [`velora.models.nf.modules.ActorModuleDiscrete`](../../reference/models/modules.md#velora.models.nf.modules.ActorModuleDiscrete)

    [`velora.models.sac.SACActorDiscrete`](../../reference/models/sac.md#velora.models.sac.SACActorDiscrete)

For `discrete` action spaces, we use the `ActorModuleDiscrete` class.

This accepts the following parameters:

| Parameter | Description            | Default            |
| --------- | ---------------------- | ------------------ |
| `state_dim` | The dimension of the state space. | - |
| `n_neurons` | The number of decision/hidden neurons. | - |
| `action_dim` | The dimension of the action space. | - |
| `optim` | The PyTorch optimizer. | `torch.optim.Adam` |
| `lr` | The optimizer learning rate. | `0.0003` |
| `device` | The device to perform computations on. E.g., `cpu` or `cuda:0`. | `None` |

## Continuous

???+ api "API Docs"

    [`velora.models.nf.modules.ActorModule`](../../reference/models/modules.md#velora.models.nf.modules.ActorModule)

    [`velora.models.sac.SACActor`](../../reference/models/sac.md#velora.models.sac.SACActor)

For `continuous` action spaces, we use the `ActorModule` class.

This accepts the following parameters:

| Parameter | Description            | Default            |
| --------- | ---------------------- | ------------------ |
| `state_dim` | The dimension of the state space. | - |
| `n_neurons` | The number of decision/hidden neurons. | - |
| `action_dim` | The dimension of the action space. | - |
| `action_scale` | The scale factor to map the normalized actions to the environment's action range. | - |
| `action_bias` | The bias/offset to center the normalized actions to the environment's action range. | - |
| `log_std_min` | The minimum log standard deviation of the action distribution. | `-5` |
| `log_std_max` | The maximum log standard deviation of the action distribution. | `2` |
| `optim` | The PyTorch optimizer. | `torch.optim.Adam` |
| `lr` | The optimizer learning rate. | `0.0003` |
| `device` | The device to perform computations on. E.g., `cpu` or `cuda:0`. | `None` |

??? tip "Computing Scale Factors"

    The scale factors are designed to go from normalized actions back to the normal environment's action range. This is fundamental for SAC's training stability.

    To calculate them, we use the following:

    ```python
    action_scale = torch.tensor(
        env.action_space.high - env.action_space.low,
        device=device,
    ) / 2.0
    
    action_bias = torch.tensor(
        env.action_space.high + env.action_space.low,
        device=device,
    ) / 2.0
    ```

## Updating Gradients

To perform a gradient update, we use the `gradient_step` method:

```python
actor.gradient_step(loss)
```

## Prediction

To make a prediction, we use the `predict` method:

```python
action, hidden = actor.predict(obs, hidden)
```

## Forward Pass

For a complete network forward pass used during training, we use the `forward` method:

```python
# ActorModuleDiscrete
actions, probs, log_prob, hidden = actor.forward(obs, hidden)

# ActorModule
actions, log_prob, hidden = actor.forward(obs, hidden)
```

## Training vs. Evaluation Mode

To quickly swap between the networks training and evaluation mode we use the `train_mode` and `eval_mode` methods:

```python
# Evaluate mode active
actor.eval_mode()

# Training mode active
actor.train_mode()
```

---

Next, we'll look at the `critic` modules! 🚀
