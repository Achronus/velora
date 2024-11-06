# Velora

**Velora** is a lightweight and extensible framework built on top of powerful libraries like `Gymnasium` (`Gym`) and `PyTorch`, specializing in model-free reinforcement learning (RL) algorithms. Designed to streamline RL experimentation, Velora offers PyTorch-based implementations of popular algorithms, simplifying the process of training and evaluating agents. Configuration management is enhanced with `Pydantic`, enabling seamless loading of settings from `YAML` files for a consistent, customizable user experience, and tested thoroughly using unit test packages like `pytest`. While Velora currently only supports `Gym` environments through a custom `EnvHandler`, its architecture is adaptable and expandable to other RL libraries.

## API Structure

The frameworks API is designed to be simple and intuitive. It's broken into two main categories: `core` and `extras`.

### Core

The primary building blocks you'll use regularly.

- `from velora import [controller]`
- `from velora.agent import [algorithm], [storage]`
- `from velora.enums import [enum]`

### Extras

Extra items occassionally used under specific conditions.

- `from velora.env import [handler]`
- `from velora.exc import [error]`
- `from velora.utils import [method]`

## Framework Analysis

![Framework Design](/assets/imgs/framework_diagram.png)

Velora's architecture adopts a robust, modular approach to Reinforcement Learning (RL) experimentation, designed for scalability and flexibility.

At the heart of the framework lies the `Controller` class which serves as the central hub for orchestrating the interactions between components.

The `EnvHandler` serves as an abstraction layer for managing various environment frameworks and handing environment interactions. Currently, it only supports [Gymnasium](https://gymnasium.farama.org/) with other options planned for the future.

The `Agent` class handles the agent's behaviour and learning process, managing the algorithm, policy updates, and experiences. It supports `PyTorch` models, optimizers, and loss functions for different strategies and algorithms.

The `Config` module simplifies hyperparameter tuning and setting implementation by reading from a single `YAML` file. By utilising `Pydantic`, settings are validated and applied automatically.

The `Analytics` module handles experiment tracking and logging, storing key performance metrics throughout the training process. Currently, it only supports [Weights and Biases](https://wandb.ai/) with other options planned for the future.
