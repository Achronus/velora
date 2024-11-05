# Velora

**Velora** is a lightweight and extensible framework built on top of powerful libraries like `Gymnasium` (`Gym`) and `PyTorch`, specializing in model-free reinforcement learning (RL) algorithms. Designed to streamline RL experimentation, Velora offers PyTorch-based implementations of popular algorithms, simplifying the process of training and evaluating agents. Configuration management is enhanced with `Pydantic`, enabling seamless loading of settings from `YAML` files for a consistent, customizable user experience, and tested thoroughly using unit test packages like `pytest`. While Velora currently only supports `Gym` environments through a custom `EnvHandler`, its architecture is adaptable and expandable to other RL libraries.

## API Structure

The frameworks API is designed to be simple and intuitive. It's broken into two main categories: `core` and `extras`.

### Core

The primary building blocks you'll use regularly.

- `from velora import [item]`
- `from velora.env import [item]`
- `from velora.models import [pytorch_model]`

### Extras

Extra items occassionally used under specific conditions.

- `from velora.enums import [enum]`
- `from velora.exc import [error]`
- `from velora.utils import [util_method]`
