# Project Roadmap

We've got a lot planned for Velora and are interested to see how far Liquid RL can take us.

RL is a huge field and we can only cover a small sample of it. Our goal is to adapt normal RL algorithms into Liquid ones and have additional extensions for things like Curiosity for custom experiments.

So far, we have a set of algorithms and extensions in mind with other options to consider. For the remainder of this roadmap we have a breakdown of different sections with checkboxes. Items that are <span style="color: #0FBB68;">green</span> are already implemented.

## Beta Release

The beta release focuses on the major building blocks of the whole framework. Officially, we will be out of the beta version when we hit `0.1`.

Here's our plans and progress so far:

### 0.0.3

<div class="grid cards" markdown>

- :material-robot: Algorithms

    ---

    - [x] [Liquid Neural Networks (CfC)](https://www.nature.com/articles/s42256-022-00556-7) - 2022
    - [x] [DDPG](https://arxiv.org/abs/1509.02971) - 2015

- :fontawesome-solid-cube: Functionality

    ---

    - [x] Replay and Rollout Buffers
    - [x] Saving and Loading models & buffers
    - [x] Training and predicting with models

</div>

<div class="grid cards" markdown>

- :material-cog: Utility

    ---

    - [x] Setting seed and device
    - [x] Gymnasium environment search and wrappers
    - [x] ✨ MORE DOCUMENTATION! ✨

</div>

### 0.1

Remaining items needed to push out of the beta version.

<div class="grid cards" markdown>

- :fontawesome-solid-cube: Functionality

    ---

    - [ ] Agent performance tracking
    - [x] Early stopping and checkpoint save system
    - [ ] Recording episode performance
    - [ ] DDPG agent examples and benchmarks

</div>

## Road to 1.0 Release

These are items that go beyond the beta release (still including them) that are mandatory for the framework to reach a `1.0` release.

<div class="grid cards" markdown>

- :material-robot: Algorithms

    ---

    - [ ] [TRPO](https://arxiv.org/abs/1502.05477) - 2016
    - [ ] [A2C](https://arxiv.org/abs/1602.01783) - 2016
    - [ ] [PPO](https://arxiv.org/abs/1707.06347) - 2017
    - [ ] [TD3](https://arxiv.org/abs/1802.09477v3) - 2018
    - [ ] [SAC](https://arxiv.org/abs/1801.01290) - 2018

- :fontawesome-solid-cube: Extensions

    ---

    - [ ] [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) - 2016
    - [ ] [Curiosity](https://arxiv.org/abs/1705.05363) - 2017
    - [ ] [HER](https://arxiv.org/abs/1707.01495) - 2018
    - [ ] [Forward-Backward (FB)](https://arxiv.org/abs/1803.10227) - 2018
    - [ ] [Forward Imagination](https://arxiv.org/abs/2110.00188) - 2021

</div>

## 1.0 and Beyond

These are extra items that we are considering for the framework beyond `1.0` release.

<div class="grid cards" markdown>

- :material-robot: Algorithms

    ---

    - [ ] [ACER](https://arxiv.org/abs/1611.01224) - 2016
    - [ ] [ACKTR](https://arxiv.org/abs/1708.05144) - 2017
    - [ ] [DSAC](https://arxiv.org/abs/2004.14547) - 2019
    - [ ] [DeepGait](https://arxiv.org/abs/1909.08399) - 2020
    - [ ] [Meta Motivo](https://metamotivo.metademolab.com/) - 2024
    - [ ] [GRPO](https://arxiv.org/abs/2402.03300) - 2024
    - [ ] [Langevin SAC](https://openreview.net/forum?id=FvQsk3la17) - 2025

- :fontawesome-solid-cube: Extensions

    ---

    - [ ] [Meta Learning](https://arxiv.org/abs/1611.05763) - 2017
    - [ ] [Imitation Learning](https://arxiv.org/abs/2108.04763) - 2021
    - [ ] [Hierarchical RL](https://arxiv.org/abs/2101.06521) - 2021
    - [ ] [Reverse Offline Model-based Imagination (ROMI)](https://arxiv.org/abs/2110.00188) - 2021
    - [ ] [Goal Oriented Forward-Backward](https://arxiv.org/abs/2103.07945) - 2021

</div>

There are many more items to consider on the [OpenAI Spinning-up papers list [:material-arrow-right-bottom:]](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#a-intrinsic-motivation) too that we plan to review in the future.

## Integrations

These are topics to consider for examples and integrations with Velora.

<div class="grid cards" markdown>

- :simple-framework: Simulation Environments

    ---

    - [ ] [ManiSkill](https://www.maniskill.ai/)
    - [ ] [Robosuite](https://robosuite.ai/docs/overview.html)

</div>
