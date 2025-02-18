# Project Roadmap

We've got a lot planned for Velora and are interested to see how far Liquid RL can take us.

RL is a huge field and we can only cover a small sample of it. Our goal is to adapt normal RL algorithms into Liquid ones and have additional extensions for things like Curiosity for custom experiments.

So far, we have a set of algorithms and extensions in mind with other options to consider. For the remainder of this roadmap we have a breakdown of different sections with checkboxes. Items that are <span style="color: #0FBB68;">green</span> are already implemented.

## Primary

These are items that are mandatory for the framework to reach a `v1` release.

<div class="grid cards" markdown>

- :material-robot: Algorithms

    ---

    - [x] [DDPG](https://arxiv.org/abs/1509.02971) - 2015
    - [ ] [TRPO](https://arxiv.org/abs/1502.05477) - 2016
    - [ ] [A2C](https://arxiv.org/abs/1602.01783) - 2016
    - [ ] [PPO](https://arxiv.org/abs/1707.06347) - 2017
    - [ ] [TD3](https://arxiv.org/abs/1802.09477v3) - 2018
    - [ ] [SAC](https://arxiv.org/abs/1801.01290) - 2018

- :fontawesome-solid-cube: Extensions

    ---

    - [x] Replay Buffer
    - [x] Rollout Buffer
    - [ ] [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) - 2016
    - [ ] [Curiosity](https://arxiv.org/abs/1705.05363) - 2017
    - [ ] [HER](https://arxiv.org/abs/1707.01495) - 2018
    - [ ] [Forward-Backward (FB)](https://arxiv.org/abs/1803.10227) - 2018
    - [ ] [Forward Imagination](https://arxiv.org/abs/2110.00188) - 2021

</div>

## Secondary

These are extra items to consider for the framework beyond `v1` release.

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
