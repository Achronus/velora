# Project Roadmap

We've recently changed the direction of Velora, moving from a global RL framework that uses LNNs with all existing RL algorithms, to one that focuses on a custom architecture built up of unique RL techniques.

Our goal is to create a completely autonomous system that learns from it's environment without human intervention. We want to help advance the field forward into the [Era of Experience [:material-arrow-right-bottom:]](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf), specifically, in the cyber domain.

Cyber threats are never ending and always changing so it's easy to get overwhelmed by them. Autonomy is needed. Thus, we are working on a custom solution - `NeuroFlow`.

You'll find it's details in the roadmap below along with additions we are planning. We've broken them down into different sections with checkboxes. Items that are <span style="color: #0FBB68;">green</span> are already implemented.

We've got a lot planned for and are excited for the future of Autonomous Cyber Defense.

## Road to 1.0 Release

So far, we've got a solid foundation for `Velora` and our `NeuroFlow` agents but there is still much to do!

Here's our plans and progress so far:

<div class="grid cards" markdown>

- :material-robot: Main Components/Modules

    ---

    - [x] [Liquid Neural Networks (CfC)](https://www.nature.com/articles/s42256-022-00556-7) - 2022
    - [x] [SAC: Continuous](https://arxiv.org/abs/1801.01290) - 2018
    - [x] [SAC: Discrete](https://arxiv.org/abs/1910.07207) - 2019
    - [x] [SAC: Automatic Entropy](https://arxiv.org/abs/1812.05905) - 2018
    - [x] Replay Buffer
    - [x] [Small Actor, Large Critics: Honey, I Shrunk the Actor](https://arxiv.org/abs/2102.11893) - 2021
    - [ ] [CAT-SAC: SAC with Curiosity-Aware Entropy Temperature](https://openreview.net/forum?id=paE8yL0aKHo) - 2020
    - [ ] [PlaNet: Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551) - 2018
    - [ ] [EWC: Overcoming Catastrophic Forgetting in Neural Networks](https://arxiv.org/abs/1612.00796) - 2016

</div>

<div class="grid cards" markdown>

- :material-cube-outline: Custom Components/Modules

    ---

    - [x] Liquid NCP Actor, NCP Critics
    - [ ] Strategy Library
    - [ ] Adaptive Network using fitness score

</div>

<div class="grid cards" markdown>

- :material-cog: Utility

    ---

    - [x] Setting seed and device
    - [x] Gymnasium environment search and wrappers
    - [x] Saving and Loading models & buffers
    - [x] Training and predicting with models
    - [x] Agent performance tracking (offline & online)
    - [x] Early stopping and checkpoint save system
    - [x] Recording episode performance

</div>

<div class="grid cards" markdown>

- :simple-framework: Simulation Environments

    ---

    - [x] [Gymnasium](https://gymnasium.farama.org/)
    - [ ] [PettingZoo](https://pettingzoo.farama.org/index.html)
    - [ ] [CybORG - Customized](https://github.com/cage-challenge/CybORG)

</div>
