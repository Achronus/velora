---
hide:
  - navigation
---

<style>
.md-content .md-typeset h1 { display: none; }
</style>

[![Logo](assets/imgs/main.jpg)](index.md)

<p id="slogan" align="center" markdown>

*Velora, a <span style="color: #38e2e2;">Liquid RL</span> framework for <span style="color: #38e2e2;">NeuroFlow</span> agents, empowering <span style="color: #38e2e2;">Autonomous Cyber Defence</span>.*

</p>

---

<div id="quick-links" style="display: flex; justify-content: center; align-items: center; gap: 3rem">
    <a href="/" target="_blank" style="text-align: center;">
        <svg xmlns="http://www.w3.org/2000/svg" height="32" width="28" viewBox="0 0 448 512"><path fill="rgba(255, 255, 255, 0.7)" d="M96 0C43 0 0 43 0 96V416c0 53 43 96 96 96H384h32c17.7 0 32-14.3 32-32s-14.3-32-32-32V384c17.7 0 32-14.3 32-32V32c0-17.7-14.3-32-32-32H384 96zm0 384H352v64H96c-17.7 0-32-14.3-32-32s14.3-32 32-32zm32-240c0-8.8 7.2-16 16-16H336c8.8 0 16 7.2 16 16s-7.2 16-16 16H144c-8.8 0-16-7.2-16-16zm16 48H336c8.8 0 16 7.2 16 16s-7.2 16-16 16H144c-8.8 0-16-7.2-16-16s7.2-16 16-16z"/></svg>
        <p style="color: #fff; margin-top: 5px; margin-bottom: 5px;">Docs</p>
    </a>
    <a href="https://github.com/Achronus/velora/" target="_blank"  style="text-align: center;">
        <svg xmlns="http://www.w3.org/2000/svg" height="32" width="28" viewBox="0 0 640 512"><path fill="rgba(255, 255, 255, 0.7)" d="M392.8 1.2c-17-4.9-34.7 5-39.6 22l-128 448c-4.9 17 5 34.7 22 39.6s34.7-5 39.6-22l128-448c4.9-17-5-34.7-22-39.6zm80.6 120.1c-12.5 12.5-12.5 32.8 0 45.3L562.7 256l-89.4 89.4c-12.5 12.5-12.5 32.8 0 45.3s32.8 12.5 45.3 0l112-112c12.5-12.5 12.5-32.8 0-45.3l-112-112c-12.5-12.5-32.8-12.5-45.3 0zm-306.7 0c-12.5-12.5-32.8-12.5-45.3 0l-112 112c-12.5 12.5-12.5 32.8 0 45.3l112 112c12.5 12.5 32.8 12.5 45.3 0s12.5-32.8 0-45.3L77.3 256l89.4-89.4c12.5-12.5 12.5-32.8 0-45.3z"/></svg>
        <p style="color: #fff; margin-top: 5px; margin-bottom: 5px;">Code</p>
    </a>
</div>

---

**Velora** is a lightweight and modular framework built on top of powerful libraries like [Gymnasium [:material-arrow-right-bottom:]](https://gymnasium.farama.org/) and [PyTorch [:material-arrow-right-bottom:]](https://pytorch.org/). It is home to a new type of RL agent called ***NeuroFlow*** (NF) that specializes in Autonomous Cyber Defence through a novel Deep Reinforcement Learning (RL) approach we call ***Liquid RL***.

## Benefits

- **Explainability**: NF agents use [Liquid Neural Networks [:material-arrow-right-bottom:]](https://arxiv.org/abs/2006.04439) (LNNs) and [Neural Circuit Policies [:material-arrow-right-bottom:]](https://arxiv.org/abs/1803.08554) (NCPs) to model Cyber system dynamics, not just data patterns. Also, they use sparse NCP connections to mimic biological efficiency, enabling clear, interpretable strategies via a labeled Strategy Library.
- **Adaptability**: NF agents dynamically grow their networks using a fitness score, adding more neurons to a backbone only when new Cyber strategies emerge, keeping agents compact and robust.
- **Planning**: NF agents use a Strategy Library and learned environment model to plan strategic sequences for proactive Cyber defense.
- **Always Learning**: using [EWC [:material-arrow-right-bottom:]](https://arxiv.org/abs/1612.00796), NF agents refine existing strategies and learn new ones post-training, adapting to evolving Cyber threats like new attack patterns.
- **Customizable**: NF agents are [PyTorch-based [:material-arrow-right-bottom:]](https://pytorch.org/), designed to be intuitive, easy to use, and modular so you can easily build your own!

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Getting Started**

    ---

    What are you waiting for?!

    [:octicons-arrow-right-24: Get Started](starting/index.md)

-   :material-scale-balance:{ .lg .middle } **Open Source, MIT**

    ---

    Velora is licensed under the MIT License.

    [:octicons-arrow-right-24: License](starting/license.md)

</div>

## Active Development

**Velora** is a tool that is continuously being developed. There's still a lot to do to make it a great framework, such as detailed API documentation, and expanding our NeuroFlow agents.

Our goal is to provide a quality open-source product that works 'out-of-the-box' that everyone can experiment with, and then gradually fix unexpected bugs and introduce more features on the road to a [`v1`](#active-development) release.

<div class="grid cards" markdown>

-   :material-map:{ .lg .middle } **Roadmap**

    ---

    Check out what we have planned for Velora.

    [:octicons-arrow-right-24: Explore](starting/roadmap.md)

</div>
