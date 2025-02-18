# Brief History & Motivation

During the LLM boom of 2023, AI became larger and larger. It was framed that scale was the solution to better, more powerful AI models. This required more money, more energy, and less explainability.

For two years LLMs continued to reign supreme without any hope of slowing down. Fed up with scale being the only solution, we set our sights on a different one, a better one. One with explainability in mind, and thus, the idea of Velora was born.

We focus on RL because of how fascinating the topic is. Agents that learn through trial-and-error, just like humans do. Does that not sound like true Artificial Intelligence to you? Our only problem was the size of models.

One fateful day, browsing our YouTube feed, a hidden gem presented itself - *[Liquid Neural Networks [:material-arrow-right-bottom:]](https://www.youtube.com/watch?v=IlliqYiRhMU)*.

An architecture that is powerful, small, robust and explainable. Yet, no one was talking about it. Have people not seen the architecture? Are they blinded by the power of LLMs? We couldn't put our finger on it. Months went by and LLMs still boomed, achieving great results but still growing in size without an end in sight, until finally we had enough.

For 2 weeks, we studied LNNs, rebuilding the architecture from scratch. Understanding all the nooks and crannies of it and finally incorporated it into our first RL algorithm - DDPG. We started it up using the [Inverted Pendulum Gymnasium environment [:material-arrow-right-bottom:]](https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/) and waited... it didn't work. The critic refused to learn. We then tried an official implementation and the same thing happened. Dreams shattered... Or so we thought.

Determined to make it work, we decomposed the problem:

- Sparse linear layers on their own - success :partying_face:
- LTC cell without recurrence - fail :face_with_symbols_over_mouth:

We found our problem. Only the hidden state was being returned. There was no cell prediction! A simple projection layer from input size to output size + a residual connection to the hidden state and boom 💥. The algorithm sprung to life.

=== "Old LTC Cell"

    ```python
    #...
    
    new_hidden = self._new_hidden(x, g_out, h_out)
    return new_hidden, new_hidden
    ```

=== "New LTC Cell"

    ```python hl_lines="6"
    self.proj = nn.Linear(self.head_size, n_hidden, device=device)

    # ...

    new_hidden = self._new_hidden(x, g_out, h_out)  # (1)
    y_pred = self._sparse_head(x, self.proj) + new_hidden  # (2)
    return y_pred, new_hidden
    ```

    1. Old return - hidden state only.
    2. Residual connection - projection + hidden state.

With this change, we had a fully functioning Liquid DDPG.

Now to test it again - 3 layers, 20 'hidden' (decision) neurons, all sparsely connected, randomly wired in a specific way, trained for 500 episodes - environment complete :partying_face:.

This was it, the first real step on our journey to making Velora a reality.

That was on the 17th February 2025 and marks the true birth 🎂 of Velora, our Liquid RL framework.

At the time of writing, this is only a day after its birth 🤭. We have big plans for Velora and hope that LNNs meet our expectations and help us understand these 5 pillars:

- **Causality** - how changes in parameters and elements (e.g., NN building blocks) of the agent alter the decision making process.
- **Fairness** - the decisions made by the agent are not biased and are independent of a selected group of sensitive features (e.g., gender, ethnicity, image backgrounds).
- **Robustness and Reliability** - the agent is effective under input or parameter perturbations (e.g., noise).
- **Usability** - the model is simple to use for accomplishing a task.
- **Trust** - the user has high confidence when applying the model in production.

We've got a long road ahead but we are excited to make it happen! 🚀

We hope you all enjoy using Velora as much as we do making it and we look forward to seeing what you all create with it! 🍻
