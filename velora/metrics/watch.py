import gymnasium as gym

from velora.gym.wrap import add_core_env_wrappers
from velora.models.base import RLAgent


def watch_notebook(model: RLAgent, env: gym.Env, *, steps: int = 1000) -> None:
    """
    Watch a trained agent perform in an environment.

    Only compatible with Notebooks.

    Parameters:
        model (RLAgent): the agent to watch
        env (gymnasium.Env): the Gymnasium environment to use
        steps (int, optional): the number of steps to watch
    """
    import matplotlib.pyplot as plt
    from IPython.display import clear_output

    env = gym.make(env.spec.name, render_mode="rgb_array")
    env = add_core_env_wrappers(env, device="cpu")
    state, _ = env.reset()

    episode_over = False
    hidden = None
    total_reward = 0

    for _ in range(steps):
        action, hidden = model.predict(state, hidden)
        state, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated

        if episode_over:
            total_reward: float = info["episode"]["r"].item()
            break

        clear_output(wait=True)
        plt.imshow(env.render())
        plt.show()

    print(f"Total reward: {total_reward}")
    env.close()
