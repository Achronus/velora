import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")


def plot_state_values(V: np.ndarray, shape: tuple[int, int]) -> None:
    V = np.reshape(V, shape)

    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(V, cmap="cool")

    for (j, i), label in np.ndenumerate(V):
        ax.text(i, j, np.round(label, 3), ha="center", va="center", fontsize=14)

    plt.tick_params(bottom="off", left="off", labelbottom="off", labelleft="off")
    plt.title("State-Value Function")
    plt.show()
