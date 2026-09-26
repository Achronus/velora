# Getting Started

!!! note

    Velora only supports Python 3.12.

## Installation

To get started, install it through [uv [:material-arrow-right-bottom:]](https://docs.astral.sh/uv/) or [pip [:material-arrow-right-bottom:]](https://pip.pypa.io/).

=== "uv"

    Configure the PyTorch index in your project's `pyproject.toml` first, so `torch` resolves to the CUDA 13 wheels:

    ```toml
    [[tool.uv.index]]
    name = "pytorch-cuda"
    url = "https://download.pytorch.org/whl/cu130"
    explicit = true

    [tool.uv.sources]
    torch = { index = "pytorch-cuda" }
    torchvision = { index = "pytorch-cuda" }
    ```

    Then add the package:

    ```bash
    uv add velora
    ```

    !!! tip

        Already ran `uv add velora`? Add the configuration above, then run `uv sync` to re-resolve against the new index.

=== "pip"

    ```bash
    pip install velora
    ```

    This installs PyTorch from PyPI's standard wheels. For the CUDA 13 builds, add the PyTorch `cu130` index:

    ```bash
    pip install velora --extra-index-url https://download.pytorch.org/whl/cu130
    ```

### Simulation (Optional)

Isaac Lab and Isaac Sim are optional and require a NVIDIA GPU. Install them with the `isaac` extra.

=== "uv"

    Add the NVIDIA index to your project's `pyproject.toml` alongside the PyTorch one:

    ```toml
    [[tool.uv.index]]
    name = "nvidia"
    url = "https://pypi.nvidia.com"
    explicit = true

    [tool.uv.sources]
    isaacsim = { index = "nvidia" }
    isaaclab = { index = "nvidia" }
    ```

    Then add the extra:

    ```bash
    uv add "velora[isaac]"
    ```

=== "pip"

    ```bash
    pip install "velora[isaac]" --extra-index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.nvidia.com
    ```


## Next Steps

<div class="grid cards" markdown>

-   :simple-readme:{ .lg .middle } __Tutorials__

    ---

    Learn how to use Velora.

    [:octicons-arrow-right-24: Read more](../tutorials/index.md)

-   :fontawesome-solid-boxes:{ .lg .middle } __API__

    ---

    Explore the API.

    [:octicons-arrow-right-24: Read more](../api/index.md)

</div>
