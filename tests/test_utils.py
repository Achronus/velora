import numpy as np
import pytest
import matplotlib.pyplot as plt

from velora.utils.plots import plot_state_values


class TestPlotStateValues:
    @staticmethod
    def test_basic():
        plt.switch_backend("Agg")
        V = np.array([0.1, 0.2, 0.3, 0.4])
        shape = (2, 2)

        try:
            plot_state_values(V, shape)
        except Exception as e:
            pytest.fail(f"plot_state_values raised an unexpected exception: {e}")

    @staticmethod
    def test_input_types():
        plt.switch_backend("Agg")
        V = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        shape = (2, 2)

        try:
            plot_state_values(V, shape)
        except Exception as e:
            pytest.fail(f"Failed with float32 input: {e}")

    @staticmethod
    def test_shape_mismatch():
        plt.switch_backend("Agg")
        V = np.array([0.1, 0.2, 0.3, 0.4])
        shape = (4, 1)  # Correct total elements, different shape

        try:
            plot_state_values(V, shape)
        except Exception as e:
            pytest.fail(f"Failed with shape mismatch: {e}")

    @staticmethod
    def test_large_array():
        plt.switch_backend("Agg")
        V = np.random.rand(16)
        shape = (4, 4)

        try:
            plot_state_values(V, shape)
        except Exception as e:
            pytest.fail(f"Failed with large random array: {e}")

    @staticmethod
    def test_negative_values():
        plt.switch_backend("Agg")
        V = np.array([-0.1, 0.2, -0.3, 0.4])
        shape = (2, 2)

        try:
            plot_state_values(V, shape)
        except Exception as e:
            pytest.fail(f"Failed with negative values: {e}")

    @staticmethod
    def test_zeros():
        plt.switch_backend("Agg")
        V = np.zeros(4)
        shape = (2, 2)

        try:
            plot_state_values(V, shape)
        except Exception as e:
            pytest.fail(f"Failed with zero values: {e}")
