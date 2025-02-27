import pytest
from unittest.mock import Mock, patch

import gymnasium as gym
import torch
import numpy as np

from velora.metrics.watch import watch_notebook
from velora.models.base import RLAgent


class TestWatchNotebookEpisodeOver:
    @pytest.fixture
    def mock_agent(self):
        agent = Mock(spec=RLAgent)
        actions = [
            (torch.tensor([0.0]), torch.zeros(1, 10)),  # stabilizing action
            (torch.tensor([3.0]), torch.zeros(1, 10)),  # extreme action to tip over
        ]
        agent.predict.side_effect = actions * 10  # repeat to ensure we have enough
        return agent

    @patch("IPython.display.clear_output")
    @patch("matplotlib.pyplot.imshow")
    @patch("matplotlib.pyplot.show")
    @patch("builtins.print")
    def test_full_cycle_with_ep_over(
        self, mock_print, mock_show, mock_imshow, mock_clear_output, mock_agent
    ):
        class QuickTerminationWrapper(gym.Wrapper):
            def step(self, action):
                obs, reward, terminated, truncated, info = self.env.step(action)
                if abs(obs[1]) > 0.5:  # Lower threshold for termination
                    terminated = True
                    info["episode"] = {"r": np.sum(reward)}
                return obs, reward, terminated, truncated, info

        env = gym.make("InvertedPendulum-v5")
        wrapped_env = QuickTerminationWrapper(env)

        watch_notebook(mock_agent, wrapped_env)

        # Check that predict was called at least once
        assert mock_agent.predict.call_count >= 1

        # But less than the full 20 steps (should terminate early)
        assert mock_agent.predict.call_count < 20

        # Verify reward was printed
        mock_print.assert_called_once()
        assert "Total reward:" in mock_print.call_args[0][0]

        # Verify visualization was shown correct number of times
        expected_vis_calls = mock_agent.predict.call_count - 1
        assert mock_clear_output.call_count == expected_vis_calls
        assert mock_imshow.call_count == expected_vis_calls
        assert mock_show.call_count == expected_vis_calls
