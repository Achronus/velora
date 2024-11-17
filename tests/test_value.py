import pytest
import torch

from velora.agent.value import V, Q, ValueFunction


@pytest.fixture
def v_function() -> V:
    return V(num_states=5)


@pytest.fixture
def q_function() -> Q:
    return Q(num_states=5, num_actions=3)


class TestValueFunction:
    @staticmethod
    def test_len_v(v_function: V):
        assert len(v_function) == 5

    @staticmethod
    def test_len_q(q_function: Q):
        assert len(q_function) == 15  # 5 states * 3 actions

    @staticmethod
    def test_shape_v(v_function: V):
        assert v_function.shape == (5,)

    @staticmethod
    def test_shape_q(q_function: Q):
        assert q_function.shape == (5, 3)

    @staticmethod
    def test_abstract_methods():
        with pytest.raises(TypeError):
            ValueFunction()


class TestV:
    @staticmethod
    def test_init():
        v = V(num_states=5)

        checks = [
            isinstance(v._values, torch.Tensor),
            v._values.shape == (5,),
            v.shape == (5,),
        ]
        assert all(checks)

    @staticmethod
    def test_device():
        v = V(num_states=5, device="cpu")
        assert str(v._values.device) == "cpu"

    @staticmethod
    def test_update_and_get_value(v_function: V):
        state = 2
        value = 0.5
        v_function.update(state, value)
        assert v_function.get_value(state) == value

    @staticmethod
    def test_multiple_updates(v_function: V):
        updates = {0: 0.1, 2: 0.5, 4: -0.3}

        for state, value in updates.items():
            v_function.update(state, value)

        assert round(v_function.get_value(state), 3) == -0.3

    @staticmethod
    def test_repr(v_function: V):
        assert repr(v_function).startswith("V(s=5, values=")

    @staticmethod
    def test_out_of_bounds_state(v_function: V):
        with pytest.raises(IndexError):
            v_function.update(state=10, value=0.5)

        with pytest.raises(IndexError):
            v_function.get_value(state=10)


class TestQ:
    @staticmethod
    def test_init():
        q = Q(num_states=5, num_actions=3)

        checks = [
            isinstance(q._values, torch.Tensor),
            q._values.shape == (5, 3),
            q.shape == (5, 3),
        ]
        assert all(checks)

    @staticmethod
    def test_device_specification():
        q = Q(num_states=5, num_actions=3, device="cpu")
        assert str(q._values.device) == "cpu"

    @staticmethod
    def test_update_and_get_value(q_function: Q):
        state, action = 2, 1
        value = 0.5
        q_function.update(state, action, value)
        assert q_function.get_value(state, action) == value

    @staticmethod
    def test_multiple_updates(q_function: Q):
        updates = {(0, 0): 0.1, (2, 1): 0.5, (4, 2): -0.3}
        for (state, action), value in updates.items():
            q_function.update(state, action, value)

        assert round(q_function.get_value(state, action), 3) == -0.3

    @staticmethod
    def test_get_state_actions(q_function: Q):
        state = 2
        values = [0.1, 0.2, 0.3]
        for action, value in enumerate(values):
            q_function.update(state, action, value)

        state_actions = q_function.get_state_actions(state)
        assert torch.allclose(state_actions, torch.tensor(values))

    @staticmethod
    def test_repr(q_function: Q):
        assert repr(q_function).startswith("Q(s=5, a=3, values=")

    @staticmethod
    def test_out_of_bounds(q_function: Q):
        with pytest.raises(IndexError):
            q_function.update(state=10, action=0, value=0.5)

        with pytest.raises(IndexError):
            q_function.update(state=0, action=10, value=0.5)

        with pytest.raises(IndexError):
            q_function.get_value(state=10, action=0)

        with pytest.raises(IndexError):
            q_function.get_value(state=0, action=10)
