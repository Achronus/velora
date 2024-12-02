import pytest
import torch

from velora.agent.value import VTable, QTable, ValueFunction


@pytest.fixture
def v_function() -> VTable:
    return VTable(num_states=5)


@pytest.fixture
def q_function() -> QTable:
    return QTable(num_states=5, num_actions=3)


class TestValueFunction:
    @staticmethod
    def test_len_v(v_function: VTable):
        assert len(v_function) == 5

    @staticmethod
    def test_len_q(q_function: QTable):
        assert len(q_function) == 15  # 5 states * 3 actions

    @staticmethod
    def test_shape_v(v_function: VTable):
        assert v_function.shape == (5,)

    @staticmethod
    def test_shape_q(q_function: QTable):
        assert q_function.shape == (5, 3)

    @staticmethod
    def test_abstract_methods():
        with pytest.raises(TypeError):
            ValueFunction()


class TestVTable:
    @staticmethod
    def test_init():
        v = VTable(num_states=5)

        checks = [
            isinstance(v._values, torch.Tensor),
            v._values.shape == (5,),
            v.shape == (5,),
        ]
        assert all(checks)

    @staticmethod
    def test_device():
        v = VTable(num_states=5, device="cpu")
        assert str(v._values.device) == "cpu"

    @staticmethod
    def test_update_and_get_value(v_function: VTable):
        state = 2
        value = 0.5
        v_function.update(state, value)
        assert v_function[state] == value

    @staticmethod
    def test_multiple_updates(v_function: VTable):
        updates = {0: 0.1, 2: 0.5, 4: -0.3}

        for state, value in updates.items():
            v_function.update(state, value)

        assert round(v_function[state], 3) == -0.3

    @staticmethod
    def test_get_multiple_values(v_function: VTable):
        state_values = v_function[:2]
        assert state_values.equal(torch.tensor((0.0, 0.0)))

    @staticmethod
    def test_repr(v_function: VTable):
        assert repr(v_function).startswith("V(s=5, values=")

    @staticmethod
    def test_out_of_bounds_state(v_function: VTable):
        with pytest.raises(IndexError):
            v_function.update(state=10, value=0.5)

        with pytest.raises(IndexError):
            v_function[10]


class TestQTable:
    @staticmethod
    def test_init():
        q = QTable(num_states=5, num_actions=3)

        checks = [
            isinstance(q._values, torch.Tensor),
            q._values.shape == (5, 3),
            q.shape == (5, 3),
        ]
        assert all(checks)

    @staticmethod
    def test_device_specification():
        q = QTable(num_states=5, num_actions=3, device="cpu")
        assert str(q._values.device) == "cpu"

    @staticmethod
    def test_update_and_get_value(q_function: QTable):
        state, action = 2, 1
        value = 0.5
        q_function.update(state, action, value)
        assert q_function[(state, action)] == value

    @staticmethod
    def test_multiple_updates(q_function: QTable):
        updates = {(0, 0): 0.1, (2, 1): 0.5, (4, 2): -0.3}
        for (state, action), value in updates.items():
            q_function.update(state, action, value)

        assert round(q_function[(state, action)], 3) == -0.3

    @staticmethod
    def test_get_state_actions(q_function: QTable):
        state = 2
        values = [0.1, 0.2, 0.3]
        for action, value in enumerate(values):
            q_function.update(state, action, value)

        state_actions = q_function[state]
        assert torch.allclose(state_actions, torch.tensor(values))

    @staticmethod
    def test_repr(q_function: QTable):
        assert repr(q_function).startswith("Q(s=5, a=3, values=")

    @staticmethod
    def test_out_of_bounds(q_function: QTable):
        with pytest.raises(IndexError):
            q_function.update(state=10, action=0, value=0.5)

        with pytest.raises(IndexError):
            q_function.update(state=0, action=10, value=0.5)

        with pytest.raises(IndexError):
            q_function[(10, 0)]

        with pytest.raises(IndexError):
            q_function[(0, 10)]

    @staticmethod
    def test_as_state_values(q_function: QTable):
        result = q_function.as_state_values()
        expected = [0.0, 0.0, 0.0, 0.0, 0.0]
        assert result == expected
