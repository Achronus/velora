import pytest

from velora.analytics.base import NullAnalytics


class TestNullAnalytics:
    @pytest.fixture
    def analytics(self) -> NullAnalytics:
        return NullAnalytics()

    @staticmethod
    def test_init(analytics: NullAnalytics):
        run = analytics.init("test1")
        assert isinstance(run, NullAnalytics)


class TestWeightsAndBiases:
    pass
