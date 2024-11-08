from velora.enums import RenderMode


class TestEnums:
    @staticmethod
    def test_render_mode():
        checks = [
            RenderMode.HUMAN == "human",
            RenderMode.RGB_ARRAY == "rgb_array",
            RenderMode.ANSI == "ANSI",
            RenderMode.RGB_ARRAY_LIST == "rgb_array_list",
            RenderMode.ANSI_LIST == "ansi_list",
        ]
        assert all(checks)
