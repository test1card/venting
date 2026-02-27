import pytest

from venting.gui.config import GuiCaseConfig
from venting.gui.state_layout import infer_layout, infer_layout_from_modes


def test_layout_isothermal():
    cfg = GuiCaseConfig(thermo="isothermal", wall_model="fixed")
    layout = infer_layout(cfg, 4)
    assert layout.m_slice == slice(0, 4)
    assert layout.t_slice is None
    assert layout.tw_slice is None


def test_layout_intermediate_fixed():
    layout = infer_layout_from_modes("intermediate", "fixed", 3)
    assert layout.m_slice == slice(0, 3)
    assert layout.t_slice == slice(3, 6)
    assert layout.tw_slice is None


def test_layout_variable_lumped():
    layout = infer_layout_from_modes("variable", "lumped", 5)
    assert layout.m_slice == slice(0, 5)
    assert layout.t_slice == slice(5, 10)
    assert layout.tw_slice == slice(10, 15)


def test_layout_rejects_nonpositive_node_count():
    with pytest.raises(ValueError):
        infer_layout_from_modes("intermediate", "fixed", 0)
