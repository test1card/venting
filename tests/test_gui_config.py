import pytest

from venting.gui.config import GuiCaseConfig


def test_gui_config_roundtrip_json():
    cfg = GuiCaseConfig(N_chain=7, profile_kind="step", step_time_s=0.2)
    loaded = GuiCaseConfig.from_json(cfg.to_json())
    assert loaded.N_chain == 7
    assert loaded.profile_kind == "step"
    assert loaded.step_time_s == 0.2


def test_gui_config_validation_rejects_bad_enum():
    cfg = GuiCaseConfig(thermo="bad")
    with pytest.raises(ValueError):
        cfg.validate()


def test_gui_config_validation_rejects_nonpositive():
    cfg = GuiCaseConfig(V_cell_m3=0.0)
    with pytest.raises(ValueError):
        cfg.validate()
