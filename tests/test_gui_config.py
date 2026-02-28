import dataclasses

import pytest

from venting.gui.config import GuiCaseConfig


def test_gui_config_roundtrip_json():
    cfg = GuiCaseConfig(
        topology="two_chain_shared_vest",
        N_chain=7,
        N_chain_b=5,
        profile_kind="step",
        step_time_s=0.2,
    )
    loaded = GuiCaseConfig.from_json(cfg.to_json())
    assert loaded.topology == "two_chain_shared_vest"
    assert loaded.N_chain == 7
    assert loaded.N_chain_b == 5
    assert loaded.profile_kind == "step"
    assert loaded.step_time_s == 0.2


def test_gui_config_validation_rejects_bad_enum():
    cfg = GuiCaseConfig(thermo="bad")
    with pytest.raises(ValueError):
        cfg.validate()


def test_read_cfg_topology_not_float_converted():
    """Regression: topology='two_chain_shared_vest' must not be passed to float().

    Reproduces the bug where _read_cfg() raised:
      ValueError: could not convert string to float: 'two_chain_shared_vest'
    because 'topology' was missing from the string-exclusion set.
    """
    # Simulate the kwargs dict that Qt widgets would produce (all values as str)
    kwargs = {f.name: str(f.default) for f in dataclasses.fields(GuiCaseConfig)}
    kwargs["topology"] = "two_chain_shared_vest"

    int_fields = {
        "N_chain",
        "N_chain_b",
        "N_par",
        "n_int_per_interface",
        "n_exit",
        "n_pts",
    }
    string_fields = {
        "int_model",
        "exit_model",
        "topology",
        "external_model",
        "profile_kind",
        "profile_pressure_unit",
        "thermo",
        "wall_model",
        "profile_file",
        "output_case_name",
    }
    float_fields = set(kwargs) - int_fields - string_fields

    # Must not raise ValueError
    for k in int_fields:
        int(kwargs[k])
    for k in float_fields:
        float(kwargs[k])

    assert "topology" not in float_fields


def test_gui_config_validation_rejects_nonpositive():
    cfg = GuiCaseConfig(V_cell_m3=0.0)
    with pytest.raises(ValueError):
        cfg.validate()
