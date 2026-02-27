from pathlib import Path

import pytest

from venting.cases import CaseConfig, NetworkConfig
from venting.compare import compare_runs, load_run
from venting.presets import get_default_panel_preset_v9
from venting.profiles import Profile
from venting.run import export_case_artifacts, run_case


def _make_run(tmp_path: Path, stem: str, cd_int: float) -> Path:
    prof = Profile("vac", lambda t: 0.0, events=((0.0, "vac"),))
    p = get_default_panel_preset_v9()
    net = NetworkConfig(
        N_chain=2,
        N_par=1,
        V_cell=p.V_cell,
        V_vest=p.V_vest,
        A_wall_cell=p.A_wall_cell,
        A_wall_vest=p.A_wall_vest,
        d_int_mm=2.0,
        n_int_per_interface=1,
        d_exit_mm=2.0,
        n_exit=1,
        Cd_int=cd_int,
        Cd_exit=0.62,
    )
    case = CaseConfig("isothermal", 0.0, 300.0, 0.03, 40)
    res = run_case(net, prof, case)
    out = tmp_path / stem
    out.mkdir()
    export_case_artifacts(out, stem, res, run_params={"x": 1})
    return out


def test_compare_identical(tmp_path: Path):
    run = _make_run(tmp_path, "a", 0.62)
    comp = compare_runs(load_run(run), load_run(run))
    assert all(
        abs(v["delta_max_abs_dP_Pa"]) < 1e-12 for v in comp["edge_deltas"].values()
    )


def test_compare_different_cd(tmp_path: Path):
    run_a = _make_run(tmp_path, "a", 0.62)
    run_b = _make_run(tmp_path, "b", 0.55)
    comp = compare_runs(load_run(run_a), load_run(run_b))
    assert any(abs(v["delta_max_abs_dP_Pa"]) > 0 for v in comp["edge_deltas"].values())


def test_compare_missing_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path / "missing")
