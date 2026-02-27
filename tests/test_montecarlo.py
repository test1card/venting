from venting.cases import CaseConfig, NetworkConfig
from venting.montecarlo import run_mc
from venting.presets import get_default_panel_preset_v9
from venting.profiles import Profile


def _base_net() -> NetworkConfig:
    p = get_default_panel_preset_v9()
    return NetworkConfig(
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
        Cd_int=0.62,
        Cd_exit=0.62,
    )


def _case() -> CaseConfig:
    return CaseConfig("isothermal", 0.0, 300.0, 0.05, 40)


def test_mc_deterministic():
    prof = Profile("vac", lambda t: 0.0, events=((0.0, "vac"),))
    r1 = run_mc(_base_net(), _case(), prof, (0.5, 0.7), (0.55, 0.65), 5, seed=42)
    r2 = run_mc(_base_net(), _case(), prof, (0.5, 0.7), (0.55, 0.65), 5, seed=42)
    assert r1["samples"] == r2["samples"]


def test_mc_zero_range():
    prof = Profile("vac", lambda t: 0.0, events=((0.0, "vac"),))
    r = run_mc(_base_net(), _case(), prof, (0.62, 0.62), (0.62, 0.62), 5, seed=1)
    for edge in r["summary"].values():
        assert abs(edge["std"]) < 1e-12


def test_mc_has_spread():
    prof = Profile("vac", lambda t: 0.0, events=((0.0, "vac"),))
    r = run_mc(_base_net(), _case(), prof, (0.5, 0.7), (0.55, 0.65), 5, seed=1)
    assert any(edge["std"] > 0 for edge in r["summary"].values())
