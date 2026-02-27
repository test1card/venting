from pathlib import Path

from venting.cases import CaseConfig, NetworkConfig
from venting.graph import build_branching_network
from venting.presets import get_default_panel_preset_v9
from venting.profiles import make_profile_linear
from venting.solver import solve_case


def test_solver_does_not_use_identity_only_jac_sparsity_hint():
    src = Path("src/venting/solver.py").read_text(encoding="utf-8")
    assert "jac_sparsity" not in src


def test_large_network_solves_with_radau():
    preset = get_default_panel_preset_v9()
    cfg = NetworkConfig(
        N_chain=20,
        N_par=2,
        V_cell=preset.V_cell,
        V_vest=preset.V_vest,
        A_wall_cell=preset.A_wall_cell,
        A_wall_vest=preset.A_wall_vest,
        d_int_mm=2.0,
        n_int_per_interface=1,
        d_exit_mm=2.0,
        n_exit=1,
        Cd_int=0.62,
        Cd_exit=0.62,
    )
    profile = make_profile_linear(101325.0, 20.0)
    nodes, edges, bcs = build_branching_network(cfg, profile)
    case = CaseConfig("intermediate", 0.0, 300.0, 1.0, 120)
    sol = solve_case(nodes, edges, bcs, case)
    assert sol.success
