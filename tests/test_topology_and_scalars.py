import numpy as np

from venting.cases import CaseConfig, NetworkConfig
from venting.diagnostics import summarize_result
from venting.graph import build_branching_network
from venting.presets import get_default_panel_preset_v9
from venting.profiles import make_profile_step
from venting.solver import solve_case


def _base_cfg() -> NetworkConfig:
    p = get_default_panel_preset_v9()
    return NetworkConfig(
        N_chain=4,
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


def test_scalar_or_list_expansion_compatible():
    prof = make_profile_step(101325.0, 0.01)
    cfg = _base_cfg()
    nodes_s, edges_s, _ = build_branching_network(cfg, prof)

    cfg_list = NetworkConfig(
        **{
            **cfg.__dict__,
            "V_cell": [cfg.V_cell] * cfg.N_chain,
            "A_wall_cell": [cfg.A_wall_cell] * cfg.N_chain,
            "d_int_mm": [2.0] * cfg.N_chain,
            "n_int_per_interface": [1] * cfg.N_chain,
            "Cd_int": [0.62] * cfg.N_chain,
            "L_int_mm": [0.0] * cfg.N_chain,
            "eps_int_um": [0.0] * cfg.N_chain,
            "K_in_int": [0.5] * cfg.N_chain,
            "K_out_int": [1.0] * cfg.N_chain,
        }
    )
    nodes_l, edges_l, _ = build_branching_network(cfg_list, prof)
    assert len(nodes_s) == len(nodes_l)
    assert len(edges_s) == len(edges_l)


def test_two_chain_shared_vest_symmetry():
    prof = make_profile_step(101325.0, 0.01)
    cfg = _base_cfg()
    cfg = NetworkConfig(
        **{**cfg.__dict__, "topology": "two_chain_shared_vest", "N_chain_b": 4}
    )
    nodes, edges, bcs = build_branching_network(cfg, prof)
    case = CaseConfig("intermediate", 0.0, 300.0, 0.2, 100)
    sol = solve_case(nodes, edges, bcs, case)
    res = summarize_result(nodes, edges, bcs, case, sol)

    # A_cell1 and B_cell1 are symmetric for identical chains
    idx_a1 = next(i for i, n in enumerate(sol.nodes_local) if n.name == "A_cell1")
    idx_b1 = next(i for i, n in enumerate(sol.nodes_local) if n.name == "B_cell1")
    rel = np.max(
        np.abs(res.P[idx_a1] - res.P[idx_b1]) / np.maximum(np.abs(res.P[idx_a1]), 1.0)
    )
    assert float(rel) < 1e-6
