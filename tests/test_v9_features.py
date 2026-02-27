import numpy as np

from venting.cases import CaseConfig
from venting.constants import T0
from venting.diagnostics import summarize_result
from venting.flow import mdot_orifice_pos_props
from venting.geometry import circle_area_from_d_mm
from venting.graph import EXT_NODE, CdConst, ExternalBC, GasNode, OrificeEdge
from venting.profiles import Profile
from venting.solver import solve_case
from venting.thermo import gamma_air


def _single_node():
    vol = 131.6e-6
    area = circle_area_from_d_mm(2.0)
    profile = Profile("vacuum", lambda t: 0.0, events=((0.0, "vacuum"),))
    nodes = [GasNode("node", vol, 181.6e-4)]
    edges = [OrificeEdge(0, EXT_NODE, area, CdConst(0.62), label="exit")]
    bcs = [ExternalBC(0, profile, T_ext=T0)]
    return nodes, edges, bcs


def test_variable_thermo_regression_near_300k():
    nodes, edges, bcs = _single_node()
    c_int = CaseConfig("intermediate", 0.0, T0, 0.005, 40)
    c_var = CaseConfig("variable", 0.0, T0, 0.005, 40)
    r_int = summarize_result(
        nodes, edges, bcs, c_int, solve_case(nodes, edges, bcs, c_int)
    )
    r_var = summarize_result(
        nodes, edges, bcs, c_var, solve_case(nodes, edges, bcs, c_var)
    )
    rel = np.max(np.abs(r_var.P[0] - r_int.P[0]) / np.maximum(r_int.P[0], 1.0))
    assert float(rel) <= 0.01


def test_variable_gamma_changes_mdot():
    p_up = 101325.0
    p_dn = 30000.0
    area = np.pi * (0.002**2) / 4.0
    m_cold = mdot_orifice_pos_props(
        p_up, 260.0, p_dn, 0.62, area, gamma=gamma_air(260.0)
    )
    m_hot = mdot_orifice_pos_props(
        p_up, 360.0, p_dn, 0.62, area, gamma=gamma_air(360.0)
    )
    assert abs(m_cold - m_hot) > 0.0


def test_lumped_wall_large_capacity_matches_fixed():
    nodes, edges, bcs = _single_node()
    fixed = CaseConfig("intermediate", 5.0, 300.0, 0.1, 40, wall_model="fixed")
    lumped = CaseConfig(
        "intermediate",
        5.0,
        300.0,
        0.1,
        40,
        wall_model="lumped",
        wall_C_per_area=1e12,
        wall_h_out=0.0,
    )
    r_fixed = summarize_result(
        nodes, edges, bcs, fixed, solve_case(nodes, edges, bcs, fixed)
    )
    r_lumped = summarize_result(
        nodes, edges, bcs, lumped, solve_case(nodes, edges, bcs, lumped)
    )
    rel = np.max(np.abs(r_lumped.P[0] - r_fixed.P[0]) / np.maximum(r_fixed.P[0], 1.0))
    assert float(rel) <= 0.01


def test_h_in_zero_matches_adiabatic():
    nodes, edges, bcs = _single_node()
    c_fixed = CaseConfig("intermediate", 0.0, 300.0, 0.1, 40, wall_model="fixed")
    c_lumped = CaseConfig(
        "intermediate",
        0.0,
        300.0,
        0.1,
        40,
        wall_model="lumped",
        wall_C_per_area=1e5,
        wall_h_out=4.0,
    )
    r_fixed = summarize_result(
        nodes, edges, bcs, c_fixed, solve_case(nodes, edges, bcs, c_fixed)
    )
    r_lumped = summarize_result(
        nodes, edges, bcs, c_lumped, solve_case(nodes, edges, bcs, c_lumped)
    )
    rel = np.max(np.abs(r_lumped.P[0] - r_fixed.P[0]) / np.maximum(r_fixed.P[0], 1.0))
    assert float(rel) <= 0.01


def test_dynamic_pump_smoke():
    nodes, edges, bcs = _single_node()
    case = CaseConfig(
        "intermediate",
        0.0,
        300.0,
        0.1,
        40,
        external_model="dynamic_pump",
        V_ext=0.2,
        T_ext=300.0,
        pump_speed_m3s=0.01,
        P_ult_Pa=10.0,
    )
    sol = solve_case(nodes, edges, bcs, case)
    assert sol.success
    assert np.isfinite(sol.y).all()
