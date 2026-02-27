import math

import numpy as np

from venting.cases import CaseConfig
from venting.constants import C_CHOKED, C_P, C_V, GAMMA, P0, R_GAS, T0, T_SAFE
from venting.diagnostics import summarize_result
from venting.flow import mdot_orifice_pos
from venting.geometry import circle_area_from_d_mm
from venting.graph import EXT_NODE, CdConst, ExternalBC, GasNode, OrificeEdge
from venting.profiles import Profile
from venting.solver import solve_case


def _single_node_cases():
    V = 131.6e-6
    A = circle_area_from_d_mm(2.0)
    Cd = 0.62
    profile = Profile("vacuum", lambda t: 0.0, events=((0.0, "vacuum"),))
    nodes = [GasNode("node", V, 181.6e-4)]
    edges = [OrificeEdge(0, EXT_NODE, A, CdConst(Cd), label="exit")]
    bcs = [ExternalBC(0, profile, T_ext=T0)]
    alpha = Cd * A * C_CHOKED * math.sqrt(R_GAS * T0) / V
    tau = 1.0 / alpha
    return nodes, edges, bcs, alpha, tau, V, A, Cd


def test_single_node_analytic_match_adiabatic():
    nodes, edges, bcs, alpha, _, _, _, _ = _single_node_cases()
    beta = (GAMMA - 1.0) / 2.0

    def p_adi(t):
        return P0 * (1.0 + beta * alpha * t) ** (-2.0 * GAMMA / (GAMMA - 1.0))

    def t_adi(t):
        return T0 * (1.0 + beta * alpha * t) ** (-2.0)

    case = CaseConfig("intermediate", 0.0, T0, 1.5, 700)
    res = summarize_result(nodes, edges, bcs, case, solve_case(nodes, edges, bcs, case))
    mask = res.P[0, :] > 0.01 * P0
    t = res.t[mask]
    p_err = np.max(
        np.abs(res.P[0, mask] - np.array([p_adi(float(tt)) for tt in t])) / P0
    )
    t_err = np.max(
        np.abs(res.T[0, mask] - np.array([t_adi(float(tt)) for tt in t])) / T0
    )
    assert float(p_err) < 5e-3
    assert float(t_err) < 5e-3


def test_single_node_analytic_match_isothermal_limit():
    nodes, edges, bcs, alpha, _, _, _, _ = _single_node_cases()

    def p_iso(t):
        return P0 * math.exp(-alpha * t)

    case = CaseConfig("intermediate", 1e6, T0, 1.5, 700)
    res = summarize_result(nodes, edges, bcs, case, solve_case(nodes, edges, bcs, case))
    mask = res.P[0, :] > 0.01 * P0
    t = res.t[mask]
    p_err = np.max(
        np.abs(res.P[0, mask] - np.array([p_iso(float(tt)) for tt in t])) / P0
    )
    assert float(p_err) < 5e-3


def test_mass_conservation_single_node():
    nodes, edges, bcs, _, _, V, _, _ = _single_node_cases()
    case = CaseConfig("intermediate", 0.0, T0, 1.5, 700)
    res = summarize_result(nodes, edges, bcs, case, solve_case(nodes, edges, bcs, case))
    m0 = P0 * V / (R_GAS * T0)
    mf = float(res.m[0, -1])
    m_out = m0 - mf
    err = abs((m0 - mf) - m_out) / m0
    assert err < 1e-3


def test_two_node_mass_conservation():
    Vc, Vv = 131.6e-6, 145.3e-6
    A = circle_area_from_d_mm(2.0)
    Cd = 0.62
    profile = Profile("vacuum", lambda t: 0.0, events=((0.0, "vacuum"),))
    nodes = [GasNode("vest", Vv, 181.6e-4), GasNode("cell", Vc, 181.6e-4)]
    edges = [
        OrificeEdge(1, 0, A, CdConst(Cd), label="cell↔vest"),
        OrificeEdge(0, EXT_NODE, A, CdConst(Cd), label="exit"),
    ]
    bcs = [ExternalBC(0, profile, T_ext=T0)]
    case = CaseConfig("intermediate", 0.0, T0, 2.0, 900)
    res = summarize_result(nodes, edges, bcs, case, solve_case(nodes, edges, bcs, case))

    m_total_0 = float(np.sum(res.m[:, 0]))
    m_total_f = float(np.sum(res.m[:, -1]))
    mdot = np.array(
        [
            mdot_orifice_pos(
                float(res.P[0, k]), float(max(res.T[0, k], T_SAFE)), 0.0, Cd, A
            )
            for k in range(len(res.t))
        ],
        dtype=float,
    )
    m_out = float(np.trapz(mdot, res.t))
    err = abs((m_total_f + m_out) - m_total_0) / m_total_0
    assert err < 1e-3

    E0 = float(np.sum(res.m[:, 0] * C_V * res.T[:, 0]))
    Ef = float(np.sum(res.m[:, -1] * C_V * res.T[:, -1]))
    Eout = float(np.trapz(mdot * C_P * np.maximum(res.T[0, :], T_SAFE), res.t))
    err_e = abs((Ef + Eout) - E0) / E0
    assert err_e < 1e-2


def test_monotonic_pressure_when_vacuum():
    nodes, edges, bcs, _, _, _, _, _ = _single_node_cases()
    case = CaseConfig("intermediate", 0.0, T0, 1.5, 700)
    res = summarize_result(nodes, edges, bcs, case, solve_case(nodes, edges, bcs, case))
    d = np.diff(res.P[0, :])
    assert np.max(d) <= 1e-6 * P0
