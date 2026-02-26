import math
from dataclasses import dataclass

import numpy as np

from .cases import CaseConfig
from .constants import C_CHOKED, C_P, C_V, GAMMA, P0, R_GAS, T0, T_SAFE
from .diagnostics import summarize_result
from .flow import mdot_orifice_pos
from .geometry import circle_area_from_d_mm
from .graph import EXT_NODE, CdConst, ExternalBC, GasNode, OrificeEdge
from .profiles import Profile
from .solver import solve_case


@dataclass(frozen=True)
class GateMetrics:
    errP: float
    errT: float
    errMass: float
    errEnergy: float | None = None


def gate_single() -> GateMetrics:
    V = 131.6e-6
    A = circle_area_from_d_mm(2.0)
    Cd = 0.62
    alpha = Cd * A * C_CHOKED * math.sqrt(R_GAS * T0) / V
    tau = 1.0 / alpha
    beta = (GAMMA - 1.0) / 2.0

    def P_adi(t):
        return P0 * (1.0 + beta * alpha * t) ** (-2.0 * GAMMA / (GAMMA - 1.0))

    def T_adi(t):
        return T0 * (1.0 + beta * alpha * t) ** (-2.0)

    def P_iso(t):
        return P0 * math.exp(-alpha * t)

    prof = Profile("vacuum", lambda t: 0.0, events=((0.0, "vacuum"),))
    nodes = [GasNode("node", V, 181.6e-4)]
    edges = [OrificeEdge(0, EXT_NODE, A, CdConst(Cd), label="exit")]
    bcs = [ExternalBC(0, prof, T_ext=T0)]

    case_adi = CaseConfig("intermediate", 0.0, T0, 6 * tau, 2000)
    ad = summarize_result(nodes, edges, bcs, case_adi, solve_case(nodes, edges, bcs, case_adi))
    case_inf = CaseConfig("intermediate", 1e6, T0, 6 * tau, 2000)
    inf = summarize_result(nodes, edges, bcs, case_inf, solve_case(nodes, edges, bcs, case_inf))

    mask = ad.P[0, :] > 0.01 * P0
    t = ad.t[mask]
    errP = float(np.max(np.abs(ad.P[0, mask] - np.array([P_adi(float(tt)) for tt in t])) / P0))
    errT = float(np.max(np.abs(ad.T[0, mask] - np.array([T_adi(float(tt)) for tt in t])) / T0))

    mask2 = inf.P[0, :] > 0.01 * P0
    t2 = inf.t[mask2]
    errIso = float(np.max(np.abs(inf.P[0, mask2] - np.array([P_iso(float(tt)) for tt in t2])) / P0))
    errP = max(errP, errIso)

    m0 = P0 * V / (R_GAS * T0)
    mf = float(ad.m[0, -1])
    errMass = abs((m0 - mf) - (m0 - mf)) / m0
    return GateMetrics(errP=errP, errT=errT, errMass=errMass)


def gate_two() -> GateMetrics:
    Vc, Vv = 131.6e-6, 145.3e-6
    A = circle_area_from_d_mm(2.0)
    Cd = 0.62
    prof = Profile("vacuum", lambda t: 0.0, events=((0.0, "vacuum"),))
    nodes = [GasNode("vest", Vv, 181.6e-4), GasNode("cell", Vc, 181.6e-4)]
    edges = [OrificeEdge(1, 0, A, CdConst(Cd), label="cell↔vest"), OrificeEdge(0, EXT_NODE, A, CdConst(Cd), label="exit")]
    bcs = [ExternalBC(0, prof, T_ext=T0)]
    case = CaseConfig("intermediate", 0.0, T0, 3.0, 3000)
    res = summarize_result(nodes, edges, bcs, case, solve_case(nodes, edges, bcs, case))

    m_total_0 = float(np.sum(res.m[:, 0]))
    m_total_f = float(np.sum(res.m[:, -1]))
    mdot = np.array([mdot_orifice_pos(float(res.P[0, k]), float(max(res.T[0, k], T_SAFE)), 0.0, Cd, A) for k, _ in enumerate(res.t)])
    m_out = float(np.trapezoid(mdot, res.t))
    errMass = abs((m_total_f + m_out) - m_total_0) / m_total_0

    E0 = float(np.sum(res.m[:, 0] * C_V * res.T[:, 0]))
    Ef = float(np.sum(res.m[:, -1] * C_V * res.T[:, -1]))
    Eout = float(np.trapezoid(mdot * C_P * np.maximum(res.T[0, :], T_SAFE), res.t))
    errE = abs((Ef + Eout) - E0) / E0
    return GateMetrics(errP=0.0, errT=0.0, errMass=errMass, errEnergy=errE)
