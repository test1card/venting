from __future__ import annotations

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
    beta = (GAMMA - 1.0) / 2.0

    def p_adi(t: float) -> float:
        return P0 * (1.0 + beta * alpha * t) ** (-2.0 * GAMMA / (GAMMA - 1.0))

    def t_adi(t: float) -> float:
        return T0 * (1.0 + beta * alpha * t) ** (-2.0)

    profile = Profile("vacuum", lambda t: 0.0, events=((0.0, "vacuum"),))
    nodes = [GasNode("node", V, 181.6e-4)]
    edges = [OrificeEdge(0, EXT_NODE, A, CdConst(Cd), label="exit")]
    bcs = [ExternalBC(0, profile, T_ext=T0)]

    case_adi = CaseConfig("intermediate", 0.0, T0, 1.5, 700)
    ad = summarize_result(
        nodes, edges, bcs, case_adi, solve_case(nodes, edges, bcs, case_adi)
    )

    mask = ad.P[0, :] > 0.01 * P0
    t = ad.t[mask]
    errP = float(
        np.max(np.abs(ad.P[0, mask] - np.array([p_adi(float(tt)) for tt in t])) / P0)
    )
    errT = float(
        np.max(np.abs(ad.T[0, mask] - np.array([t_adi(float(tt)) for tt in t])) / T0)
    )

    m0 = P0 * V / (R_GAS * T0)
    mf = float(ad.m[0, -1])
    mdot = np.array(
        [
            mdot_orifice_pos(
                float(ad.P[0, k]), float(max(ad.T[0, k], T_SAFE)), 0.0, Cd, A
            )
            for k in range(len(ad.t))
        ]
    )
    m_out = float(np.trapz(mdot, ad.t))
    errMass = abs((mf + m_out) - m0) / m0

    return GateMetrics(errP=errP, errT=errT, errMass=errMass)


def gate_two() -> GateMetrics:
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
    errMass = abs((m_total_f + m_out) - m_total_0) / m_total_0

    E0 = float(np.sum(res.m[:, 0] * C_V * res.T[:, 0]))
    Ef = float(np.sum(res.m[:, -1] * C_V * res.T[:, -1]))
    Eout = float(np.trapz(mdot * C_P * np.maximum(res.T[0, :], T_SAFE), res.t))
    errE = abs((Ef + Eout) - E0) / E0
    return GateMetrics(errP=0.0, errT=0.0, errMass=errMass, errEnergy=errE)
