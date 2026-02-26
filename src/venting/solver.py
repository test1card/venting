import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix

from .cases import CaseConfig
from .constants import C_CHOKED, C_P, C_V, M_SAFE, P0, R_GAS, T0, T_SAFE
from .flow import mdot_orifice_pos, mdot_slot_pos
from .graph import EXT_NODE, ExternalBC, GasNode, OrificeEdge, SlotChannelEdge


def build_rhs(
    nodes: list[GasNode], edges: list, bcs: list[ExternalBC], case: CaseConfig
):
    N = len(nodes)
    V = np.array([n.V for n in nodes], dtype=float)
    A_w = np.array([n.A_wall for n in nodes], dtype=float)
    bc_map = {bc.node: bc for bc in bcs}

    def p_from_mt(m: np.ndarray, T: np.ndarray) -> np.ndarray:
        return m * R_GAS * T / V

    if case.thermo == "isothermal":

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            m = y[:N]
            T = np.full(N, T0)
            P = p_from_mt(np.maximum(m, 0.0), T)
            dm = np.zeros(N)
            for e in edges:
                if isinstance(e, OrificeEdge):
                    a, b = e.a, e.b
                    if b == EXT_NODE:
                        bc = bc_map[a]
                        pext = bc.profile.P(float(t))
                        cd = e.Cd_model()
                        if P[a] >= pext:
                            md = mdot_orifice_pos(P[a], T[a], pext, cd, e.A_total)
                            dm[a] -= md
                        else:
                            md = mdot_orifice_pos(pext, bc.T_ext, P[a], cd, e.A_total)
                            dm[a] += md
                    else:
                        Pa, Pb = P[a], P[b]
                        cd = e.Cd_model()
                        if Pa >= Pb:
                            md = mdot_orifice_pos(Pa, T[a], Pb, cd, e.A_total)
                            dm[a] -= md
                            dm[b] += md
                        else:
                            md = mdot_orifice_pos(Pb, T[b], Pa, cd, e.A_total)
                            dm[b] -= md
                            dm[a] += md
                elif isinstance(e, SlotChannelEdge):
                    a, b = e.a, e.b
                    Pa, Pb = P[a], P[b]
                    if Pa >= Pb:
                        md = mdot_slot_pos(Pa, T[a], Pb, e.w, e.delta, e.L)
                        dm[a] -= md
                        dm[b] += md
                    else:
                        md = mdot_slot_pos(Pb, T[b], Pa, e.w, e.delta, e.L)
                        dm[b] -= md
                        dm[a] += md
            return dm

        return rhs, N

    if case.thermo != "intermediate":
        raise ValueError("case.thermo must be 'isothermal' or 'intermediate'")

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        m = y[:N]
        T = y[N:]
        T_eff = np.maximum(T, T_SAFE)
        P = p_from_mt(np.maximum(m, 0.0), T_eff)
        dm = np.zeros(N)
        dE = np.zeros(N)
        for e in edges:
            if isinstance(e, OrificeEdge):
                a, b = e.a, e.b
                if b == EXT_NODE:
                    bc = bc_map[a]
                    pext = bc.profile.P(float(t))
                    cd = e.Cd_model()
                    if P[a] >= pext:
                        md = mdot_orifice_pos(P[a], T_eff[a], pext, cd, e.A_total)
                        dm[a] -= md
                        dE[a] += -md * R_GAS * T[a]
                    else:
                        md = mdot_orifice_pos(pext, bc.T_ext, P[a], cd, e.A_total)
                        dm[a] += md
                        dE[a] += md * (C_P * bc.T_ext - C_V * T[a])
                else:
                    Pa, Pb = P[a], P[b]
                    cd = e.Cd_model()
                    if Pa >= Pb:
                        md = mdot_orifice_pos(Pa, T_eff[a], Pb, cd, e.A_total)
                        dm[a] -= md
                        dm[b] += md
                        dE[a] += -md * R_GAS * T[a]
                        dE[b] += md * (C_P * T[a] - C_V * T[b])
                    else:
                        md = mdot_orifice_pos(Pb, T_eff[b], Pa, cd, e.A_total)
                        dm[b] -= md
                        dm[a] += md
                        dE[b] += -md * R_GAS * T[b]
                        dE[a] += md * (C_P * T[b] - C_V * T[a])
            elif isinstance(e, SlotChannelEdge):
                a, b = e.a, e.b
                Pa, Pb = P[a], P[b]
                if Pa >= Pb:
                    md = mdot_slot_pos(Pa, T_eff[a], Pb, e.w, e.delta, e.L)
                    dm[a] -= md
                    dm[b] += md
                    dE[a] += -md * R_GAS * T[a]
                    dE[b] += md * (C_P * T[a] - C_V * T[b])
                else:
                    md = mdot_slot_pos(Pb, T_eff[b], Pa, e.w, e.delta, e.L)
                    dm[b] -= md
                    dm[a] += md
                    dE[b] += -md * R_GAS * T[b]
                    dE[a] += md * (C_P * T[b] - C_V * T[a])
        dE += case.h_conv * A_w * (case.T_wall - T)
        dT = np.where(np.maximum(m, 0.0) > M_SAFE, dE / (np.maximum(m, 0.0) * C_V), 0.0)
        return np.concatenate([dm, dT])

    return rhs, 2 * N


def solve_case(
    nodes: list[GasNode], edges: list, bcs: list[ExternalBC], case: CaseConfig
):
    N = len(nodes)
    V = np.array([n.V for n in nodes], dtype=float)
    rhs, n_vars = build_rhs(nodes, edges, bcs, case)
    m0 = P0 * V / (R_GAS * T0)
    y0 = (
        m0.copy()
        if case.thermo == "isothermal"
        else np.concatenate([m0, np.full(N, T0)])
    )
    t_eval = np.linspace(0.0, case.duration, int(case.n_pts))

    jac = lil_matrix((n_vars, n_vars))
    for i in range(n_vars):
        jac[i, i] = 1

    profile = bcs[0].profile

    def P_from_state(y: np.ndarray) -> np.ndarray:
        if case.thermo == "isothermal":
            m = y[:N]
            T = np.full(N, T0)
        else:
            m = y[:N]
            T = np.maximum(y[N:], T_SAFE)
        return np.maximum(m, 0.0) * R_GAS * T / V

    events = []
    for i in range(N):

        def ev_mi(t, y, ii=i):
            return y[ii] + 1e-15

        ev_mi.terminal = True
        ev_mi.direction = -1
        events.append(ev_mi)

    def ev_equil_rms(t, y):
        P = P_from_state(y)
        Pe = profile.P(float(t))
        return float(np.sqrt(np.mean((P - Pe) ** 2))) - case.p_rms_tol

    ev_equil_rms.terminal = True
    ev_equil_rms.direction = -1
    events.append(ev_equil_rms)

    def ev_p_stop(t, y):
        P = P_from_state(y)
        Pe = profile.P(float(t))
        return max(np.max(P) - case.p_stop, Pe - 10.0)

    ev_p_stop.terminal = True
    ev_p_stop.direction = -1
    events.append(ev_p_stop)

    A_max = 0.0
    Cd_max = 0.0
    for e in edges:
        if isinstance(e, OrificeEdge):
            A_max = max(A_max, e.A_total)
            Cd_max = max(Cd_max, e.Cd_model())
    V_min = float(np.min(V))
    mdot_ch = Cd_max * A_max * C_CHOKED * P0 / np.sqrt(R_GAS * T0) if A_max > 0 else 0.0
    tau_min = (V_min * P0) / (R_GAS * T0 * mdot_ch) if mdot_ch > 0 else 1.0
    max_step = max(min(case.duration / 2000.0, tau_min / 10.0), 1e-4)

    return solve_ivp(
        rhs,
        (0.0, case.duration),
        y0,
        method="Radau",
        t_eval=t_eval,
        rtol=1e-7 if case.thermo == "isothermal" else 1e-6,
        atol=1e-10 if case.thermo == "isothermal" else 1e-8,
        max_step=max_step,
        jac_sparsity=jac,
        events=events,
    )
