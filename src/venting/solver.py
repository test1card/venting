import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix

from .cases import CaseConfig
from .constants import C_CHOKED, C_P, C_V, M_SAFE, P0, R_GAS, T0, T_SAFE
from .flow import mdot_orifice_pos_props, mdot_short_tube_pos, mdot_slot_pos
from .graph import (
    EXT_NODE,
    ExternalBC,
    GasNode,
    OrificeEdge,
    ShortTubeEdge,
    SlotChannelEdge,
)
from .thermo import cp_air, cv_air, gamma_air, h_air, u_air

SIGMA_SB = 5.670374419e-8


def _property_model(thermo: str):
    if thermo == "variable":
        return cp_air, cv_air, gamma_air, h_air, u_air
    if thermo == "intermediate":
        return (
            lambda T: C_P,
            lambda T: C_V,
            lambda T: C_P / C_V,
            lambda T: C_P * T,
            lambda T: C_V * T,
        )
    raise ValueError("Property model valid for 'intermediate' or 'variable'")


def build_rhs(nodes: list[GasNode], edges: list, bcs: list[ExternalBC], case: CaseConfig):
    N = len(nodes)
    V = np.array([n.V for n in nodes], dtype=float)
    A_w = np.array([n.A_wall for n in nodes], dtype=float)
    bc_map = {bc.node: bc for bc in bcs}

    def p_from_mt(m: np.ndarray, T: np.ndarray) -> np.ndarray:
        return m * R_GAS * T / V

    if case.thermo == "isothermal":
        n_vars = N

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            m = y[:N]
            T = np.full(N, T0)
            P = p_from_mt(np.maximum(m, 0.0), T)
            dm = np.zeros(N)
            for e in edges:
                if isinstance(e, (OrificeEdge, ShortTubeEdge)):
                    a, b = e.a, e.b
                    cd0 = e.Cd_model()
                    if b == EXT_NODE:
                        bc = bc_map[a]
                        pext = bc.profile.P(float(t))
                        if P[a] >= pext:
                            if isinstance(e, OrificeEdge):
                                md = mdot_orifice_pos_props(P[a], T[a], pext, cd0, e.A_total)
                            else:
                                md = mdot_short_tube_pos(
                                    P[a], T[a], pext, cd0, e.A_total, e.D, e.L, e.eps, e.K_in, e.K_out
                                )
                            dm[a] -= md
                        else:
                            if isinstance(e, OrificeEdge):
                                md = mdot_orifice_pos_props(pext, bc.T_ext, P[a], cd0, e.A_total)
                            else:
                                md = mdot_short_tube_pos(
                                    pext, bc.T_ext, P[a], cd0, e.A_total, e.D, e.L, e.eps, e.K_in, e.K_out
                                )
                            dm[a] += md
                    else:
                        Pa, Pb = P[a], P[b]
                        if Pa >= Pb:
                            if isinstance(e, OrificeEdge):
                                md = mdot_orifice_pos_props(Pa, T[a], Pb, cd0, e.A_total)
                            else:
                                md = mdot_short_tube_pos(
                                    Pa, T[a], Pb, cd0, e.A_total, e.D, e.L, e.eps, e.K_in, e.K_out
                                )
                            dm[a] -= md
                            dm[b] += md
                        else:
                            if isinstance(e, OrificeEdge):
                                md = mdot_orifice_pos_props(Pb, T[b], Pa, cd0, e.A_total)
                            else:
                                md = mdot_short_tube_pos(
                                    Pb, T[b], Pa, cd0, e.A_total, e.D, e.L, e.eps, e.K_in, e.K_out
                                )
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

        return rhs, n_vars

    if case.thermo not in {"intermediate", "variable"}:
        raise ValueError("case.thermo must be 'isothermal', 'intermediate', or 'variable'")

    cp_fn, cv_fn, gamma_fn, h_fn, u_fn = _property_model(case.thermo)

    has_lumped_wall = case.wall_model == "lumped"
    n_vars = 2 * N + (N if has_lumped_wall else 0)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        m = y[:N]
        T = y[N : 2 * N]
        Tw = y[2 * N : 3 * N] if has_lumped_wall else None
        T_eff = np.maximum(T, T_SAFE)

        P = p_from_mt(np.maximum(m, 0.0), T_eff)
        dm = np.zeros(N)
        dE = np.zeros(N)

        for e in edges:
            if isinstance(e, (OrificeEdge, ShortTubeEdge)):
                a, b = e.a, e.b
                cd0 = e.Cd_model()
                if b == EXT_NODE:
                    bc = bc_map[a]
                    pext = bc.profile.P(float(t))
                    if P[a] >= pext:
                        gamma_up = gamma_fn(float(T_eff[a]))
                        if isinstance(e, OrificeEdge):
                            md = mdot_orifice_pos_props(
                                P[a], T_eff[a], pext, cd0, e.A_total, gamma=gamma_up
                            )
                        else:
                            md = mdot_short_tube_pos(
                                P[a],
                                T_eff[a],
                                pext,
                                cd0,
                                e.A_total,
                                e.D,
                                e.L,
                                e.eps,
                                e.K_in,
                                e.K_out,
                                gamma=gamma_up,
                            )
                        dm[a] -= md
                        dE[a] += -md * h_fn(float(T_eff[a]))
                    else:
                        gamma_up = gamma_fn(float(max(bc.T_ext, T_SAFE)))
                        if isinstance(e, OrificeEdge):
                            md = mdot_orifice_pos_props(
                                pext,
                                max(bc.T_ext, T_SAFE),
                                P[a],
                                cd0,
                                e.A_total,
                                gamma=gamma_up,
                            )
                        else:
                            md = mdot_short_tube_pos(
                                pext,
                                max(bc.T_ext, T_SAFE),
                                P[a],
                                cd0,
                                e.A_total,
                                e.D,
                                e.L,
                                e.eps,
                                e.K_in,
                                e.K_out,
                                gamma=gamma_up,
                            )
                        dm[a] += md
                        dE[a] += md * h_fn(float(max(bc.T_ext, T_SAFE)))
                else:
                    Pa, Pb = P[a], P[b]
                    if Pa >= Pb:
                        gamma_up = gamma_fn(float(T_eff[a]))
                        if isinstance(e, OrificeEdge):
                            md = mdot_orifice_pos_props(
                                Pa, T_eff[a], Pb, cd0, e.A_total, gamma=gamma_up
                            )
                        else:
                            md = mdot_short_tube_pos(
                                Pa,
                                T_eff[a],
                                Pb,
                                cd0,
                                e.A_total,
                                e.D,
                                e.L,
                                e.eps,
                                e.K_in,
                                e.K_out,
                                gamma=gamma_up,
                            )
                        dm[a] -= md
                        dm[b] += md
                        dE[a] += -md * h_fn(float(T_eff[a]))
                        dE[b] += md * h_fn(float(T_eff[a]))
                    else:
                        gamma_up = gamma_fn(float(T_eff[b]))
                        if isinstance(e, OrificeEdge):
                            md = mdot_orifice_pos_props(
                                Pb, T_eff[b], Pa, cd0, e.A_total, gamma=gamma_up
                            )
                        else:
                            md = mdot_short_tube_pos(
                                Pb,
                                T_eff[b],
                                Pa,
                                cd0,
                                e.A_total,
                                e.D,
                                e.L,
                                e.eps,
                                e.K_in,
                                e.K_out,
                                gamma=gamma_up,
                            )
                        dm[b] -= md
                        dm[a] += md
                        dE[b] += -md * h_fn(float(T_eff[b]))
                        dE[a] += md * h_fn(float(T_eff[b]))
            elif isinstance(e, SlotChannelEdge):
                a, b = e.a, e.b
                Pa, Pb = P[a], P[b]
                if Pa >= Pb:
                    md = mdot_slot_pos(Pa, T_eff[a], Pb, e.w, e.delta, e.L)
                    dm[a] -= md
                    dm[b] += md
                    dE[a] += -md * h_fn(float(T_eff[a]))
                    dE[b] += md * h_fn(float(T_eff[a]))
                else:
                    md = mdot_slot_pos(Pb, T_eff[b], Pa, e.w, e.delta, e.L)
                    dm[b] -= md
                    dm[a] += md
                    dE[b] += -md * h_fn(float(T_eff[b]))
                    dE[a] += md * h_fn(float(T_eff[b]))

        if has_lumped_wall:
            Qdot = case.h_conv * A_w * (Tw - T)
        else:
            Qdot = case.h_conv * A_w * (case.T_wall - T)
        dE += Qdot

        dT = np.zeros(N)
        for i in range(N):
            m_eff = max(float(m[i]), 0.0)
            if m_eff > M_SAFE:
                u_i = u_fn(float(T_eff[i]))
                cv_i = max(cv_fn(float(T_eff[i])), 1e-9)
                dT[i] = (dE[i] - u_i * dm[i]) / (m_eff * cv_i)

        if has_lumped_wall:
            dTw = np.zeros(N)
            for i in range(N):
                A = A_w[i]
                Cw = case.wall_C_per_area * A
                if Cw <= 0.0:
                    continue
                q_in = case.h_conv * A * (T[i] - Tw[i])
                q_out = case.wall_h_out * A * (Tw[i] - case.wall_T_inf)
                q_rad = (
                    case.wall_emissivity
                    * SIGMA_SB
                    * A
                    * (Tw[i] ** 4 - case.wall_T_sur**4)
                )
                q_src = case.wall_q_flux * A
                dTw[i] = (q_in - q_out - q_rad + q_src) / Cw
            return np.concatenate([dm, dT, dTw])

        return np.concatenate([dm, dT])

    return rhs, n_vars


def solve_case(nodes: list[GasNode], edges: list, bcs: list[ExternalBC], case: CaseConfig):
    nodes_local = list(nodes)
    edges_local = list(edges)
    bcs_local = list(bcs)
    ext_idx = None

    if case.external_model == "dynamic_pump":
        ext_idx = len(nodes_local)
        nodes_local.append(GasNode("external", case.V_ext, 0.0))
        rewired = []
        for e in edges_local:
            if isinstance(e, OrificeEdge) and e.b == EXT_NODE:
                rewired.append(
                    OrificeEdge(e.a, ext_idx, e.A_total, e.Cd_model, label=e.label)
                )
            elif isinstance(e, ShortTubeEdge) and e.b == EXT_NODE:
                rewired.append(
                    ShortTubeEdge(
                        e.a,
                        ext_idx,
                        e.A_total,
                        e.D,
                        e.L,
                        e.eps,
                        e.K_in,
                        e.K_out,
                        e.Cd_model,
                        label=e.label,
                    )
                )
            else:
                rewired.append(e)
        edges_local = rewired
        bcs_local = []

    N = len(nodes_local)
    V = np.array([n.V for n in nodes_local], dtype=float)
    rhs_core, n_vars = build_rhs(nodes_local, edges_local, bcs_local, case)

    m0 = P0 * V / (R_GAS * T0)
    if ext_idx is not None:
        m0[ext_idx] = max(case.P_ult_Pa, 10.0) * V[ext_idx] / (R_GAS * case.T_ext)

    if case.thermo == "isothermal":
        y0 = m0.copy()
    else:
        T_init = np.full(N, T0)
        if ext_idx is not None:
            T_init[ext_idx] = case.T_ext
        y0 = np.concatenate([m0, T_init])
        if case.wall_model == "lumped":
            y0 = np.concatenate([y0, np.full(N, case.T_wall)])

    t_eval = np.linspace(0.0, case.duration, int(case.n_pts))

    jac = lil_matrix((n_vars, n_vars))
    for i in range(n_vars):
        jac[i, i] = 1

    profile = bcs_local[0].profile if bcs_local else None

    def p_from_state(y: np.ndarray) -> np.ndarray:
        m = y[:N]
        if case.thermo == "isothermal":
            T = np.full(N, T0)
            if ext_idx is not None:
                T[ext_idx] = case.T_ext
        else:
            T = np.maximum(y[N : 2 * N], T_SAFE)
            if ext_idx is not None:
                T[ext_idx] = case.T_ext
        return np.maximum(m, 0.0) * R_GAS * T / V

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        dy = rhs_core(t, y)
        if ext_idx is not None:
            P = p_from_state(y)
            mdot_pump = case.pump_speed_m3s * max(P[ext_idx] - case.P_ult_Pa, 0.0) / (
                R_GAS * max(case.T_ext, T_SAFE)
            )
            dy[ext_idx] -= mdot_pump
            if case.thermo != "isothermal":
                dy[N + ext_idx] = 0.0
        return dy

    events = []
    for i in range(N):

        def ev_mi(t, y, ii=i):
            return y[ii] + 1e-15

        ev_mi.terminal = True
        ev_mi.direction = -1
        events.append(ev_mi)

    def ev_equil_rms(t, y):
        P = p_from_state(y)
        if ext_idx is not None:
            Pe = P[ext_idx]
            Pcmp = np.delete(P, ext_idx)
        else:
            Pe = profile.P(float(t))
            Pcmp = P
        return float(np.sqrt(np.mean((Pcmp - Pe) ** 2))) - case.p_rms_tol

    ev_equil_rms.terminal = True
    ev_equil_rms.direction = -1
    events.append(ev_equil_rms)

    def ev_p_stop(t, y):
        P = p_from_state(y)
        if ext_idx is not None:
            Pe = P[ext_idx]
            Pcmp = np.delete(P, ext_idx)
        else:
            Pe = profile.P(float(t))
            Pcmp = P
        return max(np.max(Pcmp) - case.p_stop, Pe - 10.0)

    ev_p_stop.terminal = True
    ev_p_stop.direction = -1
    events.append(ev_p_stop)

    A_max = 0.0
    Cd_max = 0.0
    for e in edges_local:
        if isinstance(e, (OrificeEdge, ShortTubeEdge)):
            A_max = max(A_max, e.A_total)
            Cd_max = max(Cd_max, e.Cd_model())
    V_min = float(np.min(V))
    mdot_ch = Cd_max * A_max * C_CHOKED * P0 / math.sqrt(R_GAS * T0) if A_max > 0 else 0.0
    tau_min = (V_min * P0) / (R_GAS * T0 * mdot_ch) if mdot_ch > 0 else 1.0
    max_step = max(min(case.duration / 2000.0, tau_min / 10.0), 1e-4)

    sol = solve_ivp(
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
    sol.ext_idx = ext_idx
    sol.node_count = len(nodes_local)
    sol.nodes_local = nodes_local
    sol.edges_local = edges_local
    sol.bcs_local = bcs_local
    return sol
