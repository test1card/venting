import math

import numpy as np
from scipy.integrate import solve_ivp

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
                                md = mdot_orifice_pos_props(
                                    P[a], T[a], pext, cd0, e.A_total
                                )
                            else:
                                md = mdot_short_tube_pos(
                                    P[a],
                                    T[a],
                                    pext,
                                    cd0,
                                    e.A_total,
                                    e.D,
                                    e.L,
                                    e.eps,
                                    e.K_in,
                                    e.K_out,
                                )
                            dm[a] -= md
                        else:
                            if isinstance(e, OrificeEdge):
                                md = mdot_orifice_pos_props(
                                    pext, bc.T_ext, P[a], cd0, e.A_total
                                )
                            else:
                                md = mdot_short_tube_pos(
                                    pext,
                                    bc.T_ext,
                                    P[a],
                                    cd0,
                                    e.A_total,
                                    e.D,
                                    e.L,
                                    e.eps,
                                    e.K_in,
                                    e.K_out,
                                )
                            dm[a] += md
                    else:
                        Pa, Pb = P[a], P[b]
                        if Pa >= Pb:
                            if isinstance(e, OrificeEdge):
                                md = mdot_orifice_pos_props(
                                    Pa, T[a], Pb, cd0, e.A_total
                                )
                            else:
                                md = mdot_short_tube_pos(
                                    Pa,
                                    T[a],
                                    Pb,
                                    cd0,
                                    e.A_total,
                                    e.D,
                                    e.L,
                                    e.eps,
                                    e.K_in,
                                    e.K_out,
                                )
                            dm[a] -= md
                            dm[b] += md
                        else:
                            if isinstance(e, OrificeEdge):
                                md = mdot_orifice_pos_props(
                                    Pb, T[b], Pa, cd0, e.A_total
                                )
                            else:
                                md = mdot_short_tube_pos(
                                    Pb,
                                    T[b],
                                    Pa,
                                    cd0,
                                    e.A_total,
                                    e.D,
                                    e.L,
                                    e.eps,
                                    e.K_in,
                                    e.K_out,
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
        raise ValueError(
            "case.thermo must be 'isothermal', 'intermediate', or 'variable'"
        )

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


def solve_case(
    nodes: list[GasNode], edges: list, bcs: list[ExternalBC], case: CaseConfig
):
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
            mdot_pump = (
                case.pump_speed_m3s
                * max(P[ext_idx] - case.P_ult_Pa, 0.0)
                / (R_GAS * max(case.T_ext, T_SAFE))
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
    mdot_ch = (
        Cd_max * A_max * C_CHOKED * P0 / math.sqrt(R_GAS * T0) if A_max > 0 else 0.0
    )
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
        events=events,
    )
    sol.ext_idx = ext_idx
    sol.node_count = len(nodes_local)
    sol.nodes_local = nodes_local
    sol.edges_local = edges_local
    sol.bcs_local = bcs_local
    return sol


def solve_case_stream(
    nodes: list[GasNode],
    edges: list,
    bcs: list[ExternalBC],
    case: CaseConfig,
    callback=None,
    n_chunks: int = 20,
    should_stop=None,
    stop_check=None,
):
    """True streaming solve via piecewise integration segments."""
    if should_stop is None and stop_check is not None:
        should_stop = stop_check
    if callback is None and should_stop is None:
        return solve_case(nodes, edges, bcs, case)
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
    rhs_core, _ = build_rhs(nodes_local, edges_local, bcs_local, case)

    m0 = P0 * V / (R_GAS * T0)
    if ext_idx is not None:
        m0[ext_idx] = max(case.P_ult_Pa, 10.0) * V[ext_idx] / (R_GAS * case.T_ext)

    if case.thermo == "isothermal":
        y = m0.copy()
    else:
        T_init = np.full(N, T0)
        if ext_idx is not None:
            T_init[ext_idx] = case.T_ext
        y = np.concatenate([m0, T_init])
        if case.wall_model == "lumped":
            y = np.concatenate([y, np.full(N, case.T_wall)])

    t_eval = np.linspace(0.0, case.duration, int(case.n_pts))

    def p_from_state(y_state: np.ndarray) -> np.ndarray:
        m = y_state[:N]
        if case.thermo == "isothermal":
            T = np.full(N, T0)
            if ext_idx is not None:
                T[ext_idx] = case.T_ext
        else:
            T = np.maximum(y_state[N : 2 * N], T_SAFE)
            if ext_idx is not None:
                T[ext_idx] = case.T_ext
        return np.maximum(m, 0.0) * R_GAS * T / V

    def rhs(t: float, y_state: np.ndarray) -> np.ndarray:
        dy = rhs_core(t, y_state)
        if ext_idx is not None:
            P = p_from_state(y_state)
            mdot_pump = (
                case.pump_speed_m3s
                * max(P[ext_idx] - case.P_ult_Pa, 0.0)
                / (R_GAS * max(case.T_ext, T_SAFE))
            )
            dy[ext_idx] -= mdot_pump
            if case.thermo != "isothermal":
                dy[N + ext_idx] = 0.0
        return dy

    A_max = 0.0
    Cd_max = 0.0
    for e in edges_local:
        if isinstance(e, (OrificeEdge, ShortTubeEdge)):
            A_max = max(A_max, e.A_total)
            Cd_max = max(Cd_max, e.Cd_model())
    V_min = float(np.min(V))
    mdot_ch = (
        Cd_max * A_max * C_CHOKED * P0 / math.sqrt(R_GAS * T0) if A_max > 0 else 0.0
    )
    tau_min = (V_min * P0) / (R_GAS * T0 * mdot_ch) if mdot_ch > 0 else 1.0
    max_step = max(min(case.duration / 2000.0, tau_min / 10.0), 1e-4)

    n_chunks = max(int(n_chunks), 1)
    bounds = np.linspace(0.0, case.duration, n_chunks + 1)
    t_out = []
    y_out = []
    success = True
    message = "stream completed"

    for i in range(n_chunks):
        if should_stop is not None and should_stop():
            success = False
            message = "stream cancelled"
            break
        t0 = bounds[i]
        t1 = bounds[i + 1]
        mask = (t_eval >= t0) & (t_eval <= t1)
        t_seg = t_eval[mask]
        if t_seg.size == 0:
            t_seg = np.array([t0, t1])
        sol_seg = solve_ivp(
            rhs,
            (float(t0), float(t1)),
            y,
            method="Radau",
            t_eval=t_seg,
            rtol=1e-7 if case.thermo == "isothermal" else 1e-6,
            atol=1e-10 if case.thermo == "isothermal" else 1e-8,
            max_step=max_step,
        )
        if not sol_seg.success:
            success = False
            message = str(sol_seg.message)
            break

        seg_t = sol_seg.t
        seg_y = sol_seg.y
        if i > 0 and seg_t.size > 0:
            seg_t = seg_t[1:]
            seg_y = seg_y[:, 1:]
        if seg_t.size > 0:
            t_out.append(seg_t)
            y_out.append(seg_y)
        y = sol_seg.y[:, -1]

        if callback is not None and t_out:
            t_cat = np.concatenate(t_out)
            y_cat = np.concatenate(y_out, axis=1)
            callback(
                {
                    "t": t_cat,
                    "y": y_cat,
                    "progress": float((i + 1) / n_chunks),
                    "done": bool(i == n_chunks - 1),
                    "node_count": N,
                    "n_nodes": N,
                    "thermo": case.thermo,
                    "wall_model": case.wall_model,
                    "ext_idx": ext_idx,
                }
            )

    if t_out:
        t_final = np.concatenate(t_out)
        y_final = np.concatenate(y_out, axis=1)
    else:
        t_final = np.array([0.0])
        y_final = y[:, None]

    if success and (t_final.size != t_eval.size or not np.allclose(t_final, t_eval)):
        y_interp = np.empty((y_final.shape[0], t_eval.size), dtype=float)
        for i_row in range(y_final.shape[0]):
            y_interp[i_row] = np.interp(t_eval, t_final, y_final[i_row])
        t_final = t_eval
        y_final = y_interp

    class _Sol:
        pass

    sol = _Sol()
    sol.t = t_final
    sol.y = y_final
    sol.success = success
    sol.message = message
    sol.ext_idx = ext_idx
    sol.node_count = len(nodes_local)
    sol.nodes_local = nodes_local
    sol.edges_local = edges_local
    sol.bcs_local = bcs_local
    return sol
