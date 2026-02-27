import math
from dataclasses import asdict

import numpy as np

from .cases import CaseConfig, SolveResult
from .constants import C_CHOKED, P0, PI_C, R_GAS, T0, T_SAFE
from .graph import EXT_NODE, GasNode, OrificeEdge, ShortTubeEdge, SlotChannelEdge
from .validity import evaluate_validity_flags


def compute_tau_exit(total_volume: float, Cd_exit: float, A_exit_total: float) -> float:
    mdot_ch = Cd_exit * A_exit_total * C_CHOKED * P0 / math.sqrt(R_GAS * T0)
    if mdot_ch <= 0.0:
        return float("inf")
    return (total_volume * P0) / (R_GAS * T0 * mdot_ch)


def summarize_result(
    nodes: list[GasNode], edges: list, bcs: list, case: CaseConfig, sol
) -> SolveResult:
    nodes_use = getattr(sol, "nodes_local", nodes)
    edges_use = getattr(sol, "edges_local", edges)
    bcs_use = getattr(sol, "bcs_local", bcs)
    N = len(nodes_use)
    V = np.array([n.V for n in nodes_use], dtype=float)
    t = sol.t

    if case.thermo == "isothermal":
        m = sol.y[:N]
        T = np.full_like(m, T0)
        if getattr(sol, "ext_idx", None) is not None:
            T[sol.ext_idx, :] = case.T_ext
    else:
        m = sol.y[:N]
        T = sol.y[N : 2 * N]
        if getattr(sol, "ext_idx", None) is not None:
            T[sol.ext_idx, :] = case.T_ext
    T_eff = np.maximum(T, T_SAFE)
    P = np.maximum(m, 0.0) * R_GAS * T_eff / V[:, None]

    ext_idx = getattr(sol, "ext_idx", None)
    if case.external_model == "dynamic_pump" and ext_idx is not None:
        P_ext = P[ext_idx, :]
    else:
        profile = bcs_use[0].profile
        P_ext = profile.P_array(t)

    dP_edges: dict[str, np.ndarray] = {}
    max_dP: dict[str, float] = {}
    peak_diag: dict[str, dict] = {}

    total_volume = float(np.sum(V if ext_idx is None else np.delete(V, ext_idx)))
    A_exit_total = None
    Cd_exit = None
    for e in edges_use:
        if isinstance(e, (OrificeEdge, ShortTubeEdge)) and "exit" in e.label.lower():
            A_exit_total = e.A_total
            Cd_exit = e.Cd_model()
            break
    if A_exit_total is None:
        A_exit_total = max(
            (e.A_total for e in edges_use if isinstance(e, (OrificeEdge, ShortTubeEdge))),
            default=0.0,
        )
        Cd_exit = 0.62
    tau_exit = compute_tau_exit(total_volume, float(Cd_exit), float(A_exit_total))

    for e in edges_use:
        if isinstance(e, (OrificeEdge, ShortTubeEdge)):
            label = e.label or f"edge({e.a}->{e.b})"
            if e.b == EXT_NODE:
                dp = P[e.a, :] - P_ext
            elif ext_idx is not None and e.b == ext_idx:
                dp = P[e.a, :] - P[e.b, :]
            else:
                dp = P[e.a, :] - P[e.b, :]
            dP_edges[label] = dp
        elif isinstance(e, SlotChannelEdge):
            label = e.label or f"slot({e.a}->{e.b})"
            dP_edges[label] = P[e.a, :] - P[e.b, :]

    for label, dp in dP_edges.items():
        idx = int(np.argmax(np.abs(dp)))
        max_dP[label] = float(np.abs(dp[idx]))
        edge_obj = next(
            e
            for e in edges_use
            if ((e.label or f"edge({e.a}->{e.b})") == label)
            or ((e.label or f"slot({e.a}->{e.b})") == label)
        )

        a, b = edge_obj.a, edge_obj.b
        Pa = float(P[a, idx])
        if b == EXT_NODE:
            Pb = float(P_ext[idx])
        else:
            Pb = float(P[b, idx])

        if Pa >= Pb:
            p_up, p_dn = Pa, Pb
            t_up = float(T_eff[a, idx])
        else:
            p_up, p_dn = Pb, Pa
            if b == EXT_NODE:
                t_up = case.T_ext
            else:
                t_up = float(T_eff[b, idx])

        regime = "viscous_slot"
        if isinstance(edge_obj, (OrificeEdge, ShortTubeEdge)):
            r_pk = (p_dn / p_up) if p_up > 0 else 0.0
            regime = "CHOKED" if r_pk <= PI_C else "subsonic"
        else:
            r_pk = float(min(Pa, Pb) / max(Pa, Pb)) if max(Pa, Pb) > 0 else 0.0

        if case.external_model == "dynamic_pump":
            peak_type = "internal"
        else:
            profile = bcs_use[0].profile
            peak_type = profile.classify_peak(float(t[idx]), float(t[-1]))

        peak_diag[label] = {
            "t_peak": float(t[idx]),
            "t_peak_over_tau_exit": (
                float(t[idx] / tau_exit) if np.isfinite(tau_exit) else float("nan")
            ),
            "P_up": p_up,
            "P_down": p_dn,
            "r": float(r_pk),
            "regime": regime,
            "T_up": t_up,
            "peak_type": peak_type,
            "dP_signed": float(dp[idx]),
        }

    validity_flags = evaluate_validity_flags(
        nodes=nodes_use,
        edges=edges_use,
        P=P,
        T=T_eff,
        m=m,
        P_ext=P_ext,
        t=t,
        l_char_m=case.l_char_m,
    )

    meta = {
        "case": asdict(case),
        "nodes": [asdict(n) for n in nodes_use],
        "edges": [asdict(e) for e in edges_use],
        "profile": (
            {"name": bcs_use[0].profile.name, "events": list(bcs_use[0].profile.events)}
            if bcs_use
            else {"name": "dynamic_pump", "events": []}
        ),
        "solver": {
            "success": bool(sol.success),
            "message": str(sol.message),
            "t_end": float(sol.t[-1]),
            "n_steps": int(len(sol.t)),
        },
        "validity_flags": validity_flags,
    }
    return SolveResult(
        t=t,
        m=m,
        T=T,
        P=P,
        P_ext=P_ext,
        peak_diag=peak_diag,
        max_dP=max_dP,
        tau_exit=tau_exit,
        meta=meta,
    )
