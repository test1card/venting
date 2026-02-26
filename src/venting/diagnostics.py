from dataclasses import asdict
import math

import numpy as np

from .cases import CaseConfig, SolveResult
from .constants import C_CHOKED, PI_C, P0, R_GAS, T0, T_SAFE
from .graph import EXT_NODE, GasNode, OrificeEdge, SlotChannelEdge


def compute_tau_exit(total_volume: float, Cd_exit: float, A_exit_total: float) -> float:
    mdot_ch = Cd_exit * A_exit_total * C_CHOKED * P0 / math.sqrt(R_GAS * T0)
    if mdot_ch <= 0.0:
        return float("inf")
    return (total_volume * P0) / (R_GAS * T0 * mdot_ch)


def summarize_result(nodes: list[GasNode], edges: list, bcs: list, case: CaseConfig, sol) -> SolveResult:
    N = len(nodes)
    V = np.array([n.V for n in nodes], dtype=float)
    profile = bcs[0].profile
    t = sol.t
    if case.thermo == "isothermal":
        m = sol.y[:N]
        T = np.full_like(m, T0)
    else:
        m = sol.y[:N]
        T = sol.y[N : 2 * N]
    T_eff = np.maximum(T, T_SAFE)
    P = np.maximum(m, 0.0) * R_GAS * T_eff / V[:, None]
    P_ext = profile.P_array(t)

    dP_edges: dict[str, np.ndarray] = {}
    max_dP: dict[str, float] = {}
    peak_diag: dict[str, dict] = {}

    total_volume = float(np.sum(V))
    A_exit_total = None
    Cd_exit = None
    for e in edges:
        if isinstance(e, OrificeEdge) and "exit" in e.label.lower():
            A_exit_total = e.A_total
            Cd_exit = e.Cd_model()
            break
    if A_exit_total is None:
        A_exit_total = max((e.A_total for e in edges if isinstance(e, OrificeEdge)), default=0.0)
        Cd_exit = 0.62
    tau_exit = compute_tau_exit(total_volume, float(Cd_exit), float(A_exit_total))

    for e in edges:
        if isinstance(e, OrificeEdge):
            label = e.label or f"orifice({e.a}->{e.b})"
            dp = P[e.a, :] - (P_ext if e.b == EXT_NODE else P[e.b, :])
            dP_edges[label] = dp
        elif isinstance(e, SlotChannelEdge):
            label = e.label or f"slot({e.a}->{e.b})"
            dP_edges[label] = P[e.a, :] - P[e.b, :]

    for label, dp in dP_edges.items():
        idx = int(np.argmax(np.abs(dp)))
        max_dP[label] = float(np.abs(dp[idx]))
        edge_obj = next(
            e
            for e in edges
            if (isinstance(e, OrificeEdge) and (e.label or f"orifice({e.a}->{e.b})") == label)
            or (isinstance(e, SlotChannelEdge) and (e.label or f"slot({e.a}->{e.b})") == label)
        )
        if isinstance(edge_obj, OrificeEdge):
            a, b = edge_obj.a, edge_obj.b
            Pa = float(P[a, idx])
            Pb = float(P_ext[idx] if b == EXT_NODE else P[b, idx])
            if Pa >= Pb:
                p_up, p_dn = Pa, Pb
                t_up = float(T_eff[a, idx])
            else:
                p_up, p_dn = Pb, Pa
                t_up = T0 if b == EXT_NODE else float(T_eff[b, idx])
            r_pk = (p_dn / p_up) if p_up > 0 else 0.0
            peak_diag[label] = {
                "t_peak": float(t[idx]),
                "t_peak_over_tau_exit": float(t[idx] / tau_exit) if np.isfinite(tau_exit) else float("nan"),
                "P_up": p_up,
                "P_down": p_dn,
                "r": float(r_pk),
                "regime": "CHOKED" if r_pk <= PI_C else "subsonic",
                "T_up": t_up,
                "peak_type": profile.classify_peak(float(t[idx]), float(t[-1])),
                "dP_signed": float(dp[idx]),
            }
        else:
            a, b = edge_obj.a, edge_obj.b
            Pa = float(P[a, idx])
            Pb = float(P[b, idx])
            peak_diag[label] = {
                "t_peak": float(t[idx]),
                "t_peak_over_tau_exit": float(t[idx] / tau_exit) if np.isfinite(tau_exit) else float("nan"),
                "P_up": max(Pa, Pb),
                "P_down": min(Pa, Pb),
                "r": float(min(Pa, Pb) / max(Pa, Pb)) if max(Pa, Pb) > 0 else 0.0,
                "regime": "viscous_slot",
                "T_up": float(T_eff[a, idx] if Pa >= Pb else T_eff[b, idx]),
                "peak_type": profile.classify_peak(float(t[idx]), float(t[-1])),
                "dP_signed": float(dp[idx]),
            }

    meta = {
        "case": asdict(case),
        "nodes": [asdict(n) for n in nodes],
        "edges": [asdict(e) for e in edges],
        "profile": {"name": profile.name, "events": list(profile.events)},
        "solver": {
            "success": bool(sol.success),
            "message": str(sol.message),
            "t_end": float(sol.t[-1]),
            "n_steps": int(len(sol.t)),
        },
    }
    return SolveResult(t=t, m=m, T=T, P=P, P_ext=P_ext, peak_diag=peak_diag, max_dP=max_dP, tau_exit=tau_exit, meta=meta)
