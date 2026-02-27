from __future__ import annotations

import math

import numpy as np

from .constants import GAMMA, R_GAS, T_SAFE
from .flow import mu_air_sutherland
from .graph import SlotChannelEdge


def _acoustic_flag(
    P: np.ndarray, T: np.ndarray, t: np.ndarray, l_char_m: float
) -> dict:
    t_mean = float(np.mean(np.maximum(T, T_SAFE)))
    c = math.sqrt(GAMMA * R_GAS * t_mean)
    t_ac = l_char_m / max(c, 1e-12)

    dP_dt = np.gradient(P, t, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_dyn = np.abs(P / dP_dt)
    t_dyn = t_dyn[np.isfinite(t_dyn) & (t_dyn > 0)]
    t_dyn_min = float(np.min(t_dyn)) if t_dyn.size else float("inf")

    ratio = t_dyn_min / t_ac if t_ac > 0 else float("inf")
    if ratio >= 20.0:
        status = "ok"
    elif ratio >= 5.0:
        status = "warning"
    else:
        status = "warning"

    return {
        "status": status,
        "t_acoustic_s": t_ac,
        "t_dynamic_min_s": t_dyn_min,
        "ratio_t_dyn_to_t_ac": ratio,
        "message": "0D pressure uniformity is stronger when ratio >> 1",
    }


def _slot_laminar_flag(
    edges: list,
    P: np.ndarray,
    T: np.ndarray,
    node_index: dict[int, int],
) -> dict:
    re_max = 0.0
    for e in edges:
        if not isinstance(e, SlotChannelEdge):
            continue
        a = node_index[e.a]
        b = node_index[e.b]
        for k in range(P.shape[1]):
            if P[a, k] >= P[b, k]:
                up = a
            else:
                up = b
            p_up = max(float(P[up, k]), 0.0)
            t_up = max(float(T[up, k]), T_SAFE)
            rho = p_up / (R_GAS * t_up)
            mu = mu_air_sutherland(t_up)
            # local velocity scale from pressure-driven slot relation
            u = (
                (e.delta**2)
                * abs(float(P[a, k] - P[b, k]))
                / (12.0 * mu * max(e.L, 1e-12))
            )
            d_h = 2.0 * e.delta
            re = rho * u * d_h / max(mu, 1e-20)
            re_max = max(re_max, re)

    if re_max == 0.0:
        return {
            "status": "ok",
            "Re_max": 0.0,
            "message": "No slot/channel edges in this case",
        }
    status = "ok" if re_max < 1000.0 else "warning"
    return {
        "status": status,
        "Re_max": re_max,
        "message": "Slot model assumes laminar flow; warning if Re approaches turbulent regime",
    }


def _state_flag(m: np.ndarray, T: np.ndarray) -> dict:
    min_m = float(np.min(m))
    min_t = float(np.min(T))
    if np.any(~np.isfinite(m)) or np.any(~np.isfinite(T)):
        return {"status": "fail", "message": "NaN/Inf detected in state arrays"}
    if min_m < -1e-12:
        return {
            "status": "fail",
            "min_m_kg": min_m,
            "message": "Negative mass detected",
        }
    if min_t < 0.0:
        return {
            "status": "fail",
            "min_T_K": min_t,
            "message": "Negative temperature detected",
        }
    if min_t < T_SAFE:
        return {
            "status": "warning",
            "min_T_K": min_t,
            "message": "Temperature reached denominator safety floor",
        }
    return {"status": "ok", "min_m_kg": min_m, "min_T_K": min_t}


def evaluate_validity_flags(
    nodes: list,
    edges: list,
    P: np.ndarray,
    T: np.ndarray,
    m: np.ndarray,
    P_ext: np.ndarray,
    t: np.ndarray,
    l_char_m: float,
) -> dict:
    node_index = {i: i for i in range(len(nodes))}
    flags = {
        "state_integrity": _state_flag(m, T),
        "acoustic_uniformity_0D": _acoustic_flag(P, T, t, l_char_m),
        "slot_laminarity": _slot_laminar_flag(edges, P, T, node_index),
    }

    max_ext = float(np.max(P_ext)) if P_ext.size else 0.0
    flags["external_pressure_units"] = {
        "status": "warning" if 200.0 <= max_ext <= 2000.0 else "ok",
        "max_P_ext_Pa": max_ext,
        "message": "Values in 200-2000 range are often mmHg entered as Pa",
    }
    return flags
