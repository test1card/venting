import math

from .constants import C_CHOKED, GAMMA, PI_C, R_GAS, T_SAFE


def mu_air_sutherland(T: float) -> float:
    t_eff = max(T, T_SAFE)
    mu0 = 1.716e-5
    t_ref = 273.15
    s = 111.0
    return mu0 * (t_eff / t_ref) ** 1.5 * (t_ref + s) / (t_eff + s)


def mdot_orifice_pos(
    P_up: float, T_up: float, P_dn: float, Cd: float, A: float
) -> float:
    if P_up <= 0.0 or A <= 0.0:
        return 0.0
    t_eff = max(T_up, T_SAFE)
    r = max(P_dn, 0.0) / P_up
    if r >= 1.0:
        return 0.0
    if r <= PI_C:
        return Cd * A * P_up * C_CHOKED / math.sqrt(R_GAS * t_eff)
    bracket = r ** (2.0 / GAMMA) - r ** ((GAMMA + 1.0) / GAMMA)
    if bracket <= 0.0:
        return 0.0
    return (
        Cd
        * A
        * P_up
        * math.sqrt(2.0 * GAMMA / ((GAMMA - 1.0) * R_GAS * t_eff) * bracket)
    )


def mdot_slot_pos(
    P_up: float, T_up: float, P_dn: float, w: float, delta: float, L: float
) -> float:
    if P_up <= 0.0:
        return 0.0
    t_eff = max(T_up, T_SAFE)
    mu = mu_air_sutherland(t_eff)
    K = w * (delta**3) / (12.0 * mu * L)
    dp2 = max(P_up**2 - max(P_dn, 0.0) ** 2, 0.0)
    return K * dp2 / (R_GAS * t_eff)
