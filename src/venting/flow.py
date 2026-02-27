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


def friction_factor(Re: float, eps_over_D: float) -> float:
    """Darcy friction factor.

    Laminar: 64/Re.
    Turbulent: Swamee–Jain approximation.
    """
    if Re <= 0.0:
        return 0.0
    if Re < 2300.0:
        return 64.0 / Re
    term = eps_over_D / 3.7 + 5.74 / (Re**0.9)
    return 0.25 / (math.log10(max(term, 1e-20)) ** 2)


def mdot_short_tube_pos(
    P_up: float,
    T_up: float,
    P_dn: float,
    Cd0: float,
    A_total: float,
    D: float,
    L: float,
    eps: float,
    K_in: float,
    K_out: float,
) -> float:
    """Lossy-nozzle short-tube model via effective discharge coefficient.

    This is *not* Fanno-flow choking; it applies friction/minor-loss correction
    through ``Cd_eff = Cd0/sqrt(1 + K_tot)`` with
    ``K_tot = K_in + K_out + 4*f*(L/D)``.
    """
    if P_up <= 0.0 or A_total <= 0.0 or D <= 0.0:
        return 0.0

    t_eff = max(T_up, T_SAFE)
    if L <= 0.0 and K_in <= 0.0 and K_out <= 0.0:
        return mdot_orifice_pos(P_up, t_eff, P_dn, Cd0, A_total)

    Cd_eff = float(Cd0)
    mdot = 0.0
    for _ in range(2):
        mdot_new = mdot_orifice_pos(P_up, t_eff, P_dn, Cd_eff, A_total)
        rho = P_up / (R_GAS * t_eff)
        u = mdot_new / max(rho * A_total, 1e-18)
        mu = mu_air_sutherland(t_eff)
        Re = rho * u * D / max(mu, 1e-18)
        f = friction_factor(Re, eps / max(D, 1e-12))
        K_tot = K_in + K_out + 4.0 * f * (L / max(D, 1e-12))
        Cd_eff = Cd0 / math.sqrt(1.0 + max(K_tot, 0.0))
        mdot = 0.5 * mdot + 0.5 * mdot_new

    return max(mdot, 0.0)
