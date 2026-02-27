import math

from .constants import GAMMA, R_GAS, T_SAFE


def mu_air_sutherland(T: float) -> float:
    t_eff = max(T, T_SAFE)
    mu0 = 1.716e-5
    t_ref = 273.15
    s = 111.0
    return mu0 * (t_eff / t_ref) ** 1.5 * (t_ref + s) / (t_eff + s)


def mdot_orifice_pos_props(
    P_up: float,
    T_up: float,
    P_dn: float,
    Cd: float,
    A: float,
    gamma: float = GAMMA,
    r_gas: float = R_GAS,
) -> float:
    if P_up <= 0.0 or A <= 0.0:
        return 0.0
    t_eff = max(T_up, T_SAFE)
    r = max(P_dn, 0.0) / P_up
    if r >= 1.0:
        return 0.0

    pi_c = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
    c_choked = math.sqrt(
        gamma * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (gamma - 1.0))
    )

    if r <= pi_c:
        return Cd * A * P_up * c_choked / math.sqrt(r_gas * t_eff)

    bracket = r ** (2.0 / gamma) - r ** ((gamma + 1.0) / gamma)
    if bracket <= 0.0:
        return 0.0
    return Cd * A * P_up * math.sqrt(2.0 * gamma / ((gamma - 1.0) * r_gas * t_eff) * bracket)


def mdot_orifice_pos(
    P_up: float, T_up: float, P_dn: float, Cd: float, A: float
) -> float:
    return mdot_orifice_pos_props(P_up, T_up, P_dn, Cd, A, gamma=GAMMA, r_gas=R_GAS)


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
    """Darcy friction factor (not Fanning)."""
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
    gamma: float = GAMMA,
    r_gas: float = R_GAS,
) -> float:
    """Lossy nozzle via effective Cd for short-tube (thick wall).

    This model applies additional minor/friction losses via Cd_eff and does not
    model Fanno friction choking.
    """
    if P_up <= 0.0 or A_total <= 0.0 or D <= 0.0:
        return 0.0

    t_eff = max(T_up, T_SAFE)
    if L <= 0.0 and K_in <= 0.0 and K_out <= 0.0 and eps <= 0.0:
        return mdot_orifice_pos_props(
            P_up, t_eff, P_dn, Cd0, A_total, gamma=gamma, r_gas=r_gas
        )

    Cd_eff = float(Cd0)
    mdot = 0.0
    for _ in range(5):
        mdot_new = mdot_orifice_pos_props(
            P_up, t_eff, P_dn, Cd_eff, A_total, gamma=gamma, r_gas=r_gas
        )
        rho = P_up / (r_gas * t_eff)
        u = mdot_new / max(rho * A_total, 1e-18)
        mu = mu_air_sutherland(t_eff)
        Re = rho * u * D / max(mu, 1e-18)
        f_D = friction_factor(Re, eps / max(D, 1e-12))
        K_fric = f_D * (L / max(D, 1e-12))
        K_tot = K_in + K_out + K_fric
        Cd_eff_new = 1.0 / math.sqrt(max(Cd0 ** (-2.0) + K_tot, 1e-18))

        if mdot > 0 and abs(mdot_new - mdot) / max(mdot_new, 1e-18) < 0.01:
            mdot = mdot_new
            break
        mdot = 0.5 * mdot + 0.5 * mdot_new
        Cd_eff = 0.5 * Cd_eff + 0.5 * Cd_eff_new

    return max(mdot, 0.0)
