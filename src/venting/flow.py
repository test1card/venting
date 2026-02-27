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
    return (
        Cd
        * A
        * P_up
        * math.sqrt(2.0 * gamma / ((gamma - 1.0) * r_gas * t_eff) * bracket)
    )


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


def _fanno_length_to_sonic(M: float, gamma: float) -> float:
    m2 = max(M * M, 1e-12)
    term1 = (1.0 - m2) / (gamma * m2)
    term2 = ((gamma + 1.0) / (2.0 * gamma)) * math.log(
        ((gamma + 1.0) * m2) / (2.0 + (gamma - 1.0) * m2)
    )
    return term1 + term2


def _p_over_pstar(M: float, gamma: float) -> float:
    return (1.0 / max(M, 1e-12)) * math.sqrt(
        (gamma + 1.0) / (2.0 + (gamma - 1.0) * M * M)
    )


def _solve_m2_from_m1(m1: float, f4ld: float, gamma: float) -> tuple[float, bool]:
    f1 = _fanno_length_to_sonic(m1, gamma)
    target = f1 - f4ld
    if target <= 0.0:
        return 1.0, True
    lo = m1
    hi = 1.0 - 1e-7
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _fanno_length_to_sonic(mid, gamma) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), False


def _fanno_state(
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
    gamma: float,
    r_gas: float,
):
    if P_up <= 0.0 or A_total <= 0.0 or D <= 0.0:
        return {"mdot": 0.0, "choked": False, "mach_exit": 0.0, "f_D": 0.0}

    t_eff = max(T_up, T_SAFE)
    if L <= 0.0:
        md = mdot_orifice_pos_props(P_up, t_eff, P_dn, Cd0, A_total, gamma, r_gas)
        return {"mdot": md, "choked": False, "mach_exit": 0.0, "f_D": 0.0}

    if max(P_dn, 0.0) / P_up > 0.8:
        md = mdot_short_tube_pos(
            P_up,
            t_eff,
            P_dn,
            Cd0,
            A_total,
            D,
            L,
            eps,
            K_in,
            K_out,
            gamma,
            r_gas,
        )
        return {"mdot": md, "choked": False, "mach_exit": 0.0, "f_D": 0.0}

    a_eff = Cd0 * A_total
    re_guess = 1e5
    m1_solution = 0.1
    m2_solution = 0.1
    choked_solution = False
    f_D = 0.03

    for _ in range(5):
        f_D = max(friction_factor(re_guess, eps / max(D, 1e-12)), 1e-4)
        f4ld = 4.0 * f_D * L / max(D, 1e-12)

        if f4ld < 1e-3:
            md = mdot_short_tube_pos(
                P_up,
                t_eff,
                P_dn,
                Cd0,
                A_total,
                D,
                L,
                eps,
                K_in,
                K_out,
                gamma,
                r_gas,
            )
            return {"mdot": md, "choked": False, "mach_exit": 0.0, "f_D": f_D}

        lo = 1e-5
        hi = 1.0 - 1e-7
        for _i in range(80):
            mid = 0.5 * (lo + hi)
            if _fanno_length_to_sonic(mid, gamma) > f4ld:
                lo = mid
            else:
                hi = mid
        m1_cap = 0.5 * (lo + hi)

        def outlet_pressure(
            m1: float, f4ld_local: float = f4ld
        ) -> tuple[float, float, bool]:
            m2, choked = _solve_m2_from_m1(m1, f4ld_local, gamma)
            p1 = P_up / (1.0 + 0.5 * (gamma - 1.0) * m1 * m1) ** (gamma / (gamma - 1.0))
            t1 = t_eff / (1.0 + 0.5 * (gamma - 1.0) * m1 * m1)
            p2 = p1 * (_p_over_pstar(m2, gamma) / _p_over_pstar(m1, gamma))
            p_out = p2 / (1.0 + K_out)
            if K_in > 0.0:
                p_out /= 1.0 + K_in
            return p_out, t1, choked

        p_cap, _t1_cap, _ = outlet_pressure(m1_cap)
        if max(P_dn, 0.0) <= p_cap:
            m1_solution = m1_cap
            m2_solution = 1.0
            choked_solution = True
        else:
            lo = 1e-5
            hi = m1_cap
            for _j in range(60):
                mid = 0.5 * (lo + hi)
                p_mid, _, _ = outlet_pressure(mid)
                if p_mid > P_dn:
                    lo = mid
                else:
                    hi = mid
            m1_solution = 0.5 * (lo + hi)
            m2_solution, choked_solution = _solve_m2_from_m1(m1_solution, f4ld, gamma)

        t1 = t_eff / (1.0 + 0.5 * (gamma - 1.0) * m1_solution * m1_solution)
        mdot = P_up * m1_solution * a_eff * math.sqrt(gamma / (r_gas * max(t1, T_SAFE)))
        rho1 = P_up / (r_gas * max(t_eff, T_SAFE))
        u1 = mdot / max(rho1 * A_total, 1e-18)
        re_new = rho1 * u1 * D / max(mu_air_sutherland(t_eff), 1e-18)
        if abs(re_new - re_guess) / max(re_guess, 1.0) < 0.05:
            re_guess = re_new
            break
        re_guess = 0.5 * re_guess + 0.5 * re_new

    mdot = (
        P_up
        * m1_solution
        * a_eff
        * math.sqrt(
            gamma
            / (
                r_gas
                * max(
                    t_eff / (1.0 + 0.5 * (gamma - 1.0) * m1_solution * m1_solution),
                    T_SAFE,
                )
            )
        )
    )
    return {
        "mdot": max(mdot, 0.0),
        "choked": bool(choked_solution),
        "mach_exit": float(m2_solution),
        "f_D": float(f_D),
    }


def mdot_fanno_tube(
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
    """Mass flow in a short tube including Fanno friction choking.

    Reference: Shapiro, Dynamics and Thermodynamics of Compressible Fluid Flow,
    1953, Chapter 6.
    """
    if L <= 0.0:
        return mdot_orifice_pos_props(P_up, T_up, P_dn, Cd0, A_total, gamma, r_gas)

    md_f = _fanno_state(
        P_up,
        T_up,
        P_dn,
        Cd0,
        A_total,
        D,
        L,
        eps,
        K_in,
        K_out,
        gamma,
        r_gas,
    )["mdot"]
    md_l = mdot_short_tube_pos(
        P_up, T_up, P_dn, Cd0, A_total, D, L, eps, K_in, K_out, gamma, r_gas
    )
    md = min(md_f, md_l)
    if P_up > 0 and (P_dn / P_up) < 0.2:
        md = min(md, 0.99 * md_l)
    return md


def fanno_choked_state(
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
) -> tuple[bool, float]:
    state = _fanno_state(
        P_up,
        T_up,
        P_dn,
        Cd0,
        A_total,
        D,
        L,
        eps,
        K_in,
        K_out,
        gamma,
        r_gas,
    )
    return bool(state["choked"]), float(state["mach_exit"])
