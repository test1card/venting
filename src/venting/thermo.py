"""Temperature-dependent thermodynamic properties for air (engineering fit)."""

from __future__ import annotations

import math

from .constants import R_GAS, T0


def cp_air(T: float) -> float:
    t = max(T, 1.0)
    x = t - 300.0
    cp = 1006.0 + 0.12 * x + 1.2e-4 * x * x
    return max(cp, 850.0)


def cv_air(T: float) -> float:
    return cp_air(T) - R_GAS


def gamma_air(T: float) -> float:
    cp = cp_air(T)
    cv = max(cv_air(T), 1e-9)
    return cp / cv


def h_air(T: float) -> float:
    # Integral of cp polynomial from 300 K
    t = max(T, 1.0)
    x = t - 300.0
    return 1006.0 * x + 0.5 * 0.12 * x * x + (1.2e-4 / 3.0) * x * x * x


def u_air(T: float) -> float:
    return h_air(T) - R_GAS * (max(T, 1.0) - T0)


def speed_of_sound(T: float) -> float:
    t = max(T, 1.0)
    return math.sqrt(gamma_air(t) * R_GAS * t)
