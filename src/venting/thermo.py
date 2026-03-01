"""Temperature-dependent thermodynamic properties for dry air.

NASA-7 polynomial fit (200-1000 K) for dry-air mixture:
0.78084 N2 + 0.20946 O2 + 0.00934 Ar.
Source: McBride, Gordon, Reno, NASA TM-4513 (1993).
"""

from __future__ import annotations

import math

from .constants import R_GAS

# NASA-7 coefficients for cp/R, valid for 200-1000 K.
_A1 = 3.53562097
_A2 = -4.15142542e-4
_A3 = 1.03930790e-6
_A4 = 0.0
_A5 = 0.0

T_FIT_LOW = 200.0
T_FIT_HIGH = 1000.0
_T_REF = 298.15


def _cp_over_r(t: float) -> float:
    return _A1 + _A2 * t + _A3 * t**2 + _A4 * t**3 + _A5 * t**4


def cp_air(T: float) -> float:
    """Specific heat cp for dry air [J/(kg·K)] using NASA-7 fit."""
    t = min(max(float(T), 1.0), T_FIT_HIGH)
    return max(_cp_over_r(t) * R_GAS, 700.0)


def cv_air(T: float) -> float:
    return cp_air(T) - R_GAS


def gamma_air(T: float) -> float:
    cp = cp_air(T)
    cv = max(cv_air(T), 1e-12)
    return cp / cv


def _h_over_r(t: float) -> float:
    return (
        _A1 * t
        + 0.5 * _A2 * t**2
        + (_A3 / 3.0) * t**3
        + 0.25 * _A4 * t**4
        + 0.2 * _A5 * t**5
    )


def h_air(T: float) -> float:
    """Specific enthalpy offset from 298.15 K [J/kg]."""
    t = min(max(float(T), 1.0), T_FIT_HIGH)
    return (_h_over_r(t) - _h_over_r(_T_REF)) * R_GAS


def u_air(T: float) -> float:
    t = min(max(float(T), 1.0), T_FIT_HIGH)
    return h_air(t) - R_GAS * t


def speed_of_sound(T: float) -> float:
    t = max(float(T), 1.0)
    return math.sqrt(gamma_air(t) * R_GAS * t)
