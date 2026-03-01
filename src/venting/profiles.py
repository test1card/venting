import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Profile:
    name: str
    P: Callable[[float], float]
    events: tuple[tuple[float, str], ...]

    def P_array(self, t: np.ndarray) -> np.ndarray:
        return np.array([self.P(float(tt)) for tt in t], dtype=float)

    def classify_peak(self, t_peak: float, t_end: float, tol_s: float = 0.5) -> str:
        for te, label in self.events:
            if abs(t_peak - te) <= tol_s:
                return f"boundary({label})"
        if t_peak <= tol_s:
            return "boundary(sim_start)"
        if t_peak >= (t_end - tol_s):
            return "boundary(sim_end)"
        return "internal"


def make_profile_linear(p0: float, rate_mmhg_per_s: float) -> Profile:
    rate = rate_mmhg_per_s * 133.322
    t_zero = p0 / rate

    def p_fn(t: float) -> float:
        return max(p0 - rate * t, 0.0)

    return Profile("linear", p_fn, events=((t_zero, "P_ext=0"),))


def make_profile_step(p0: float, step_time_s: float) -> Profile:
    def p_fn(t: float) -> float:
        return 0.0 if t >= step_time_s else p0

    return Profile("step", p_fn, events=((step_time_s, "step_to_vacuum"),))


def make_profile_exponential(
    p0: float, rate0_mmhg_per_s: float, p_floor: float = 10.0
) -> Profile:
    if rate0_mmhg_per_s <= 0.0:
        raise ValueError("rate0_mmhg_per_s must be positive")
    if p_floor <= 0.0:
        raise ValueError("p_floor must be positive")
    rate0 = rate0_mmhg_per_s * 133.322
    tau = p0 / rate0
    t_floor = tau * math.log(p0 / p_floor)

    def p_fn(t: float) -> float:
        return max(p0 * math.exp(-t / tau), p_floor)

    return Profile(
        "barometric_exp",
        p_fn,
        events=((0.0, "start_max_slope"), (t_floor, f"P_ext={p_floor:.0f}Pa")),
    )


def make_profile_from_table(
    name: str,
    table_path: Path,
    pressure_unit: str = "Pa",
) -> Profile:
    """Load profile table with CSV columns: t_s,P_Pa.

    pressure_unit: "Pa" | "mmHg"
    """
    arr = np.loadtxt(str(table_path), delimiter=",")
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("Profile table must be CSV with columns: t_s, P_Pa")
    t_tab = np.array(arr[:, 0], dtype=float)
    p_tab = np.array(arr[:, 1], dtype=float)
    if not np.all(np.diff(t_tab) > 0):
        raise ValueError("Profile table time must be strictly increasing")

    if pressure_unit.lower() == "mmhg":
        p_tab = p_tab * 133.322
    elif pressure_unit.lower() != "pa":
        raise ValueError("pressure_unit must be 'Pa' or 'mmHg'")

    p_max = float(np.max(p_tab))
    if pressure_unit.lower() == "pa" and 200.0 <= p_max <= 2000.0:
        warnings.warn(
            "Profile table pressure looks like mmHg values provided as Pa. "
            "Use pressure_unit='mmHg' or convert CSV to Pa.",
            stacklevel=2,
        )

    def p_fn(t: float) -> float:
        if t <= t_tab[0]:
            return float(p_tab[0])
        if t >= t_tab[-1]:
            return float(p_tab[-1])
        return float(np.interp(t, t_tab, p_tab))

    events = tuple((float(tt), f"bp{i}") for i, tt in enumerate(t_tab))
    return Profile(name, p_fn, events=events)
