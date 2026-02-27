import math


def mm_to_m(x_mm: float) -> float:
    return x_mm * 1e-3


def mm2_to_m2(x_mm2: float) -> float:
    return x_mm2 * 1e-6


def mm3_to_m3(x_mm3: float) -> float:
    return x_mm3 * 1e-9


def circle_area_from_d_mm(d_mm: float) -> float:
    d_m = mm_to_m(d_mm)
    return math.pi * (d_m / 2.0) ** 2


def assert_pos(name: str, val: float) -> None:
    if not (val > 0.0):
        raise ValueError(f"{name} must be > 0, got {val}")
