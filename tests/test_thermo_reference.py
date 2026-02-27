import numpy as np

from venting.thermo import cp_air, gamma_air, h_air


def test_cp_at_300K():
    assert abs(cp_air(300.0) - 1006.0) < 5.0


def test_gamma_at_300K():
    assert abs(gamma_air(300.0) - 1.4) < 0.005


def test_cp_monotonic_200_to_600():
    temps = np.linspace(200.0, 600.0, 20)
    cps = [cp_air(float(t)) for t in temps]
    assert all(cps[i] <= cps[i + 1] for i in range(len(cps) - 1))


def test_h_zero_at_reference():
    assert abs(h_air(298.15)) < 1.0
