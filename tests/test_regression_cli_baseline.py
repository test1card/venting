from venting.gates import gate_single, gate_two


def test_gate_baseline_regression_stable():
    g1 = gate_single()
    g2 = gate_two()
    assert g1.errP < 1e-6
    assert g1.errT < 1e-6
    assert g1.errMass < 1e-4
    assert g2.errMass < 1e-4
    assert g2.errEnergy is not None and g2.errEnergy < 1e-4
