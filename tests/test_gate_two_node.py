from venting.gates import gate_two


def test_gate_two_node_mass_energy_conservation():
    m = gate_two()
    assert m.errMass < 1e-3
    assert m.errEnergy is not None
    assert m.errEnergy < 1e-2
