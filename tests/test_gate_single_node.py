import pytest

from venting.gates import gate_single


def test_gate_single_node_matches_analytics():
    m = gate_single()
    assert m.errP < 5e-3
    assert m.errT < 5e-3
    assert m.errMass < 1e-3
