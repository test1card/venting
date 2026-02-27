from pathlib import Path

import pytest

from venting.cases import CaseConfig
from venting.constants import T0
from venting.diagnostics import summarize_result
from venting.geometry import circle_area_from_d_mm
from venting.graph import EXT_NODE, CdConst, ExternalBC, GasNode, OrificeEdge
from venting.profiles import Profile, make_profile_from_table
from venting.solver import solve_case


def test_profile_table_rejects_mmhg_looking_values_when_unit_pa(tmp_path: Path):
    p = tmp_path / "profile.csv"
    p.write_text("0,760\n1,740\n", encoding="utf-8")
    with pytest.raises(ValueError, match="looks like mmHg"):
        make_profile_from_table("tab", p, pressure_unit="Pa")


def test_profile_table_accepts_mmhg_when_explicit(tmp_path: Path):
    p = tmp_path / "profile.csv"
    p.write_text("0,760\n1,740\n", encoding="utf-8")
    prof = make_profile_from_table("tab", p, pressure_unit="mmHg")
    assert prof.P(0.0) > 1.0e5


def test_validity_flags_are_emitted_in_meta():
    V = 131.6e-6
    A = circle_area_from_d_mm(2.0)
    profile = Profile("vacuum", lambda t: 0.0, events=((0.0, "vacuum"),))
    nodes = [GasNode("node", V, 181.6e-4)]
    edges = [OrificeEdge(0, EXT_NODE, A, CdConst(0.62), label="exit")]
    bcs = [ExternalBC(0, profile, T_ext=T0)]
    case = CaseConfig("intermediate", 0.0, T0, 0.8, 300)
    sol = solve_case(nodes, edges, bcs, case)
    res = summarize_result(nodes, edges, bcs, case, sol)

    flags = res.meta.get("validity_flags", {})
    assert "state_integrity" in flags
    assert "acoustic_uniformity_0D" in flags
    assert "external_pressure_units" in flags
    assert "thermo_fit_range" in flags
    assert "knudsen_regime" in flags
    assert flags["state_integrity"]["status"] in {"ok", "warning", "fail"}


def test_knudsen_reference_values():
    import math

    from venting.constants import D_MOL_AIR, K_BOLTZMANN

    def kn(p, t=300.0, d=2e-3):
        lam = K_BOLTZMANN * t / (math.sqrt(2.0) * math.pi * (D_MOL_AIR**2) * p)
        return lam / d

    assert kn(101325.0) < 0.01
    assert kn(1.0) >= 0.1
    assert 0.01 <= kn(100.0) < 0.1
