import numpy as np

from venting.cases import CaseConfig
from venting.constants import T0
from venting.flow import mdot_orifice_pos, mdot_short_tube_pos
from venting.geometry import circle_area_from_d_mm
from venting.graph import CdConst, ExternalBC, GasNode, ShortTubeEdge
from venting.profiles import Profile
from venting.solver import solve_case


def test_short_tube_L0_matches_orifice():
    P_up = 101325.0
    P_dn = 50000.0
    T_up = 300.0
    Cd0 = 0.62
    A = circle_area_from_d_mm(2.0)
    D = 2.0e-3

    m_or = mdot_orifice_pos(P_up, T_up, P_dn, Cd0, A)
    m_st = mdot_short_tube_pos(P_up, T_up, P_dn, Cd0, A, D, 0.0, 0.0, 0.0, 0.0)
    rel = abs(m_st - m_or) / max(m_or, 1e-20)
    assert rel < 0.02


def test_short_tube_mdot_decreases_with_L():
    P_up = 101325.0
    P_dn = 30000.0
    T_up = 300.0
    Cd0 = 0.62
    A = circle_area_from_d_mm(2.0)
    D = 2.0e-3

    m_05 = mdot_short_tube_pos(P_up, T_up, P_dn, Cd0, A, D, 0.5e-3, 0.0, 0.5, 1.0)
    m_10 = mdot_short_tube_pos(P_up, T_up, P_dn, Cd0, A, D, 1.0e-3, 0.0, 0.5, 1.0)
    m_20 = mdot_short_tube_pos(P_up, T_up, P_dn, Cd0, A, D, 2.0e-3, 0.0, 0.5, 1.0)

    assert m_05 > m_10 > m_20


def test_short_tube_in_network_runs():
    V = 131.6e-6
    A = circle_area_from_d_mm(2.0)
    profile = Profile("vacuum", lambda t: 0.0, events=((0.0, "vacuum"),))
    nodes = [GasNode("node", V, 181.6e-4)]
    edges = [
        ShortTubeEdge(
            a=0,
            b=-1,
            A_total=A,
            D=2.0e-3,
            L=1.0e-3,
            eps=0.0,
            K_in=0.5,
            K_out=1.0,
            Cd_model=CdConst(0.62),
            label="exit_short_tube",
        )
    ]
    bcs = [ExternalBC(0, profile, T_ext=T0)]
    case = CaseConfig("intermediate", 0.0, T0, 0.5, 200)
    sol = solve_case(nodes, edges, bcs, case)
    assert sol.success
    assert np.isfinite(sol.y).all()
