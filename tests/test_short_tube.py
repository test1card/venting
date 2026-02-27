import numpy as np

from venting.cases import CaseConfig
from venting.constants import T0
from venting.diagnostics import summarize_result
from venting.flow import mdot_fanno_tube, mdot_orifice_pos, mdot_short_tube_pos
from venting.geometry import circle_area_from_d_mm
from venting.graph import CdConst, ExternalBC, GasNode, ShortTubeEdge
from venting.profiles import Profile
from venting.solver import solve_case


def test_short_tube_L0_matches_orifice():
    p_up = 101325.0
    p_dn = 50000.0
    t_up = 300.0
    cd0 = 0.62
    area = circle_area_from_d_mm(2.0)
    diam = 2.0e-3

    m_or = mdot_orifice_pos(p_up, t_up, p_dn, cd0, area)
    m_st = mdot_short_tube_pos(p_up, t_up, p_dn, cd0, area, diam, 0.0, 0.0, 0.0, 0.0)
    rel = abs(m_st - m_or) / max(m_or, 1e-20)
    assert rel < 0.02


def test_short_tube_mdot_decreases_with_L():
    p_up = 101325.0
    p_dn = 30000.0
    t_up = 300.0
    cd0 = 0.62
    area = circle_area_from_d_mm(2.0)
    diam = 2.0e-3

    m_05 = mdot_short_tube_pos(p_up, t_up, p_dn, cd0, area, diam, 0.5e-3, 0.0, 0.5, 1.0)
    m_10 = mdot_short_tube_pos(p_up, t_up, p_dn, cd0, area, diam, 1.0e-3, 0.0, 0.5, 1.0)
    m_20 = mdot_short_tube_pos(p_up, t_up, p_dn, cd0, area, diam, 2.0e-3, 0.0, 0.5, 1.0)

    assert m_05 > m_10 > m_20


def test_short_tube_mdot_decreases_with_roughness():
    p_up = 2.0e5
    p_dn = 1.0e5
    t_up = 300.0
    cd0 = 0.62
    area = circle_area_from_d_mm(4.0)
    diam = 4.0e-3

    m_smooth = mdot_short_tube_pos(
        p_up, t_up, p_dn, cd0, area, diam, 3e-3, 0.0, 0.5, 1.0
    )
    m_rough = mdot_short_tube_pos(
        p_up, t_up, p_dn, cd0, area, diam, 3e-3, 200e-6, 0.5, 1.0
    )
    assert m_smooth > m_rough


def test_short_tube_validity_fields_present():
    vol = 131.6e-6
    area = circle_area_from_d_mm(2.0)
    profile = Profile("vacuum", lambda t: 0.0, events=((0.0, "vacuum"),))
    nodes = [GasNode("node", vol, 181.6e-4)]
    edges = [
        ShortTubeEdge(
            a=0,
            b=-1,
            A_total=area,
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
    res = summarize_result(nodes, edges, bcs, case, sol)
    flag = res.meta["validity_flags"]["short_tube_flow"]

    assert "Mach_max" in flag
    assert "Re_max" in flag
    assert "Cd_eff_min" in flag
    assert "K_tot_max" in flag
    assert "frac_fric" in flag


def test_short_tube_in_network_runs():
    vol = 131.6e-6
    area = circle_area_from_d_mm(2.0)
    profile = Profile("vacuum", lambda t: 0.0, events=((0.0, "vacuum"),))
    nodes = [GasNode("node", vol, 181.6e-4)]
    edges = [
        ShortTubeEdge(
            a=0,
            b=-1,
            A_total=area,
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


def test_fanno_choked_less_than_isentropic():
    p_up = 101325.0
    p_dn = 0.0
    t_up = 300.0
    cd0 = 0.62
    area = circle_area_from_d_mm(2.0)
    diam = 2.0e-3
    m_fanno = mdot_fanno_tube(p_up, t_up, p_dn, cd0, area, diam, 3e-3, 0.0, 0.5, 1.0)
    m_lossy = mdot_short_tube_pos(
        p_up, t_up, p_dn, cd0, area, diam, 3e-3, 0.0, 0.5, 1.0
    )
    assert m_fanno < m_lossy


def test_fanno_zero_length_matches_orifice():
    p_up = 101325.0
    p_dn = 50000.0
    t_up = 300.0
    cd0 = 0.62
    area = circle_area_from_d_mm(2.0)
    diam = 2.0e-3
    m_or = mdot_orifice_pos(p_up, t_up, p_dn, cd0, area)
    m_f = mdot_fanno_tube(p_up, t_up, p_dn, cd0, area, diam, 0.0, 0.0, 0.5, 1.0)
    assert abs(m_f - m_or) / max(m_or, 1e-20) < 1e-3


def test_fanno_low_mach_matches_lossy_nozzle():
    p_up = 101325.0
    p_dn = 0.9 * p_up
    t_up = 300.0
    cd0 = 0.62
    area = circle_area_from_d_mm(2.0)
    diam = 2.0e-3
    m_fanno = mdot_fanno_tube(p_up, t_up, p_dn, cd0, area, diam, 3e-3, 0.0, 0.5, 1.0)
    m_lossy = mdot_short_tube_pos(
        p_up, t_up, p_dn, cd0, area, diam, 3e-3, 0.0, 0.5, 1.0
    )
    assert abs(m_fanno - m_lossy) / max(m_lossy, 1e-20) < 0.05


def test_fanno_choking_mach_limit():
    p_up = 101325.0
    t_up = 300.0
    cd0 = 0.62
    area = circle_area_from_d_mm(2.0)
    diam = 2.0e-3
    m1 = mdot_fanno_tube(p_up, t_up, 5000.0, cd0, area, diam, 3e-3, 0.0, 0.5, 1.0)
    m2 = mdot_fanno_tube(p_up, t_up, 500.0, cd0, area, diam, 3e-3, 0.0, 0.5, 1.0)
    assert m2 <= 1.05 * m1
