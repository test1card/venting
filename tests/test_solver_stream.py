import numpy as np

from venting.cases import CaseConfig
from venting.constants import T0
from venting.diagnostics import summarize_result
from venting.geometry import circle_area_from_d_mm
from venting.graph import EXT_NODE, CdConst, ExternalBC, GasNode, OrificeEdge
from venting.profiles import Profile
from venting.solver import solve_case, solve_case_stream

TOL_STREAM = 1e-3


def _canonical_single_node():
    v = 131.6e-6
    a = circle_area_from_d_mm(2.0)
    profile = Profile("vacuum", lambda t: 0.0, events=((0.0, "vacuum"),))
    nodes = [GasNode("node", v, 181.6e-4)]
    edges = [OrificeEdge(0, EXT_NODE, a, CdConst(0.62), label="exit")]
    bcs = [ExternalBC(0, profile, T_ext=T0)]
    case = CaseConfig("intermediate", 0.0, T0, 0.4, 120)
    return nodes, edges, bcs, case


def test_streaming_matches_batch_pressure():
    nodes, edges, bcs, case = _canonical_single_node()
    sol_batch = solve_case(nodes, edges, bcs, case)
    sol_stream = solve_case_stream(nodes, edges, bcs, case)

    res_b = summarize_result(nodes, edges, bcs, case, sol_batch)
    res_s = summarize_result(nodes, edges, bcs, case, sol_stream)

    rel = np.max(np.abs(res_b.P - res_s.P) / np.maximum(np.abs(res_b.P), 1.0))
    assert float(rel) < TOL_STREAM


def test_streaming_callback_receives_progressive_chunks():
    nodes, edges, bcs, case = _canonical_single_node()
    seen = []

    def cb(payload):
        seen.append((payload["progress"], payload["t"].shape[0], payload["t"][-1]))

    solve_case_stream(nodes, edges, bcs, case, callback=cb, n_chunks=8)
    assert len(seen) >= 2
    assert seen[-1][0] == 1.0
    assert all(seen[i][1] <= seen[i + 1][1] for i in range(len(seen) - 1))
    assert all(seen[i][2] <= seen[i + 1][2] for i in range(len(seen) - 1))
    assert seen[0][2] < case.duration


def test_streaming_can_cancel_early():
    nodes, edges, bcs, case = _canonical_single_node()
    stop = {"value": False}

    def cb(_payload):
        stop["value"] = True

    sol = solve_case_stream(
        nodes,
        edges,
        bcs,
        case,
        callback=cb,
        n_chunks=10,
        should_stop=lambda: stop["value"],
    )
    assert sol.t[-1] < case.duration


def test_dt_chunk_default():
    nodes, edges, bcs, case = _canonical_single_node()
    case = CaseConfig(case.thermo, case.h_conv, case.T_wall, 10.0, 101)
    seen = []

    def cb(payload):
        seen.append(payload["progress"])

    solve_case_stream(nodes, edges, bcs, case, callback=cb, dt_chunk_s=2.0)
    assert 4 <= len(seen) <= 6


def test_dt_chunk_small_duration():
    nodes, edges, bcs, case = _canonical_single_node()
    case = CaseConfig(case.thermo, case.h_conv, case.T_wall, 0.01, 10)
    seen = []

    def cb(payload):
        seen.append(payload["progress"])

    solve_case_stream(nodes, edges, bcs, case, callback=cb, dt_chunk_s=2.0)
    assert len(seen) >= 1
