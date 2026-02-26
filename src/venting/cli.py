from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .cases import CaseConfig, NetworkConfig
from .constants import P0, P_STOP, T0
from .diagnostics import summarize_result
from .gates import gate_single, gate_two
from .geometry import mm2_to_m2, mm3_to_m3
from .graph import build_branching_network
from .io import dump_meta_json, make_results_dir, write_run_json, write_summary_csv
from .plotting import plot_basic
from .profiles import (
    make_profile_exponential,
    make_profile_from_table,
    make_profile_linear,
    make_profile_step,
)
from .solver import solve_case


def _profile(args):
    if args.profile == "linear":
        return make_profile_linear(P0, args.rate_mmhg)
    if args.profile == "step":
        return make_profile_step(P0, args.step_time)
    if args.profile == "barometric":
        return make_profile_exponential(P0, args.rate_mmhg, p_floor=10.0)
    return make_profile_from_table("envelope_table", Path(args.profile_file))


def _run_one(args, d_int: float, d_exit: float, h: float | None = None):
    prof = _profile(args)
    A_cell_mm2, A_vest_mm2, h_mm = 4708.2806538, 5199.9595503, 27.9430913
    V_cell = mm3_to_m3(A_cell_mm2 * h_mm)
    V_vest = mm3_to_m3(A_vest_mm2 * h_mm)
    A_wall_cell = mm2_to_m2(2 * A_cell_mm2 + 313.0 * h_mm)
    A_wall_vest = mm2_to_m2(2 * A_vest_mm2 + 350.0 * h_mm)

    net_cfg = NetworkConfig(
        N_chain=10,
        N_par=2,
        V_cell=V_cell,
        V_vest=V_vest,
        A_wall_cell=A_wall_cell,
        A_wall_vest=A_wall_vest,
        d_int_mm=d_int,
        n_int_per_interface=args.n_int,
        d_exit_mm=d_exit,
        n_exit=args.n_exit,
        Cd_int=args.cd_int,
        Cd_exit=args.cd_exit,
    )
    case = CaseConfig(
        thermo=args.thermo,
        h_conv=args.h if h is None else h,
        T_wall=T0,
        duration=args.duration,
        n_pts=args.npts,
        p_stop=P_STOP,
        p_rms_tol=1.0,
    )
    nodes, edges, bcs = build_branching_network(net_cfg, prof)
    sol = solve_case(nodes, edges, bcs, case)
    res = summarize_result(nodes, edges, bcs, case, sol)

    out = make_results_dir(args.cmd)
    name = (
        f"v84_{prof.name}_{case.thermo}_dint{d_int:g}_"
        f"dexit{d_exit:g}_h{case.h_conv:g}"
    )
    np.savez_compressed(
        out / f"{name}.npz",
        t=res.t,
        m=res.m,
        T=res.T,
        P=res.P,
        P_ext=res.P_ext,
        tau_exit=res.tau_exit,
    )
    dump_meta_json(out, f"{name}_meta.json", res.meta)
    write_summary_csv(out, res)
    write_run_json(
        out,
        params={
            "command": args.cmd,
            "profile": args.profile,
            "d_int": d_int,
            "d_exit": d_exit,
            "thermo": case.thermo,
            "h": case.h_conv,
            "duration": case.duration,
            "npts": case.n_pts,
            "cd_int": args.cd_int,
            "cd_exit": args.cd_exit,
        },
        solver_settings={"method": "Radau", "rtol": "1e-7|1e-6", "atol": "1e-10|1e-8"},
    )
    if args.do_plots:
        plot_basic(out, name, res)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="venting")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gate")
    g.add_argument("--single", action="store_true")
    g.add_argument("--two", action="store_true")

    for name in ["sweep", "thermal", "sweep2d"]:
        s = sub.add_parser(name)
        s.add_argument(
            "--profile",
            default="linear",
            choices=["linear", "step", "barometric", "table"],
        )
        s.add_argument("--profile-file", default="")
        s.add_argument("--rate-mmhg", type=float, default=20.0)
        s.add_argument("--step-time", type=float, default=0.01)
        s.add_argument(
            "--thermo", default="isothermal", choices=["isothermal", "intermediate"]
        )
        s.add_argument("--h", type=float, default=0.0)
        s.add_argument("--duration", type=float, default=150.0)
        s.add_argument("--npts", type=int, default=800)
        s.add_argument("--n-int", type=int, default=1)
        s.add_argument("--n-exit", type=int, default=1)
        s.add_argument("--cd-int", type=float, default=0.62)
        s.add_argument("--cd-exit", type=float, default=0.62)
        s.add_argument("--do-plots", action="store_true")

    sub.choices["sweep"].add_argument("--d-int", type=float, default=2.0)
    sub.choices["sweep"].add_argument("--d-exit", type=float, default=2.0)

    sub.choices["thermal"].add_argument("--d", type=float, default=2.0)
    sub.choices["thermal"].add_argument("--h-list", default="0,1,5,15")

    sub.choices["sweep2d"].add_argument("--d-int-list", default="1.0,2.0")
    sub.choices["sweep2d"].add_argument("--d-exit-list", default="1.0,2.0")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "gate":
        run_single = args.single or (not args.single and not args.two)
        run_two = args.two or (not args.single and not args.two)
        if run_single:
            print(gate_single())
        if run_two:
            print(gate_two())
        return
    if args.cmd == "sweep":
        _run_one(args, args.d_int, args.d_exit)
        return
    if args.cmd == "thermal":
        hs = [float(x) for x in args.h_list.split(",") if x]
        for h in hs:
            _run_one(args, args.d, args.d, h=h)
        return

    d_ints = [float(x) for x in args.d_int_list.split(",") if x]
    d_exits = [float(x) for x in args.d_exit_list.split(",") if x]
    for d_int in d_ints:
        for d_exit in d_exits:
            _run_one(args, d_int, d_exit)


if __name__ == "__main__":
    main()
