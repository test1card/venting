from __future__ import annotations

import argparse
from pathlib import Path

from .cases import CaseConfig, NetworkConfig
from .constants import P0, P_STOP, T0
from .gates import gate_single, gate_two
from .plotting import plot_basic
from .presets import get_default_panel_preset_v9
from .profiles import (
    make_profile_exponential,
    make_profile_from_table,
    make_profile_linear,
    make_profile_step,
)
from .run import export_case_artifacts, make_case_output_dir, run_case


def _profile(args):
    if args.external_model == "dynamic_pump":
        return make_profile_step(P0, 1e9)
    if args.profile == "linear":
        return make_profile_linear(P0, args.rate_mmhg)
    if args.profile == "step":
        return make_profile_step(P0, args.step_time)
    if args.profile == "barometric":
        return make_profile_exponential(P0, args.rate_mmhg, p_floor=10.0)
    return make_profile_from_table(
        "envelope_table",
        Path(args.profile_file),
        pressure_unit=args.profile_pressure_unit,
    )


def _run_one(args, d_int: float, d_exit: float, h: float | None = None):
    prof = _profile(args)
    preset = get_default_panel_preset_v9()

    net_cfg = NetworkConfig(
        N_chain=10,
        N_par=2,
        V_cell=preset.V_cell,
        V_vest=preset.V_vest,
        A_wall_cell=preset.A_wall_cell,
        A_wall_vest=preset.A_wall_vest,
        d_int_mm=d_int,
        n_int_per_interface=args.n_int,
        d_exit_mm=d_exit,
        n_exit=args.n_exit,
        Cd_int=args.cd_int,
        Cd_exit=args.cd_exit,
        int_model=args.int_model,
        exit_model=args.exit_model,
        L_int_mm=args.L_int_mm,
        L_exit_mm=args.L_exit_mm,
        K_in=args.K_in,
        K_out=args.K_out,
        eps_um=args.eps_um,
        K_in_int=args.K_in_int,
        K_out_int=args.K_out_int,
        eps_int_um=args.eps_int_um,
        K_in_exit=args.K_in_exit,
        K_out_exit=args.K_out_exit,
        eps_exit_um=args.eps_exit_um,
    )
    case = CaseConfig(
        thermo=args.thermo,
        h_conv=args.h if h is None else h,
        T_wall=T0,
        duration=args.duration,
        n_pts=args.npts,
        p_stop=P_STOP,
        p_rms_tol=1.0,
        wall_model=args.wall_model,
        wall_C_per_area=args.wall_C_per_area,
        wall_h_out=args.wall_h_out,
        wall_T_inf=args.wall_T_inf,
        wall_emissivity=args.wall_emissivity,
        wall_T_sur=args.wall_T_sur,
        wall_q_flux=args.wall_q_flux,
        external_model=args.external_model,
        V_ext=args.V_ext,
        T_ext=args.T_ext,
        pump_speed_m3s=args.pump_speed_m3s,
        P_ult_Pa=args.P_ult_Pa,
    )

    res = run_case(net_cfg, prof, case)

    out = make_case_output_dir(args.cmd)
    name = (
        f"v1000_{args.external_model}_{prof.name}_{case.thermo}_"
        f"dint{d_int:g}_dexit{d_exit:g}_h{case.h_conv:g}"
    )
    run_params = {
        "command": args.cmd,
        "profile": args.profile,
        "external_model": args.external_model,
        "d_int": d_int,
        "d_exit": d_exit,
        "thermo": case.thermo,
        "wall_model": case.wall_model,
        "h": case.h_conv,
        "duration": case.duration,
        "npts": case.n_pts,
        "cd_int": args.cd_int,
        "cd_exit": args.cd_exit,
        "int_model": args.int_model,
        "exit_model": args.exit_model,
        "L_int_mm": args.L_int_mm,
        "L_exit_mm": args.L_exit_mm,
        "K_in_int": net_cfg.K_in_int,
        "K_out_int": net_cfg.K_out_int,
        "eps_int_um": net_cfg.eps_int_um,
        "K_in_exit": net_cfg.K_in_exit,
        "K_out_exit": net_cfg.K_out_exit,
        "eps_exit_um": net_cfg.eps_exit_um,
        "K_in_alias": args.K_in,
        "K_out_alias": args.K_out,
        "eps_um_alias": args.eps_um,
        "profile_pressure_unit": args.profile_pressure_unit,
        "pump_speed_m3s": case.pump_speed_m3s,
        "V_ext": case.V_ext,
        "P_ult_Pa": case.P_ult_Pa,
        "panel_preset": preset.to_dict(),
    }
    export_case_artifacts(out, name, res, run_params)
    if args.do_plots:
        plot_basic(out, name, res)


def _add_common_args(s: argparse.ArgumentParser) -> None:
    s.add_argument(
        "--profile", default="linear", choices=["linear", "step", "barometric", "table"]
    )
    s.add_argument(
        "--external-model", default="profile", choices=["profile", "dynamic_pump"]
    )
    s.add_argument("--profile-file", default="")
    s.add_argument("--profile-pressure-unit", default="Pa", choices=["Pa", "mmHg"])
    s.add_argument("--rate-mmhg", type=float, default=20.0)
    s.add_argument("--step-time", type=float, default=0.01)
    s.add_argument(
        "--thermo",
        default="isothermal",
        choices=["isothermal", "intermediate", "variable"],
    )
    s.add_argument("--h", type=float, default=0.0)
    s.add_argument("--duration", type=float, default=150.0)
    s.add_argument("--npts", type=int, default=800)
    s.add_argument("--n-int", type=int, default=1)
    s.add_argument("--n-exit", type=int, default=1)
    s.add_argument("--cd-int", type=float, default=0.62)
    s.add_argument("--cd-exit", type=float, default=0.62)
    s.add_argument("--int-model", choices=["orifice", "short_tube"], default="orifice")
    s.add_argument("--exit-model", choices=["orifice", "short_tube"], default="orifice")
    s.add_argument("--L-int-mm", type=float, default=0.0)
    s.add_argument("--L-exit-mm", type=float, default=0.0)
    s.add_argument("--K-in-int", type=float, default=None)
    s.add_argument("--K-out-int", type=float, default=None)
    s.add_argument("--eps-int-um", type=float, default=None)
    s.add_argument("--K-in-exit", type=float, default=None)
    s.add_argument("--K-out-exit", type=float, default=None)
    s.add_argument("--eps-exit-um", type=float, default=None)
    s.add_argument(
        "--K-in",
        type=float,
        default=0.5,
        help="Deprecated alias for both internal/exit K_in",
    )
    s.add_argument(
        "--K-out",
        type=float,
        default=1.0,
        help="Deprecated alias for both internal/exit K_out",
    )
    s.add_argument(
        "--eps-um",
        type=float,
        default=0.0,
        help="Deprecated alias for both internal/exit roughness",
    )
    s.add_argument("--wall-model", choices=["fixed", "lumped"], default="fixed")
    s.add_argument("--wall-C-per-area", type=float, default=1e9)
    s.add_argument("--wall-h-out", type=float, default=0.0)
    s.add_argument("--wall-T-inf", type=float, default=T0)
    s.add_argument("--wall-emissivity", type=float, default=0.0)
    s.add_argument("--wall-T-sur", type=float, default=T0)
    s.add_argument("--wall-q-flux", type=float, default=0.0)
    s.add_argument("--V-ext", type=float, default=0.1)
    s.add_argument("--T-ext", type=float, default=T0)
    s.add_argument("--pump-speed-m3s", type=float, default=0.0)
    s.add_argument("--P-ult-Pa", type=float, default=0.0)
    s.add_argument("--do-plots", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="venting")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gate")
    g.add_argument("--single", action="store_true")
    g.add_argument("--two", action="store_true")

    sub.add_parser("gui")

    for name in ["sweep", "thermal", "sweep2d"]:
        s = sub.add_parser(name)
        _add_common_args(s)

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
    if args.cmd == "gui":
        from .gui.main import main as gui_main

        gui_main()
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
