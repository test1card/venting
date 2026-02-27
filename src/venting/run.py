from __future__ import annotations

from pathlib import Path

import numpy as np

from .cases import CaseConfig, NetworkConfig
from .diagnostics import summarize_result
from .graph import build_branching_network
from .io import (
    dump_meta_json,
    make_results_dir,
    print_validity_summary,
    write_run_json,
    write_summary_csv,
    write_validity_json,
)
from .profiles import Profile
from .solver import solve_case


def run_case(net_cfg: NetworkConfig, profile: Profile, case_cfg: CaseConfig):
    nodes, edges, bcs = build_branching_network(net_cfg, profile)
    sol = solve_case(nodes, edges, bcs, case_cfg)
    return summarize_result(nodes, edges, bcs, case_cfg, sol)


def export_case_artifacts(
    outdir: Path,
    stem: str,
    res,
    run_params: dict,
    solver_settings: dict | None = None,
) -> None:
    solver_payload = solver_settings or {
        "method": "Radau",
        "rtol": "1e-7|1e-6",
        "atol": "1e-10|1e-8",
    }
    np.savez_compressed(
        outdir / f"{stem}.npz",
        t=res.t,
        m=res.m,
        T=res.T,
        P=res.P,
        P_ext=res.P_ext,
        tau_exit=res.tau_exit,
    )
    dump_meta_json(outdir, f"{stem}_meta.json", res.meta)
    write_validity_json(outdir, stem, res.meta.get("validity_flags", {}))
    print_validity_summary(res.meta.get("validity_flags", {}))
    write_summary_csv(outdir, res)
    write_run_json(outdir, params=run_params, solver_settings=solver_payload)


def make_case_output_dir(case_name: str):
    return make_results_dir(case_name)
