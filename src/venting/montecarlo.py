from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .cases import CaseConfig, NetworkConfig
from .diagnostics import summarize_result
from .graph import build_branching_network
from .profiles import Profile
from .solver import solve_case


def run_mc(
    net_cfg_base: NetworkConfig,
    case_cfg: CaseConfig,
    profile: Profile,
    cd_int_range: tuple[float, float],
    cd_exit_range: tuple[float, float],
    n_samples: int,
    seed: int | None = None,
) -> dict:
    """Monte Carlo sweep over Cd uncertainty."""
    rng = np.random.default_rng(seed)
    samples = []
    edge_values: dict[str, list[float]] = {}

    for i in range(n_samples):
        cd_int = float(rng.uniform(*cd_int_range))
        cd_exit = float(rng.uniform(*cd_exit_range))
        cfg = replace(net_cfg_base, Cd_int=cd_int, Cd_exit=cd_exit)
        nodes, edges, bcs = build_branching_network(cfg, profile)
        sol = solve_case(nodes, edges, bcs, case_cfg)
        res = summarize_result(nodes, edges, bcs, case_cfg, sol)
        row = {"sample": i, "Cd_int": cd_int, "Cd_exit": cd_exit}
        for edge, value in res.max_dP.items():
            row[f"max_dP_{edge}"] = float(value)
            edge_values.setdefault(edge, []).append(float(value))
        samples.append(row)

    summary = {}
    for edge, values in edge_values.items():
        arr = np.asarray(values, dtype=float)
        summary[edge] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p5": float(np.percentile(arr, 5)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
        }

    return {"samples": samples, "summary": summary}


def write_mc_outputs(result: dict, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    sample_rows = result["samples"]
    if sample_rows:
        cols = list(sample_rows[0].keys())
        import csv

        with (outdir / "mc_results.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            writer.writerows(sample_rows)
    (outdir / "mc_summary.json").write_text(
        json.dumps(result["summary"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
