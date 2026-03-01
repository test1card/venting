from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def load_run(run_dir: Path) -> dict:
    """Load npz + metadata artifacts from a results directory."""
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    npz_files = sorted(run_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz file found in {run_dir}")
    npz = np.load(npz_files[0])

    meta_files = sorted(run_dir.glob("*_meta.json"))
    meta = json.loads(meta_files[0].read_text(encoding="utf-8")) if meta_files else {}

    summary_path = run_dir / "summary.csv"
    summary = []
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as fh:
            summary = list(csv.DictReader(fh))

    return {
        "dir": run_dir,
        "name": npz_files[0].stem,
        "t": npz["t"],
        "P": npz["P"],
        "T": npz["T"],
        "tau_exit": float(npz["tau_exit"]),
        "meta": meta,
        "summary": summary,
    }


def compare_runs(run_a: dict, run_b: dict) -> dict:
    """Compare run metrics and return structured differences."""
    Pa = run_a["P"][:, -1]
    Pb = run_b["P"][:, -1]
    Ta = run_a["T"][:, -1]
    Tb = run_b["T"][:, -1]

    if Pa.shape == Pb.shape:
        max_final_P_diff = float(np.max(np.abs(Pb - Pa)))
        max_final_T_diff = float(np.max(np.abs(Tb - Ta)))
    else:
        max_final_P_diff = float("nan")
        max_final_T_diff = float("nan")

    delta_summary = {}
    by_edge_a = {row["edge"]: row for row in run_a.get("summary", [])}
    by_edge_b = {row["edge"]: row for row in run_b.get("summary", [])}
    for edge in sorted(set(by_edge_a) | set(by_edge_b)):
        ra = by_edge_a.get(edge)
        rb = by_edge_b.get(edge)
        if not ra or not rb:
            continue
        delta_summary[edge] = {
            "delta_max_abs_dP_Pa": float(rb["max_abs_dP_Pa"])
            - float(ra["max_abs_dP_Pa"]),
            "delta_t_peak_s": float(rb["t_peak_s"]) - float(ra["t_peak_s"]),
            "regime_a": ra.get("regime", ""),
            "regime_b": rb.get("regime", ""),
        }

    tau_a = float(run_a.get("tau_exit", np.nan))
    tau_b = float(run_b.get("tau_exit", np.nan))
    rel_tau = (tau_b - tau_a) / tau_a if abs(tau_a) > 0 else float("nan")

    return {
        "run_a": str(run_a["dir"]),
        "run_b": str(run_b["dir"]),
        "edge_deltas": delta_summary,
        "max_final_P_diff": max_final_P_diff,
        "max_final_T_diff": max_final_T_diff,
        "rel_tau_exit_delta": float(rel_tau),
    }


def format_comparison_table(comparison: dict) -> str:
    lines = [
        f"A: {comparison['run_a']}",
        f"B: {comparison['run_b']}",
        "edge | Δmax|ΔP| [Pa] | Δt_peak [s] | regime A -> B",
    ]
    for edge, d in comparison["edge_deltas"].items():
        lines.append(
            f"{edge} | {d['delta_max_abs_dP_Pa']:.6g} | {d['delta_t_peak_s']:.6g} | {d['regime_a']} -> {d['regime_b']}"
        )
    lines.append(f"max final P diff: {comparison['max_final_P_diff']:.6g}")
    lines.append(f"max final T diff: {comparison['max_final_T_diff']:.6g}")
    lines.append(f"relative tau_exit delta: {comparison['rel_tau_exit_delta']:.6g}")
    return "\n".join(lines)


def write_comparison_csv(comparison: dict, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "edge",
                "delta_max_abs_dP_Pa",
                "delta_t_peak_s",
                "regime_a",
                "regime_b",
            ],
        )
        writer.writeheader()
        for edge, d in comparison["edge_deltas"].items():
            writer.writerow({"edge": edge, **d})
