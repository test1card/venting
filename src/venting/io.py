from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def make_results_dir(case_name: str) -> Path:
    out = Path("results") / f"{utc_timestamp()}_{case_name}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def current_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def package_version() -> str:
    try:
        return version("venting")
    except PackageNotFoundError:
        return "10.0.0"


def write_run_json(outdir: Path, params: dict, solver_settings: dict) -> None:
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(),
        "python_version": sys.version,
        "package_version": package_version(),
        "platform": platform.platform(),
        "parameters": params,
        "solver_settings": solver_settings,
    }
    (outdir / "run.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_summary_csv(outdir: Path, res) -> None:
    rows = []
    for edge, peak in res.peak_diag.items():
        rows.append(
            {
                "edge": edge,
                "max_abs_dP_Pa": res.max_dP.get(edge, float("nan")),
                "t_peak_s": peak.get("t_peak", float("nan")),
                "r_peak": peak.get("r", float("nan")),
                "regime": peak.get("regime", ""),
                "peak_type": peak.get("peak_type", ""),
                "tau_exit_s": res.tau_exit,
            }
        )
    with (outdir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "edge",
                "max_abs_dP_Pa",
                "t_peak_s",
                "r_peak",
                "regime",
                "peak_type",
                "tau_exit_s",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def dump_meta_json(outdir: Path, filename: str, meta: dict) -> None:
    clean_meta = dict(meta)
    if "case" in clean_meta and not isinstance(clean_meta["case"], dict):
        clean_meta["case"] = asdict(clean_meta["case"])
    (outdir / filename).write_text(
        json.dumps(clean_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_validity_json(outdir: Path, stem: str, validity_flags: dict) -> Path:
    path = outdir / f"{stem}_validity.json"
    path.write_text(
        json.dumps(validity_flags, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def print_validity_summary(validity_flags: dict) -> None:
    print("Validity flags:")
    for key, payload in validity_flags.items():
        status = payload.get("status", "n/a")
        msg = payload.get("message", "")
        print(f"  - {key}: {status} {msg}".rstrip())
