#!/usr/bin/env python3
"""Build venting.exe (GUI) and venting_cli.exe (CLI).

Usage:
    python build_windows.py             # build both
    python build_windows.py --gui-only  # GUI exe only
    python build_windows.py --cli-only  # CLI exe only
    python build_windows.py --debug     # diagnostic build (console=True, no UPX)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def check(pkg):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


def build_spec(spec, *, upx, extra_flags=None):
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        spec,
        "--clean",
        "--noconfirm",
    ]
    if not upx:
        cmd.append("--noupx")
    if extra_flags:
        cmd.extend(extra_flags)

    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Build venting Windows executables")
    parser.add_argument("--gui-only", action="store_true", help="Build GUI exe only")
    parser.add_argument("--cli-only", action="store_true", help="Build CLI exe only")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Diagnostic build: no UPX, debug logging",
    )
    args = parser.parse_args()

    check("PyInstaller")
    upx = bool(shutil.which("upx")) and not args.debug
    if not upx and not args.debug:
        print(
            "UPX not found -- exe will be larger. "
            "https://github.com/upx/upx/releases"
        )

    for d in ["build", "dist"]:
        if Path(d).exists():
            shutil.rmtree(d)

    build_gui = not args.cli_only
    build_cli = not args.gui_only

    extra = []
    if args.debug:
        extra.append("--log-level=DEBUG")

    ok = True

    if build_gui:
        print("\n=== Building venting.exe (GUI, console=False) ===")
        if not build_spec("venting.spec", upx=upx, extra_flags=extra):
            print("ERROR: venting.exe build failed")
            ok = False

    if build_cli:
        print("\n=== Building venting_cli.exe (CLI, console=True) ===")
        if not build_spec("venting_cli.spec", upx=upx, extra_flags=extra):
            print("ERROR: venting_cli.exe build failed")
            ok = False

    if not ok:
        sys.exit(1)

    print("\n=== Build results ===")
    for name in ["venting.exe", "venting_cli.exe"]:
        exe = Path("dist") / name
        if exe.exists():
            print(f"  {name}: {exe.stat().st_size / 1e6:.0f} MB")
    print("\nReady to distribute. No Python installation required.")


if __name__ == "__main__":
    main()
