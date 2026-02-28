#!/usr/bin/env python3
"""Build venting.exe. Run: python build_windows.py"""

import shutil
import subprocess
import sys
from pathlib import Path


def check(pkg):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


def main():
    check("PyInstaller")
    upx = bool(shutil.which("upx"))
    if not upx:
        print("UPX not found -- exe will be larger. https://github.com/upx/upx/releases")

    for d in ["build", "dist"]:
        if Path(d).exists():
            shutil.rmtree(d)

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "venting.spec",
        "--clean",
        "--noconfirm",
    ]
    if not upx:
        cmd.append("--noupx")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(1)

    exe = Path("dist/venting.exe")
    if exe.exists():
        print(f"\nDONE: dist/venting.exe  ({exe.stat().st_size / 1e6:.0f} MB)")
        print("The file runs without installing Python -- ready to distribute.")
    else:
        print("ERROR: dist/venting.exe was not created")
        sys.exit(1)


if __name__ == "__main__":
    main()
