#!/usr/bin/env python3
"""Сборка venting.exe. Запуск: python build_windows.py"""

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
        print("UPX не найден — exe будет крупнее. https://github.com/upx/upx/releases")

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
        print(f"\nГОТОВО: dist/venting.exe  ({exe.stat().st_size / 1e6:.0f} МБ)")
        print("Файл работает без установки Python — можно отправлять.")
    else:
        print("ОШИБКА: dist/venting.exe не создан")
        sys.exit(1)


if __name__ == "__main__":
    main()
