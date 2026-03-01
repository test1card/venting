"""Bootstrap entry point for venting.

Provides:
- Earliest-possible crash logging to a file so that failures in windowed
  (console=False) PyInstaller builds are never silent.
- Frozen-exe default: if running as a PyInstaller bundle with no CLI args,
  launch the GUI directly instead of requiring a subcommand.
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Bootstrap log directory
# ---------------------------------------------------------------------------
_FROZEN: bool = getattr(sys, "frozen", False)


def _log_dir() -> Path:
    """Return a writable directory for boot/crash logs."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or "."
    else:
        base = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    d = Path(base) / "venting" / "logs"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        d = Path(os.environ.get("TEMP", "/tmp")) / "venting_logs"
        d.mkdir(parents=True, exist_ok=True)
    return d


def _setup_logging() -> logging.Logger:
    log = logging.getLogger("venting.boot")
    log.setLevel(logging.DEBUG)
    try:
        fh = logging.FileHandler(_log_dir() / "boot.log", encoding="utf-8", delay=False)
        fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
        log.addHandler(fh)
    except OSError:
        pass  # Cannot log — will still try to show MessageBox on fatal
    return log


def _show_fatal_messagebox(crash_path: str, short_msg: str) -> None:
    """Show a minimal Windows MessageBox on fatal error (no new deps)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(
            0,
            f"Venting crashed on startup.\n\n{short_msg}\n\n"
            f"Full traceback saved to:\n{crash_path}",
            "Venting — Fatal Error",
            0x10,  # MB_ICONERROR
        )
    except Exception:
        pass  # Best-effort


# ---------------------------------------------------------------------------
# 2. Main entry logic
# ---------------------------------------------------------------------------
def _entry() -> None:
    log = _setup_logging()
    log.info(
        "boot: argv=%s  executable=%s  frozen=%s  cwd=%s",
        sys.argv,
        sys.executable,
        _FROZEN,
        os.getcwd(),
    )

    try:
        from venting import __version__

        log.info("venting version: %s  python: %s", __version__, sys.version)
    except Exception:
        log.warning("could not determine venting version")

    # ------------------------------------------------------------------
    # Frozen exe with no args → launch GUI directly (H1 fix)
    # ------------------------------------------------------------------
    if _FROZEN and len(sys.argv) <= 1:
        log.info("frozen build, no args → launching GUI")
        try:
            from venting.gui.main import main as gui_main

            raise SystemExit(gui_main())
        except SystemExit:
            raise
        except Exception:
            tb = traceback.format_exc()
            crash_file = _log_dir() / "crash.log"
            crash_file.write_text(tb, encoding="utf-8")
            log.critical("GUI launch failed:\n%s", tb)
            _show_fatal_messagebox(str(crash_file), tb.splitlines()[-1])
            raise SystemExit(1) from None

    # ------------------------------------------------------------------
    # Normal CLI path
    # ------------------------------------------------------------------
    try:
        from venting.cli import main

        main()
    except SystemExit as exc:
        log.info("CLI exited with code %s", exc.code)
        raise
    except Exception:
        tb = traceback.format_exc()
        crash_file = _log_dir() / "crash.log"
        crash_file.write_text(tb, encoding="utf-8")
        log.critical("unhandled exception:\n%s", tb)
        if _FROZEN:
            _show_fatal_messagebox(str(crash_file), tb.splitlines()[-1])
        raise SystemExit(1) from None


if __name__ == "__main__":
    _entry()
