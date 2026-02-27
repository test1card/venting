# Changelog

## v0.8.5 - version alignment
- Bumped active package/model naming from v8.4 to v8.5 (0.8.5) to avoid confusion.
- Updated CLI artifact prefix to `v85_` for new runs.
- Updated documentation references to the active v8.5 naming.

## v0.8.4 - repo hardening
- Hardened packaging for standard PEP 517/518 src-layout installation.
- Set v8.4 as the only active runtime package; legacy script remains archived only.
- Added CI user-like install flow (`pip install -e ".[dev]"`, `pip check`, lint, coverage tests).
- Reworked tests into deterministic gate/physics validation suite (`tests/test_gates.py`).
- Added structured run outputs (`run.json`, `summary.csv`) through `venting.io`.
- Updated docs for verification equations, sign conventions, and acceptance criteria.
