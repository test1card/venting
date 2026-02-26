# Changelog

## v0.8.4 - repo hardening
- Hardened packaging for standard PEP 517/518 src-layout installation.
- Set v8.4 as the only active runtime package; legacy script remains archived only.
- Added CI user-like install flow (`pip install -e ".[dev]"`, `pip check`, lint, coverage tests).
- Reworked tests into deterministic gate/physics validation suite (`tests/test_gates.py`).
- Added structured run outputs (`run.json`, `summary.csv`) through `venting.io`.
- Updated docs for verification equations, sign conventions, and acceptance criteria.
