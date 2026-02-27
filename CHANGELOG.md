# Changelog

## v10.0.0
- Fixed Radau setup by removing invalid identity-only Jacobian sparsity hint.
- Added scalar-or-list network parameter support and a new `two_chain_shared_vest` topology.
- Updated GUI worker/progress handling and stop-check integration for true streaming callbacks.
- Added regression tests for baseline gate behavior and topology/scalar-list compatibility.
- Added desktop GUI (`python -m venting gui`) with optional dependencies (`.[gui]`) for configuring and running venting cases.
- Added live plotting tabs for pressure, temperature, mass, peak diagnostics, and validity flags while solving in a background thread.
- Added JSON save/load schema for GUI cases with explicit units and validation.
- Added shared presets and run pipeline modules to keep CLI and GUI execution paths consistent.
- Added streaming solver wrapper for progressive updates with regression tests against batch solve.

## v9.0.0
- Added physically consistent short-tube thick-wall edge model with Darcy friction, minor losses, and iterative `Cd_eff` composition (lossy-nozzle, not Fanno).
- Added split internal/exit short-tube loss parameters (`K_in_*`, `K_out_*`, `eps_*`) with backward-compatible alias CLI flags.
- Added variable thermodynamics mode (`--thermo variable`) with temperature-dependent `cp/cv/gamma/h/u` and gamma-aware compressible discharge.
- Added lumped wall thermal model (`--wall-model lumped`) with wall heat capacity, optional outside convection/radiation/source heat flux.
- Added dynamic external pump model (`--external-model dynamic_pump`) with finite external volume and ultimate-pressure pump sink law.
- Expanded validity diagnostics with `short_tube_flow` metrics (`Mach_max`, `Re_max`, `K_tot_max`, `Cd_eff_min`, `L_over_D`, `frac_fric`), and always save validity as `*_validity.json` plus `meta.json`.
- Updated packaging metadata to 9.0.0 and refreshed docs for v9 assumptions.

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
