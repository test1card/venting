# CLAUDE.md — Venting Codebase Guide for AI Assistants

This file provides context for AI assistants (Claude, Copilot, etc.) working in the
`venting` repository. Read it before modifying any code.

---

## Project Overview

**Venting v10.0.0** is a 0D/network depressurization solver for aerospace and
industrial pressure-relief analysis. It models rigid gas volumes (nodes) connected
by orifices or channels (edges) and integrates the resulting stiff ODE system using
SciPy's Radau solver.

- **Not CFD.** No spatial fields, no shocks, no acoustics — purely lumped-parameter.
- **Primary output:** maximum `|ΔP|` across each edge and whether flow is choked.
- **Python ≥ 3.10**, MIT license.

---

## Repository Layout

```
venting/
├── src/venting/          # Main package (src-layout, installed as `venting`)
│   ├── __init__.py       # Version: 10.0.0
│   ├── __main__.py       # `python -m venting` entry point
│   ├── cli.py            # Argparse CLI (subcommands: gate, sweep, sweep2d, …)
│   ├── constants.py      # Physical constants (γ, R, π_c, σ_SB, k_B, safety thresholds)
│   ├── cases.py          # Frozen dataclasses: CaseConfig, NetworkConfig; mutable SolveResult
│   ├── geometry.py       # Unit converters (mm↔m, circle area)
│   ├── profiles.py       # External pressure profiles (linear/step/barometric/table)
│   ├── graph.py          # Network topology builder: GasNode, OrificeEdge, ShortTubeEdge, SlotChannelEdge
│   ├── flow.py           # Mass-flow functions: orifice, short_tube, slot, Fanno; viscosity (Sutherland)
│   ├── solver.py         # ODE RHS, solve_ivp (Radau), events, streaming
│   ├── diagnostics.py    # Peak detection, τ_exit, pressure regime classification
│   ├── validity.py       # Physical validity flags (acoustic timescale, Re, Mach, thermo)
│   ├── gates.py          # 5 analytical gate tests (single-node, two-node, conservation)
│   ├── io.py             # Artifact export: run.json, summary.csv, .npz, meta.json
│   ├── run.py            # High-level case execution pipeline
│   ├── compare.py        # Run-comparison and loading utilities
│   ├── plotting.py       # Matplotlib ΔP-vs-time visualization (Agg backend)
│   ├── montecarlo.py     # Latin-hypercube parametric sampling
│   ├── presets.py        # Default panel geometry (volumes, wall areas)
│   ├── thermo.py         # NASA-7 polynomial fits: cp(T), cv(T), γ(T), h(T), u(T), speed_of_sound(T)
│   ├── state_layout.py   # ODE state-vector slicing (m, T, T_wall indices)
│   └── gui/              # Optional PySide6/pyqtgraph desktop interface
│       ├── __init__.py
│       ├── main.py
│       ├── app.py
│       ├── config.py
│       └── state_layout.py
├── tests/                # pytest test suite (~1000 lines, 15 files)
├── docs/                 # Markdown physics docs and verification criteria
├── archive/              # Legacy monolithic script (reference only, not imported)
├── .github/workflows/    # CI: Python 3.10–3.12 matrix, ruff, black, pytest
├── pyproject.toml        # Project metadata, build config, tool settings
├── requirements.txt      # Runtime deps (numpy, scipy, matplotlib)
├── requirements-dev.txt  # Dev deps (-e .[dev])
├── .pre-commit-config.yaml
├── README.md             # Bilingual (Russian + English) physics and CLI reference
└── CONTRIBUTING.md
```

---

## Development Setup

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (ruff check + ruff format + black on every commit)
pre-commit install
```

### Optional GUI

```bash
pip install -e ".[gui]"   # adds PySide6, pyqtgraph
python -m venting gui
```

---

## Running Tests

```bash
# Fast (CI-equivalent)
pytest -q

# With coverage
pytest -q --cov=venting --cov-report=term-missing

# Single test file
pytest tests/test_gates.py -v
```

All tests are deterministic. Gate tests validate against hardcoded analytic solutions
with tight tolerances (`< 0.5%` for physics, `< 0.1%` for mass conservation).

---

## Linting and Formatting

```bash
ruff check .          # Lint (F, E, I, B, UP rules; E501 ignored)
ruff check . --fix    # Auto-fix safe issues
black .               # Format (line-length 88)
black --check .       # Format check only
```

Pre-commit hooks run ruff (check + format) and black automatically on `git commit`.

---

## CLI Commands

```bash
python -m venting gate            # Run all 5 analytical gate tests
python -m venting gate --single   # Single-node validation only
python -m venting gate --two      # Two-node validation only

python -m venting sweep           # 1D parameter sweep (d_int, d_exit, Cd)
python -m venting sweep2d         # 2D grid sweep
python -m venting thermal         # Multi-h thermal sensitivity
python -m venting mc              # Latin-hypercube parametric sampling

python -m venting compare dir1 dir2   # Compare two run result directories

python -m venting gui             # Launch PySide6 desktop interface
```

---

## Architecture and Data Flow

```
NetworkConfig / CaseConfig
        │
        ▼
    graph.py  ──► builds list of GasNode + edge objects
        │
        ▼
    solver.py ──► constructs ODE RHS using flow.py functions
        │          integrates with scipy.integrate.solve_ivp (Radau, dense Jacobian)
        ▼
    SolveResult (time array + state array)
        │
        ├──► diagnostics.py  ──► peak ΔP, τ_exit, regime (CHOKED/subsonic)
        ├──► validity.py     ──► physical flag checks
        └──► io.py           ──► run.json, summary.csv, .npz, meta.json
```

### ODE State Vector Layout (`state_layout.py`)

For a network with `N` nodes:

| Slice | Variables |
|-------|-----------|
| `[0:N]` | mass `m[i]` (kg) for each node |
| `[N:2N]` | temperature `T[i]` (K) — only if `mode != "isothermal"` |
| `[2N:3N]` | wall temperature `T_wall[i]` (K) — only if `lumped_wall=True` |

---

## Key Dataclasses (`cases.py`)

- **`CaseConfig`** (`frozen=True`) — thermodynamic mode (`"isothermal"` / `"intermediate"` / `"variable"`),
  wall model settings, external model selection.
- **`NetworkConfig`** (`frozen=True`) — node volumes/temperatures, edge geometry (diameters, Cd, lengths),
  network topology (`N_chain`, `N_par`, `topology`).
- **`SolveResult`** (mutable `@dataclass`) — `t`, `m`, `T`, `P`, `P_ext`, `peak_diag`,
  `max_dP`, `tau_exit`, `meta`.

---

## Flow Models (`flow.py`)

| Model | Function | Notes |
|-------|----------|-------|
| Sharp-edged orifice | `mdot_orifice_pos(...)` / `mdot_orifice_pos_props(...)` | Cd-based, choked/subsonic |
| Short-tube | `mdot_short_tube_pos(...)` | Darcy friction + K_in/K_out minor losses; uses effective Cd, not Fanno |
| Slot channel | `mdot_slot_pos(...)` | Viscous laminar (Poiseuille) |
| Fanno flow | `mdot_fanno_tube(...)` | Friction-limited choked flow (available but not default in v10) |
| Viscosity helper | `mu_air_sutherland(T)` | Sutherland dynamic viscosity for air |

**Sentinel values** (defined in `graph.py`):
- `EXT_NODE = -1` — marks external-atmosphere boundary nodes
- `M_SAFE`, `T_SAFE` — minimum safe mass/temperature to avoid division by zero
- `P_STOP` — solver early-stop pressure threshold

---

## Thermodynamic Modes

| Mode | T evolution | Notes |
|------|------------|-------|
| `"isothermal"` | `T = T₀` (fixed) | Fast; valid when walls are highly conductive |
| `"intermediate"` | Solves `dT/dt` with wall heat transfer | Default recommendation |
| `"variable"` | Same as intermediate + NASA-7 `cp(T)`, `cv(T)`, `γ(T)` | Most accurate |

Use `thermo.py` functions when `mode == "variable"`. The NASA-7 fits are valid for
air in the range 200–1000 K.

---

## External Pressure Profiles (`profiles.py`)

Profiles define `P_ext(t)` for the environment node:

| Profile | Description |
|---------|-------------|
| `"linear"` | Linearly drops from `P0` to `P_final` over `t_final` |
| `"step"` | Instantaneous step at `t_step` |
| `"barometric"` | Exponential decay (e.g., rocket ascent) |
| `"table"` | Interpolated from user-supplied `(t, P)` data |

---

## Output Artifacts (`io.py`)

| File | Contents |
|------|----------|
| `run.json` | Reproducibility metadata: git commit, Python version, all input parameters |
| `summary.csv` | Per-edge metrics: max ΔP, peak time, flow regime, peak type |
| `*.npz` | Compressed time series: `t, m, T, P, P_ext, τ_exit` |
| `*_meta.json` | Peak diagnostics and validity flags |
| `*_validity.json` | Physical validity summary (Re, Mach, acoustic checks) |

---

## Physics Conventions

- **Ideal gas EOS:** `P = m R T / V`
- **Mass balance:** `dm/dt = Σ ṁ_in − Σ ṁ_out`
- **Energy balance (non-isothermal):** `m cv dT/dt = Σ ṁ_in (cp T_in − cv T) − Σ ṁ_out (R T) + h A ΔT`
- **Choking condition:** `P_up / P_down ≥ π_c = ((γ+1)/2)^(γ/(γ−1))`
- **Default constants:** `γ = 1.4`, `R = 287.05 J/(kg·K)`
- **No state clipping:** The solver avoids aggressive clipping; safety guards apply
  only to denominators (`M_SAFE`, `T_SAFE`).

---

## Validity Checks (`validity.py`)

After each solve, `validity.py` produces 7 flags via `evaluate_validity_flags()`:

- **`state_integrity`:** No NaN/Inf, positive mass and temperature.
- **`acoustic_uniformity_0D`:** Is dynamic timescale >> acoustic propagation time?
  (0D assumption must hold.)
- **`thermo_fit_range`:** Are temperatures within NASA-7 polynomial range (200–1000 K)?
- **`slot_laminarity`:** Is slot channel Reynolds number in the laminar regime?
- **`short_tube_flow`:** Re, Mach, K_tot, Cd_eff, Fanno choking fraction.
- **`knudsen_regime`:** Knudsen number check (continuum vs slip vs free-molecular).
- **`external_pressure_units`:** Warns if P_ext values look like mmHg entered as Pa.

Always inspect validity flags before trusting results.

---

## Testing Conventions

- **Gate tests (`test_gates.py`):** Compare solver to hardcoded analytic solutions.
  Tolerances: `< 0.5%` for pressures, `< 0.1%` for mass conservation.
- **Validity checks are masked:** Only high-pressure region (`P > 0.01 P0`) is
  validated to avoid noise near equilibrium.
- **No flaky tests:** All tests use deterministic inputs with fixed random seeds
  where sampling is involved.
- **GUI tests skip gracefully** if PySide6 is not installed.

---

## Code Style Rules

- **Formatter:** Black (line-length 88). Run before committing.
- **Linter:** Ruff with rules `F, E, I, B, UP` (no `E501`).
- **Type hints:** Used throughout; mypy is available but not enforced in CI.
- **Dataclasses:** Prefer `@dataclass(frozen=True)` for config objects.
- **Pure functions:** Flow model functions in `flow.py` are stateless.
- **No magic numbers:** Physical constants live in `constants.py`.
- **Archive is read-only:** `archive/venting_v84.py` is reference only; do not
  import from it in production code or tests.

---

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) runs on every push and pull request:

1. Python matrix: **3.10, 3.11, 3.12** on `ubuntu-latest`
2. `pip check` — dependency consistency
3. `ruff check .` — linting
4. `black --check .` — formatting
5. `pytest -q --cov=venting` — tests with coverage

All steps must pass before merging.

---

## Known Limitations

- **Not CFD:** No spatial velocity or temperature gradients, no acoustic wave
  propagation, no shock capturing.
- **C_d is the primary uncertainty:** Always run a Cd sweep to understand
  sensitivity before drawing conclusions.
- **Short-tube model:** Lossy-nozzle via Cd_eff (v10 default). Fanno
  friction-choking is implemented but not the default.
- **Streaming vs. batch solve:** Results agree within integrator tolerances but
  may differ at the last decimal place due to chunking.
- **GUI packaging:** Windows EXE via PyInstaller is automated in `build_windows.yml`;
  other platforms are not yet automated.

---

## Quick Reference: Adding a New Feature

1. **New flow model** → add a function to `flow.py`; wire into `graph.py` edge
   dispatch; update `state_layout.py` if state vector changes.
2. **New external profile** → add a class/function to `profiles.py`; register it
   in the CLI (`cli.py`) and in `cases.py`.
3. **New CLI subcommand** → add a `add_parser` block in `cli.py` and a handler
   function; keep handler thin, delegate to `run.py` or domain modules.
4. **New test** → add to `tests/`; gate tests go in `test_gates.py` if they have
   an analytic solution, otherwise create a regression test file.
5. **Always run:** `ruff check . --fix && black . && pytest -q` before committing.
