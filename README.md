# venting (v8.4 only)

This repository contains **one active model version: v8.4** of the edge-based 0D venting solver.
Legacy versions are archived and must not participate in CLI, tests, or CI.

## Install

```bash
pip install -r requirements-dev.txt
pip install -e .
```

## CLI

```bash
python -m venting.cli gate --single
python -m venting.cli gate --two
python -m venting.cli sweep --profile linear --d-int 2 --d-exit 2 --cd-int 0.62 --cd-exit 0.62
python -m venting.cli thermal --profile linear --d 2 --h-list 0,1,5,15
python -m venting.cli sweep2d --profile linear --d-int-list 1.5,2.0 --d-exit-list 1.5,2.0
```

Results are stored in `results/<timestamp>_<case>/`.

## Verification

```bash
pytest -q
ruff check .
black --check .
```

## Model policy

- No state clipping for pressure/temperature.
- Only denominator safeguards (`T_SAFE`, tiny mass threshold in `dT` denominator).
- Intermediate model uses upstream temperature for mass-flow terms.
