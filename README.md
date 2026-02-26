# venting (v8.4 only)

This repository contains only one active runtime model version: **v8.4**.
All legacy scripts are archived in `archive/` and are not imported by package runtime.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python -m venting.cli --help
python -m venting.cli gate
```

## CLI examples

```bash
python -m venting.cli gate --single
python -m venting.cli gate --two
python -m venting.cli sweep --profile linear --d-int 2 --d-exit 2 --cd-int 0.62 --cd-exit 0.62
python -m venting.cli thermal --profile linear --d 2 --h-list 0,1,5,15
python -m venting.cli sweep2d --profile linear --d-int-list 1.5,2.0 --d-exit-list 1.5,2.0
```

All run artifacts are saved in `results/<timestamp>_<case>/` and include:
- `run.json`
- `summary.csv`
- binary solver output (`*.npz`) and metadata (`*_meta.json`)

## Quality checks

```bash
ruff check .
black --check .
pytest -q --cov=venting --cov-report=term-missing
```
