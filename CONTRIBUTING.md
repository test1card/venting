# Contributing

1. Create a virtual environment and install project + dev extras:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

2. Run quality checks:

```bash
ruff check .
black --check .
pytest -q --cov=venting --cov-report=term-missing
```

3. Keep only v10 active in runtime package (`src/venting`).
   Any legacy files must remain in `archive/` and not be imported by CLI/tests.
