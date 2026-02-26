# Contributing

1. Install dev dependencies:
   ```bash
   pip install -r requirements-dev.txt
   pip install -e .
   ```
2. Run checks locally:
   ```bash
   ruff check .
   black --check .
   pytest -q
   ```
3. Keep active code on v8.4 only. Legacy variants must stay in `archive/`.
