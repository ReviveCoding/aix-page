# Reproducibility

The canonical language is Python 3.12. CPU checks deliberately do not install or require CUDA.

## Level 1 — CPU smoke, no KDD

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy srcix_page
.\.venv\Scripts\python.exe -m pytest -q
```

## Level 2 — synthetic method qualification

The statistical, leakage, experiment, policy, oracle-firewall, and replay tests use tiny deterministic fixtures:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\statistical tests\integration tests\leakage testseplay -q
```

These tests validate methods; they do not regenerate or alter the published locked experiment.

## Level 3 — KDD modeling

Independently obtain the official data and follow [Data access](DATA_ACCESS.md). Build the streaming Parquet/DuckDB layer, stable roles, cross-fitted features, and bounded model datasets with the scripts in `scripts/`. Full reproduction is storage- and time-intensive and the data remain outside Git.

## Level 4 — GPU model qualification

The qualified local reference used Windows, Python 3.12.10, XGBoost 3.4.1, PyTorch 2.11.0+cu130, and an RTX 4090 Laptop GPU. PyTorch's CUDA 13.0 wheel came from `https://download.pytorch.org/whl/cu130`; a generic CPU-only install does not reproduce that qualification.

```powershell
.\scripts\install_gpu.ps1
```

That script verifies XGBoost build metadata, runs real CUDA tree training, checks `torch.cuda.is_available()`, and synchronizes a CUDA matrix multiplication. Results are hardware-dependent.

## Evidence replay boundary

The public provenance manifest hashes each compact evidence file against its local canonical source. Row-level locked outcomes and model binaries are intentionally excluded, so the complete private replay cannot be performed from the public clone alone. Public CI validates code and compact methods without downloading KDD or running CUDA.
