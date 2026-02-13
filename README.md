# JFM_CS — Spectral Analysis Pipeline for Bluff-Body Wakes

[![CI](https://github.com/ricardofrantz/JFM_CS/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ricardofrantz/JFM_CS/actions/workflows/ci.yml)
[![Docs](https://github.com/ricardofrantz/JFM_CS/actions/workflows/docs.yaml/badge.svg?branch=main)](https://github.com/ricardofrantz/JFM_CS/actions/workflows/docs.yaml)
[![PyPI - Package](https://img.shields.io/pypi/v/dsgbr.svg)](https://pypi.org/project/dsgbr/)
[![License](https://img.shields.io/github/license/ricardofrantz/JFM_CS)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/dsgbr.svg)](https://pypi.org/project/dsgbr/)

Spectral analysis pipeline for fluid-dynamics wake data, centered on DSGBR peak detection and scientific reproducibility across spectral/phase-space analyses.

This repository powers our JFM manuscript work on bluff-body wake transitions and supports publication-ready outputs in `data/`, `figures/`, and `paper/`.

## Prerequisites

```bash
int25   # Loads Intel oneAPI 2025 + Python venv (~/.venv)
```

## Quick Start

```bash
# Single case
python process.py --shape sphere --cases 332

# Parallel processing (4 workers)
OMP_NUM_THREADS=4 python process.py --workers 4 --shape both

# Fast peak-only refresh
python process.py --peaks

# Fractal diagnostics
python process.py --fractal

# Figures
python plots/plot_main.py
```

## Install DSGBR

```bash
uv pip install -e .[dev]
uv pip install dsgbr
```

```python
from dsgbr import dsgbr_detector, detect_peaks_case_adaptive, DetectionConfig
```

Optional plotting support:

```bash
uv pip install dsgbr[plotting]
```

## Repository Layout

| Path              | Purpose                                        |
| ----------------- | ---------------------------------------------- |
| `process.py`      | Main spectral workflow entry point             |
| `case_configs.py` | Case definitions and DSGBR parameter sets      |
| `core/`           | Shared scientific utilities                    |
| `dsgbr/`          | DSGBR detector source and API                  |
| `plots/`          | Publication figure scripts                     |
| `docs/`           | Supplemental references and technical notes    |
| `paper/`          | Manuscript sources (Overleaf-synced submodule) |

## DSGBR package health checks

```bash
uv run pytest
uv build
uv run twine check dist/*
```

## Release and QA

CI is orchestrated with `uv` using:

- `.github/workflows/ci.yml`: lint + tests + package build matrix
- `.github/workflows/docs.yaml`: docs link checks
- `.github/workflows/check-links.yaml`: markdown link checks
- `.github/workflows/update-pre-commits.yaml`: optional pre-commit refresh automation

Package metadata uses `pyproject.toml` and `LICENSE` for SPDX-compliant proprietary licensing.

## Project highlights

### DSGBR peak detector (`dsgbr/`)

Dual Savitzky–Golay Baseline Ratio (DSGBR) is tuned for high dynamic range spectra, with adaptive ratio tests and frequency-aware pruning.

### Scientific pipeline features

- Periodic/intermittent flow regime classification from spectral signatures
- Correlation dimension and delay-embedding analysis in `core/`
- Reproducible figures and artifact-rich intermediate `.npz` outputs

## Dependencies

```bash
uv pip install numpy scipy matplotlib SciencePlots
```

Core data root defaults to `~/data/` (`core/spcfunc.py`).
