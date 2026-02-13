# DSGBR

[![CI](https://github.com/ricardofrantz/dsgbr/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ricardofrantz/dsgbr/actions/workflows/ci.yml)
[![Docs](https://github.com/ricardofrantz/dsgbr/actions/workflows/docs.yaml/badge.svg?branch=main)](https://github.com/ricardofrantz/dsgbr/actions/workflows/docs.yaml)
[![PyPI](https://img.shields.io/pypi/v/dsgbr.svg)](https://pypi.org/project/dsgbr/)
[![License](https://img.shields.io/badge/license-proprietary-red.svg)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/dsgbr.svg)](https://pypi.org/project/dsgbr/)

## Why DSGBR exists

**DSGBR** (Dual Savitzky–Golay Baseline Ratio) is a reusable spectral peak detector
for frequency-domain signals.
It was designed for robust detection in dense, noisy spectra and is published as an
independent Python package so it can be used in any project.

The detector builds a short-scale `SEARCH` signal and a longer-scale `BASELINE` signal
using Savitzky–Golay filtering. A peak is accepted when `SEARCH / BASELINE`
passes a configurable ratio threshold and additional spacing/guardrail rules.

## Install

```bash
uv pip install dsgbr
uv pip install -e ".[dev]"   # local development install
```

## Quick start

```python
import numpy as np
from dsgbr import DetectionConfig, dsgbr_detector

frequencies = np.array([1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2])
psd = np.array([0.9, 1.2, 2.1, 8.0, 3.0, 5.0, 1.2])

case_info = {
    "smooth": "savgol",
    "smooth_window": 5,
    "baseline_window_frac": 0.2,
    "ratio_threshold": 1.8,
    "max_peaks": 8,
}

peak_f, peak_h = dsgbr_detector(frequencies, psd, case_info=case_info)
print("peaks", peak_f, peak_h)

config = DetectionConfig.from_case_info(case_info)
print("alias map", config.ratio_threshold)
```

```python
from dsgbr import compute_support_series

support = compute_support_series(frequencies, psd, case_info={"smooth": "savgol"})
print(sorted(support.keys()))
```

## Public API

- `dsgbr_detector(...)`  
  Detect peaks from `frequencies` and `psd`.
- `detect_peaks_case_adaptive(...)`  
  Backward-compatible API alias.
- `compute_support_series(...)`  
  Return debug arrays used to interpret detection decisions.
- `DetectionConfig` and `DSGBRDetectionConfig`  
  Typed configuration object with aliases (`RT`, `SW`, `BWF`, `DH`, `DL`, `SF`, `MP`).
- `select_peaks_by_frequency_bands(...)`  
  Post-process high-count detections with band-aware down-selection.

## Development and quality gates

- Package tooling: `pyproject.toml`
- QA/test extras:
  - `uv pip install -e ".[qa]"`
  - `uv pip install -e ".[tests]"`
  - `uv run pytest`
  - `uv build`
  - `uv run twine check dist/*`

### Optional plotting support

```bash
uv pip install -e ".[plotting]"
```

This adds optional helpers in plotting and examples that depend on `matplotlib`.

## Repository structure

- `dsgbr/DSGBR.py` — core detection implementation
- `dsgbr/__init__.py` — public package API
- `dsgbr/DSGBR.md` — algorithm reference and practical parameter guide
- `tests/` — deterministic package tests for importability and behavior
- `.github/` — CI, release, and repository automation
- `pyproject.toml` — packaging and tooling metadata
