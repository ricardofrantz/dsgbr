# Changelog

## [0.1.0] - 2026-02-13

### Added

- Initial release as standalone package.
- `dsgbr_detector()` — five-stage peak detection pipeline.
- `DetectionConfig` — frozen dataclass with 17 tunable parameters.
- `compute_support_series()` — visualization helper returning intermediate arrays.
- `select_peaks_by_frequency_bands()` — band-balanced peak down-selection.
- `find_nearest_frequency()` — closest-frequency lookup utility.
- Backward-compatible `detect_peaks_case_adaptive()` wrapper.
- `DSGBR.py` shim for `from dsgbr.DSGBR import ...` import paths.
- Input validation in `DetectionConfig.__post_init__`.
- NumPy-style docstrings throughout.
- Comprehensive test suite with Hypothesis property-based tests.
- BSD 3-Clause license.
