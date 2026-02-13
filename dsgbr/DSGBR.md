# DSGBR Reference Guide

Dual Savitzky–Golay Baseline Ratio (DSGBR) is a spectral peak detector built for
broadband and quasi-periodic signals where many peaks can coexist in dense frequency
regions. This guide documents the standalone package behavior for external users.

## Core idea

For frequencies `f` and PSD `P(f)`:

1. Build `SEARCH` by smoothing `P(f)` (typically on log-scale).
2. Build `BASELINE` from a longer-scale Savitzky–Golay smoothing pass.
3. Candidate peaks are local maxima in `SEARCH` where `SEARCH / BASELINE >= ratio_threshold`.
4. Apply spacing and ultra-low-frequency guardrails.
5. Optionally down-select across frequency bands to satisfy `max_peaks`.

## Public API

- `dsgbr_detector(frequencies, psd, *, case_info=None, return_support=False)`
  - Returns `(peak_frequencies, peak_amplitudes)`.
  - If `return_support=True`, returns `(peak_frequencies, peak_amplitudes, support)`.
- `compute_support_series(frequencies, psd, case_info=None)`
  - Returns internal arrays used for reproducibility and plotting (`SEARCH`, `BASELINE`, ratio, indices).
- `select_peaks_by_frequency_bands(peak_frequencies, peak_heights, *, max_peaks, strategy, n_bands)`
  - Down-selects peaks by band while preserving representation across frequency ranges.
- `DetectionConfig`
  - Frozen dataclass for configuration.
- `detect_peaks_case_adaptive(...)`
  - Backward-compatible wrapper around `dsgbr_detector(...)`.

`DSGBR_PARAM_ALIASES` maps compact keys (`RT`, `SW`, `BWF`, `DH`, `DL`, `SF`, `MP`) to canonical names.

## Defaults (important)

- `smooth_window = 3`
- `smooth_polyorder = 2`
- `smooth_on_log = True`
- `baseline_window_frac = 0.001`
- `baseline_on_log = True`
- `ratio_threshold = 1.8`
- `distance_low = 2`
- `distance_high = 1`
- `max_peaks = 25`
- `band_strategy = "proportional"`

## Return metadata from support

`compute_support_series` includes:

- `search_series`
- `baseline_series`
- `local_baseline`
- `ratio_series`
- `rthreshold`
- `detector_config`
- `candidate_indices`
- `accepted_indices`
- `peak_frequencies`
- `peak_heights`

For most users, only `peak_frequencies` and `peak_heights` are required.

## Package scope

This repository is independent of project-specific pipelines and was extracted
for direct reuse in external projects.

