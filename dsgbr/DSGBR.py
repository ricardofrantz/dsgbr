#!/usr/bin/env python3
"""Dual Savitzky–Golay Baseline Ratio (DSGBR) peak detection utilities.

The DSGBR detector applies two Savitzky–Golay smoothing passes — one to build the
SEARCH series, another to obtain a broader BASELINE — and accepts peaks where the
SEARCH/BASELINE ratio exceeds a configurable threshold. This approach preserves
peak height/width/position better than moving-average filters and remains stable
in dense spectral combs.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.signal import find_peaks, peak_widths, savgol_filter

# Bidirectional mapping for DSGBR parameters
# Allows use of either short keys (RT, SW, BWF) or full parameter names
DSGBR_PARAM_ALIASES = {
    # Short -> Long mappings
    "RT": "ratio_threshold",
    "SW": "smooth_window",
    "BWF": "baseline_window_frac",
    "DL": "distance_low",
    "DH": "distance_high",
    "SF": "switch_frequency",
    "MP": "max_detected_peaks",  # Maximum peaks to detect and save to NPZ (LOWEST frequencies first)
}

try:  # SciPy <1.12 compatibility
    from scipy.signal import PeakPropertyWarning  # type: ignore
except ImportError:  # pragma: no cover - fallback for older SciPy

    class PeakPropertyWarning(RuntimeWarning):
        """Fallback warning class when SciPy does not expose PeakPropertyWarning."""

        pass


@dataclass(frozen=True)
class DetectionConfig:
    """Configuration for the DSGBR detector.

    Parameters are organized by function:
    - SEARCH series: Peak detection smoothing parameters
    - BASELINE series: Baseline estimation parameters
    - Detection: Peak acceptance criteria
    - Spacing: Frequency-dependent peak separation rules
    - ULF guardrail: Ultra-low-frequency filtering
    - Selection: Final peak down-selection strategy
    """

    # ==================== SEARCH SERIES PARAMETERS ====================
    # Used in _build_search_series() for peak detection smoothing

    smooth: str = "savgol"  # Line 319 - smoothing method type
    """Smoothing algorithm for SEARCH series construction."""

    smooth_window: int = 3  # Line 321 - window size for Savitzky-Golay filter (SW)
    """Window size for SEARCH series smoothing (odd integer ≥ 3).
    Commonly abbreviated as SW in sensitivity analysis and testing."""

    smooth_polyorder: int = 2  # Line 326 - polynomial order for Savitzky-Golay
    """Polynomial order for Savitzky-Golay filtering (must be < smooth_window)."""

    smooth_on_log: bool = True  # Line 325 - whether to smooth on log scale
    """Apply smoothing to log₁₀(PSD) instead of linear PSD."""

    # ==================== BASELINE SERIES PARAMETERS ====================
    # Used in _build_baseline_series() for baseline estimation

    baseline_window: int | None = None  # Line 338 - fixed baseline window size
    """Fixed window size for baseline smoothing (overrides baseline_window_frac if set)."""

    baseline_window_frac: float = 0.001  # Line 341 - fractional baseline window (BWF)
    """Baseline window as fraction of data length (e.g., 0.001 = 0.1% of data points).
    Commonly abbreviated as BWF in sensitivity analysis and testing."""

    baseline_on_log: bool = True  # Line 335, 350, 352 - baseline smoothing domain
    """Apply baseline smoothing to log₁₀(SEARCH) instead of linear SEARCH."""

    # ==================== DETECTION PARAMETERS ====================
    # Used for peak acceptance criteria

    ratio_threshold: float = 1.8  # Lines 235, 436 - SEARCH/BASELINE ratio threshold (RT)
    """Minimum SEARCH/BASELINE ratio for peak acceptance (≥ 1.0).
    Commonly abbreviated as RT in sensitivity analysis and testing."""

    # ==================== SPACING PARAMETERS ====================
    # Used for frequency-dependent peak separation rules

    switch_frequency: float = 2e-2  # Line 244 - frequency threshold for spacing rules
    """Frequency threshold: f ≥ switch_frequency uses distance_high, else distance_low."""

    distance_low: int = 2  # Line 242 - minimum spacing for low frequencies
    """Minimum bin separation for peaks below switch_frequency."""

    distance_high: int = 1  # Line 243 - minimum spacing for high frequencies
    """Minimum bin separation for peaks at or above switch_frequency."""

    # ==================== ULF GUARDRAIL PARAMETERS ====================
    # Used in _apply_ulf_guardrail() for ultra-low-frequency filtering

    ulf_fmax: float = 1e-3  # Lines 361, 365 - ULF band upper limit
    """Maximum frequency considered ultra-low-frequency (ULF) for special filtering."""

    ulf_min_q: float = 9.0  # Line 393 - minimum quality factor for ULF peaks
    """Minimum Q-factor (f_center/FWHM) for ULF peak retention."""

    ulf_max_points: int = 5  # Line 398 - maximum number of ULF peaks
    """Maximum number of ULF peaks to retain (ranked by amplitude)."""

    # ==================== SELECTION PARAMETERS ====================
    # Used for final peak down-selection in select_peaks_by_frequency_bands()

    max_peaks: int = 25  # Lines 266, 268 - maximum total peaks to return
    """Maximum number of peaks to return from detection."""

    band_strategy: str = "proportional"  # Line 272 - frequency band allocation method
    """Strategy for allocating peaks across frequency bands: 'proportional' or 'equal'."""

    n_bands: int = 10  # Line 273 - number of frequency bands for selection
    """Number of logarithmic frequency bands for peak allocation."""

    @classmethod
    def from_case_info(cls, case_info: Any | None) -> DetectionConfig:
        if not isinstance(case_info, dict) or not case_info:
            return cls()

        def _convert(value: Any, dtype: type):
            if value is None:
                raise ValueError
            if dtype is bool:
                if isinstance(value, str):
                    return value.strip().lower() in {"1", "true", "yes", "on"}
                return bool(value)
            if dtype is int:
                return int(value)
            if dtype is float:
                return float(value)
            return dtype(value)

        aliases: dict[str, tuple[type, tuple[str, ...]]] = {
            "smooth": (str, ("smooth",)),
            "smooth_window": (int, ("smooth_window", "SW")),
            "smooth_polyorder": (int, ("smooth_polyorder",)),
            "smooth_on_log": (bool, ("smooth_on_log",)),
            "baseline_window": (int, ("baseline_window", "prominence_window", "baseline")),
            "baseline_window_frac": (
                float,
                ("baseline_window_frac", "prominence_window_frac", "BWF"),
            ),
            "baseline_on_log": (bool, ("baseline_on_log", "prominence_on_log")),
            "ratio_threshold": (float, ("ratio_threshold", "RT")),
            "switch_frequency": (
                float,
                ("switch_frequency", "two_pass_high_fmin", "two_pass_fmin", "SF"),
            ),
            "distance_low": (int, ("distance_low", "distance", "DL")),
            "distance_high": (int, ("distance_high", "two_pass_distance_high", "DH")),
            "ulf_fmax": (float, ("ulf_fmax",)),
            "ulf_min_q": (float, ("ulf_min_q",)),
            "ulf_max_points": (int, ("ulf_max_points",)),
            "max_peaks": (int, ("max_peaks", "MP")),
            "band_strategy": (str, ("band_strategy",)),
            "n_bands": (int, ("n_bands",)),
        }

        data: dict[str, Any] = {}
        for field, (dtype, keys) in aliases.items():
            for key in keys:
                if key in case_info:
                    try:
                        value = _convert(case_info[key], dtype)
                    except (TypeError, ValueError):
                        continue
                    else:
                        data[field] = value
                        break

        if data.get("baseline_window") is not None and data["baseline_window"] <= 0:
            data.pop("baseline_window", None)

        return cls(**data)

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


def find_nearest_frequency(
    target_freq: float, frequencies: np.ndarray, heights: np.ndarray
) -> float:
    """Return the closest available frequency to ``target_freq`` for reporting."""

    if target_freq <= 0 or len(frequencies) == 0:
        return 0.0
    idx = int(np.argmin(np.abs(np.asarray(frequencies) - float(target_freq))))
    return float(frequencies[idx])


def select_peaks_by_frequency_bands(
    peak_frequencies: np.ndarray,
    peak_heights: np.ndarray,
    *,
    max_peaks: int = 200,
    strategy: str = "proportional",
    n_bands: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Down-select peaks by distributing allowance across logarithmic bands."""

    peak_frequencies = np.asarray(peak_frequencies)
    peak_heights = np.asarray(peak_heights)

    if peak_frequencies.size <= max_peaks:
        return peak_frequencies, peak_heights

    freq_pos = peak_frequencies[peak_frequencies > 0]
    if freq_pos.size == 0:
        return np.array([]), np.array([])

    freq_min = float(freq_pos.min())
    freq_max = float(peak_frequencies.max())
    if freq_min > 1e-4:
        freq_min = 1e-4

    bands = max(1, int(n_bands))
    log_min, log_max = np.log10(freq_min), np.log10(freq_max)
    band_edges = np.logspace(log_min, log_max, bands + 1)

    candidates = []
    for i in range(bands):
        mask = (peak_frequencies >= band_edges[i]) & (peak_frequencies < band_edges[i + 1])
        candidates.append(np.where(mask)[0])

    allotments = [0] * bands
    if strategy == "equal":
        per_band = max_peaks // bands
        remainder = max_peaks % bands
        for i in range(bands):
            allotments[i] = per_band + (1 if i < remainder else 0)
    else:
        remaining = max_peaks
        for i, cand in enumerate(candidates):
            if cand.size > 0:
                allotments[i] = 1
                remaining -= 1
        if remaining > 0:
            weights = np.array([cand.size for cand in candidates], dtype=float)
            total = weights.sum() if weights.sum() > 0 else 1.0
            fractional = remaining * (weights / total)
            integer = np.floor(fractional).astype(int)
            allotments = [a + b for a, b in zip(allotments, integer.tolist(), strict=False)]
            leftover = remaining - int(integer.sum())
            if leftover > 0:
                order = np.argsort(fractional - integer)[::-1]
                for idx in order[:leftover]:
                    allotments[idx] += 1

    selected_freqs: list[float] = []
    selected_heights: list[float] = []
    for i, cand in enumerate(candidates):
        n_select = min(allotments[i], cand.size)
        if n_select <= 0:
            continue
        idx = np.argsort(peak_heights[cand])[::-1][:n_select]
        selected_freqs.extend(peak_frequencies[cand][idx])
        selected_heights.extend(peak_heights[cand][idx])
    selected_freqs = np.asarray(selected_freqs)
    selected_heights = np.asarray(selected_heights)

    if selected_freqs.size == 0:
        return selected_freqs, selected_heights

    freq_order = np.argsort(selected_freqs)
    selected_freqs = selected_freqs[freq_order]
    selected_heights = selected_heights[freq_order]

    if selected_freqs.size > max_peaks:
        selected_freqs = selected_freqs[:max_peaks]
        selected_heights = selected_heights[:max_peaks]

    return selected_freqs, selected_heights


def dsgbr_detector(
    frequencies: np.ndarray,
    psd: np.ndarray,
    *,
    case_info: Any | None = None,
    return_support: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run DSGBR peak detection and optionally return the intermediate support."""

    frequencies = np.asarray(frequencies)
    psd = np.asarray(psd)

    if frequencies.size == 0 or psd.size == 0:
        support = _initial_support()
        return (
            (np.array([]), np.array([]), support)
            if return_support
            else (np.array([]), np.array([]))
        )

    cfg = DetectionConfig.from_case_info(case_info)
    search_series = _build_search_series(psd, cfg)
    baseline_series = _build_baseline_series(search_series, cfg)
    ratio_series = search_series / np.maximum(baseline_series, 1e-300)

    support = _build_support(search_series, baseline_series, ratio_series, cfg)

    candidate_indices, _ = find_peaks(search_series, distance=1)
    support["candidate_indices"] = candidate_indices.copy()
    if candidate_indices.size == 0:
        return (
            (np.array([]), np.array([]), support)
            if return_support
            else (np.array([]), np.array([]))
        )

    candidate_indices = candidate_indices[ratio_series[candidate_indices] >= cfg.ratio_threshold]
    support["candidate_indices"] = candidate_indices.copy()
    if candidate_indices.size == 0:
        return (
            (np.array([]), np.array([]), support)
            if return_support
            else (np.array([]), np.array([]))
        )

    order = np.argsort(search_series[candidate_indices])[::-1]
    accepted: list[int] = []
    low_distance = max(1, int(cfg.distance_low))
    high_distance = max(1, int(cfg.distance_high))
    switch_frequency = max(0.0, float(cfg.switch_frequency))
    for idx in order:
        peak_idx = int(candidate_indices[idx])
        freq = float(frequencies[peak_idx])
        min_dist = high_distance if freq >= switch_frequency else low_distance
        if all(abs(peak_idx - existing) >= min_dist for existing in accepted):
            accepted.append(peak_idx)

    if not accepted:
        support["accepted_indices"] = np.array([], dtype=int)
        return (
            (np.array([]), np.array([]), support)
            if return_support
            else (np.array([]), np.array([]))
        )

    accepted_idx = np.array(sorted(set(accepted)), dtype=int)
    accepted_idx = _refine_peak_indices(accepted_idx, psd)
    accepted_idx = _apply_ulf_guardrail(accepted_idx, frequencies, search_series, cfg)
    support["accepted_indices"] = accepted_idx.copy()
    if accepted_idx.size == 0:
        return (
            (np.array([]), np.array([]), support)
            if return_support
            else (np.array([]), np.array([]))
        )

    peak_f = frequencies[accepted_idx]
    peak_h = psd[accepted_idx]

    max_peaks = max(1, int(cfg.max_peaks))
    if peak_f.size > max_peaks:
        peak_f, peak_h = select_peaks_by_frequency_bands(
            peak_f,
            peak_h,
            max_peaks=max_peaks,
            strategy=cfg.band_strategy,
            n_bands=cfg.n_bands,
        )
        if peak_f.size == 0:
            support["accepted_indices"] = np.array([], dtype=int)
            return (
                (np.array([]), np.array([]), support)
                if return_support
                else (np.array([]), np.array([]))
            )

    order = np.argsort(peak_f)
    peak_f = peak_f[order]
    peak_h = peak_h[order]

    # CRITICAL: Peaks are sorted by frequency (ascending)
    # This ensures low-frequency fundamentals are ALWAYS preserved
    # when max_detected_peaks limit is applied

    support["peak_frequencies"] = peak_f
    support["peak_heights"] = peak_h
    final_indices = np.array([int(np.argmin(np.abs(frequencies - f))) for f in peak_f], dtype=int)
    support["accepted_indices"] = final_indices

    if return_support:
        return peak_f, peak_h, support
    return peak_f, peak_h


def detect_peaks_case_adaptive(
    frequencies: np.ndarray,
    psd: np.ndarray,
    case_info: Any | None = None,
    *,
    return_support: bool = False,
):
    """Compatibility wrapper used by the pipeline."""

    return dsgbr_detector(frequencies, psd, case_info=case_info, return_support=return_support)


def compute_support_series(
    frequencies: np.ndarray,
    psd: np.ndarray,
    case_info: Any | None = None,
) -> dict[str, Any]:
    """Return SEARCH, BASELINE, ratio, thresholds, and indices for plotting."""

    _, _, support = dsgbr_detector(frequencies, psd, case_info=case_info, return_support=True)
    return support


def _build_search_series(psd: np.ndarray, cfg: DetectionConfig) -> np.ndarray:
    if cfg.smooth and cfg.smooth.lower() != "none":
        try:
            win = int(cfg.smooth_window)
            if win % 2 == 0:
                win += 1
            if win >= 3 and win < len(psd):
                arr = np.log10(psd + 1e-300) if cfg.smooth_on_log else psd
                arr = savgol_filter(arr, window_length=win, polyorder=int(cfg.smooth_polyorder))
                return np.power(10.0, arr) if cfg.smooth_on_log else arr
        except Exception:
            pass
    return psd


def _build_baseline_series(search_series: np.ndarray, cfg: DetectionConfig) -> np.ndarray:
    base = search_series
    if cfg.baseline_on_log:
        base = np.log10(base + 1e-300)
    try:
        if cfg.baseline_window and int(cfg.baseline_window) > 0:
            win = int(cfg.baseline_window)
        elif cfg.baseline_window_frac and cfg.baseline_window_frac > 0:
            win = int(max(7, round(len(search_series) * float(cfg.baseline_window_frac))))
        else:
            win = max(15, (len(search_series) // 200) * 2 + 1)
        if win % 2 == 0:
            win += 1
        if 3 < win < len(search_series):
            base_sm = savgol_filter(base, window_length=win, polyorder=2)
        else:
            base_sm = base
        return np.power(10.0, base_sm) if cfg.baseline_on_log else base_sm
    except Exception:
        return search_series if not cfg.baseline_on_log else np.power(10.0, base)


def _apply_ulf_guardrail(
    indices: np.ndarray,
    frequencies: np.ndarray,
    search_series: np.ndarray,
    cfg: DetectionConfig,
) -> np.ndarray:
    if indices.size == 0 or cfg.ulf_fmax <= 0:
        return indices

    freq = np.asarray(frequencies)
    ul_mask = freq[indices] < cfg.ulf_fmax
    if not np.any(ul_mask):
        return indices

    ul_indices = indices[ul_mask]
    df = float(np.median(np.diff(freq))) if freq.size > 1 else 1.0

    valid_mask = search_series[ul_indices] > 0
    if not np.any(valid_mask):
        return indices[~ul_mask]

    target_indices = ul_indices[valid_mask]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PeakPropertyWarning)
        widths, _, _, _ = peak_widths(search_series, target_indices, rel_height=0.5)

    if widths.size == 0:
        return indices[~ul_mask]

    positive_widths = widths > 0
    if not np.any(positive_widths):
        return indices[~ul_mask]

    target_indices = target_indices[positive_widths]
    widths = widths[positive_widths]

    fwhm = widths * df
    q_vals = np.maximum(1e-12, freq[target_indices]) / np.maximum(1e-12, fwhm)
    keep = q_vals >= cfg.ulf_min_q
    ul_indices = target_indices[keep]
    if ul_indices.size == 0:
        return indices[~ul_mask]

    cap = max(0, int(cfg.ulf_max_points))
    if cap and ul_indices.size > cap:
        order = np.argsort(search_series[ul_indices])[::-1][:cap]
        ul_indices = ul_indices[order]

    combined = np.concatenate([indices[~ul_mask], ul_indices])
    combined = np.unique(combined.astype(int))
    return combined


def _initial_support() -> dict[str, Any]:
    cfg = DetectionConfig()
    baseline = np.array([], dtype=float)
    return {
        "search_series": np.array([], dtype=float),
        "baseline_series": baseline,
        "local_baseline": baseline,
        "ratio_series": np.array([], dtype=float),
        "rthreshold": np.array([], dtype=float),
        "detector_config": cfg.to_metadata(),
        "candidate_indices": np.array([], dtype=int),
        "accepted_indices": np.array([], dtype=int),
        "peak_frequencies": np.array([], dtype=float),
        "peak_heights": np.array([], dtype=float),
    }


def _build_support(
    search_series: np.ndarray,
    baseline_series: np.ndarray,
    ratio_series: np.ndarray,
    cfg: DetectionConfig,
) -> dict[str, Any]:
    return {
        "search_series": search_series,
        "baseline_series": baseline_series,
        "local_baseline": baseline_series,
        "ratio_series": ratio_series,
        "rthreshold": baseline_series * cfg.ratio_threshold,
        "detector_config": cfg.to_metadata(),
        "candidate_indices": np.array([], dtype=int),
        "accepted_indices": np.array([], dtype=int),
        "peak_frequencies": np.array([], dtype=float),
        "peak_heights": np.array([], dtype=float),
    }


def _refine_peak_indices(indices: np.ndarray, psd: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        return indices

    arr = np.asarray(psd)
    if arr.size == 0:
        return indices

    refined: list[int] = []
    upper = arr.size - 1

    for raw_idx in np.asarray(indices, dtype=int):
        idx = int(np.clip(raw_idx, 0, upper))
        left = max(0, idx - 3)
        right = min(upper, idx + 3)
        window = arr[left : right + 1]
        if window.size == 0:
            refined.append(idx)
            continue

        local = int(np.argmax(window)) + left
        best = local

        for _ in range(6):  # bounded hill-climb towards the nearest PSD maximum
            current = arr[best]
            left_idx = best - 1 if best > 0 else best
            right_idx = best + 1 if best < upper else best

            left_val = arr[left_idx] if left_idx < best else float("-inf")
            right_val = arr[right_idx] if right_idx > best else float("-inf")

            if left_val <= current and right_val <= current:
                break

            if right_val > left_val:
                if right_idx == best:
                    break
                best = right_idx
            else:
                if left_idx == best:
                    break
                best = left_idx

        refined.append(best)

    return np.array(sorted(set(refined)), dtype=int)


__all__ = [
    "DetectionConfig",
    "compute_support_series",
    "detect_peaks_case_adaptive",
    "dsgbr_detector",
    "find_nearest_frequency",
    "select_peaks_by_frequency_bands",
]
