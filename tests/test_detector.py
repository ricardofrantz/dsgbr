"""Tests for the core DSGBR detection pipeline."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from dsgbr import compute_support_series, dsgbr_detector


class TestEmptyAndFlat:
    """Behavior on degenerate inputs."""

    def test_empty_input(self, empty_spectrum):
        f, p = empty_spectrum
        peak_f, peak_h = dsgbr_detector(f, p)
        assert peak_f.size == 0
        assert peak_h.size == 0

    def test_empty_with_support(self, empty_spectrum):
        f, p = empty_spectrum
        peak_f, _peak_h, support = dsgbr_detector(f, p, return_support=True)
        assert peak_f.size == 0
        assert support["search_series"].size == 0
        assert support["detector_config"]["ratio_threshold"] == 1.8

    def test_flat_spectrum_no_peaks(self, flat_spectrum):
        f, p = flat_spectrum
        peak_f, _peak_h = dsgbr_detector(f, p)
        assert peak_f.size == 0

    def test_flat_spectrum_with_low_threshold(self, flat_spectrum):
        f, p = flat_spectrum
        peak_f, _ = dsgbr_detector(f, p, case_info={"RT": 1.0})
        # Even with RT=1.0, a perfectly flat spectrum has no local maxima
        # in the search series, so no peaks should be detected
        assert peak_f.size == 0


class TestKnownPeaks:
    """Detection of injected peaks."""

    def test_detects_all_known_peaks(self, known_peaks_spectrum):
        f, p, spike_indices = known_peaks_spectrum
        cfg = {"smooth": "none", "ratio_threshold": 1.2, "baseline_window": 61, "max_peaks": 10}
        peak_f, _ = dsgbr_detector(f, p, case_info=cfg)
        known_freqs = set(np.round(f[spike_indices], 12))
        detected_freqs = set(np.round(peak_f, 12))
        assert known_freqs.issubset(detected_freqs)

    def test_output_sorted_ascending(self, known_peaks_spectrum):
        f, p, _ = known_peaks_spectrum
        cfg = {"smooth": "none", "ratio_threshold": 1.2, "baseline_window": 61}
        peak_f, _ = dsgbr_detector(f, p, case_info=cfg)
        assert np.all(np.diff(peak_f) > 0)

    def test_support_dict_structure(self, known_peaks_spectrum):
        f, p, _ = known_peaks_spectrum
        cfg = {"smooth": "none", "ratio_threshold": 1.2, "baseline_window": 61}
        _, _, support = dsgbr_detector(f, p, case_info=cfg, return_support=True)
        required_keys = {
            "search_series",
            "baseline_series",
            "local_baseline",
            "ratio_series",
            "rthreshold",
            "detector_config",
            "candidate_indices",
            "accepted_indices",
            "peak_frequencies",
            "peak_heights",
        }
        assert required_keys.issubset(support.keys())
        assert support["accepted_indices"].dtype == int


class TestRatioThreshold:
    """Ratio threshold controls peak acceptance."""

    def test_higher_threshold_fewer_peaks(self, known_peaks_spectrum):
        f, p, _ = known_peaks_spectrum
        base_cfg = {"smooth": "none", "baseline_window": 61}

        _, _ = dsgbr_detector(f, p, case_info={**base_cfg, "RT": 1.2})
        low_f, _ = dsgbr_detector(f, p, case_info={**base_cfg, "RT": 1.2})
        high_f, _ = dsgbr_detector(f, p, case_info={**base_cfg, "RT": 5.0})
        assert high_f.size <= low_f.size

    def test_very_high_threshold_no_peaks(self, known_peaks_spectrum):
        f, p, _ = known_peaks_spectrum
        peak_f, _ = dsgbr_detector(f, p, case_info={"RT": 100.0})
        assert peak_f.size == 0


class TestSpacing:
    """Frequency-dependent spacing rules."""

    def test_switch_frequency_affects_spacing(self):
        """With large distance_low, low-freq peaks get filtered but high-freq survive."""
        frequencies = np.linspace(0.01, 1.0, 1000)
        psd = np.ones_like(frequencies)
        # Two peaks well-separated in high-freq region
        psd[700] = 10.0
        psd[900] = 10.0
        # With distance_high=1, both high-freq peaks should survive
        cfg_high = {
            "smooth": "none",
            "ratio_threshold": 1.2,
            "baseline_window": 61,
            "distance_high": 1,
            "distance_low": 300,
            "switch_frequency": 0.01,
        }
        peak_f_high, _ = dsgbr_detector(frequencies, psd, case_info=cfg_high)
        assert peak_f_high.size >= 2

        # With switch_frequency very high (all peaks treated as low-freq),
        # the large distance_low should reduce detections
        cfg_low = {
            "smooth": "none",
            "ratio_threshold": 1.2,
            "baseline_window": 61,
            "distance_high": 1,
            "distance_low": 300,
            "switch_frequency": 10.0,
        }
        peak_f_low, _ = dsgbr_detector(frequencies, psd, case_info=cfg_low)
        assert peak_f_low.size <= peak_f_high.size


class TestMaxPeaks:
    """max_peaks limit is respected."""

    def test_respects_max_peaks_limit(self):
        frequencies = np.linspace(1e-3, 1.0, 500)
        psd = np.ones_like(frequencies)
        spike_indices = np.array([20, 60, 110, 180, 250, 330, 420])
        psd[spike_indices] = np.array([12, 11, 13, 10, 9.5, 9.7, 9.8])
        peak_f, _ = dsgbr_detector(
            frequencies,
            psd,
            case_info={"smooth": "none", "RT": 1.2, "baseline_window": 61, "max_peaks": 3},
        )
        assert peak_f.size <= 3
        assert np.all(np.diff(peak_f) > 0)

    def test_max_peaks_one(self):
        frequencies = np.linspace(1e-3, 1.0, 500)
        psd = np.ones_like(frequencies)
        psd[100] = 20.0
        psd[300] = 15.0
        peak_f, _ = dsgbr_detector(
            frequencies,
            psd,
            case_info={"smooth": "none", "RT": 1.2, "baseline_window": 61, "max_peaks": 1},
        )
        assert peak_f.size <= 1


class TestDenseComb:
    """Dense spectral comb handling."""

    def test_dense_comb_detection(self, dense_comb_spectrum):
        f, p = dense_comb_spectrum
        cfg = {"smooth": "none", "ratio_threshold": 1.5, "baseline_window": 101, "max_peaks": 15}
        peak_f, _ = dsgbr_detector(f, p, case_info=cfg)
        assert peak_f.size <= 15
        assert peak_f.size > 0


class TestSmoothingModes:
    """Different smoothing configurations."""

    def test_smoothing_none(self, known_peaks_spectrum):
        f, p, _spike_indices = known_peaks_spectrum
        peak_f, _ = dsgbr_detector(
            f, p, case_info={"smooth": "none", "RT": 1.2, "baseline_window": 61}
        )
        assert peak_f.size >= 4

    def test_smoothing_savgol(self, known_peaks_spectrum):
        f, p, _ = known_peaks_spectrum
        peak_f, _ = dsgbr_detector(
            f, p, case_info={"smooth": "savgol", "SW": 5, "RT": 1.2, "baseline_window": 61}
        )
        assert peak_f.size > 0

    def test_smoothing_on_log_vs_linear(self, known_peaks_spectrum):
        f, p, _ = known_peaks_spectrum
        base = {"smooth": "savgol", "SW": 5, "RT": 1.5, "baseline_window": 61}
        log_f, _ = dsgbr_detector(f, p, case_info={**base, "smooth_on_log": True})
        lin_f, _ = dsgbr_detector(f, p, case_info={**base, "smooth_on_log": False})
        # Both should detect peaks but may differ
        assert log_f.size > 0 or lin_f.size > 0


class TestComputeSupport:
    """compute_support_series helper."""

    def test_consistent_shapes(self):
        f = np.linspace(1e-3, 1e-2, 100)
        p = np.random.default_rng(0).uniform(0.5, 2.0, size=100)
        support = compute_support_series(f, p)
        assert support["search_series"].shape == p.shape
        assert support["baseline_series"].shape == p.shape
        assert support["ratio_series"].shape == p.shape
        assert support["rthreshold"].shape == p.shape

    def test_detector_config_in_support(self):
        f = np.linspace(1e-3, 1.0, 100)
        p = np.ones_like(f)
        support = compute_support_series(f, p, case_info={"RT": 2.5})
        assert support["detector_config"]["ratio_threshold"] == 2.5


# ---------------------------------------------------------------------------
# Hypothesis property-based tests
# ---------------------------------------------------------------------------

# Strategy for valid PSD arrays: positive floats, reasonable size
_psd_strategy = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=10, max_value=500),
    elements=st.floats(min_value=1e-10, max_value=1e6, allow_nan=False, allow_infinity=False),
)


class TestHypothesis:
    """Property-based tests using Hypothesis."""

    @given(psd=_psd_strategy)
    @settings(max_examples=50, deadline=5000)
    def test_detector_never_crashes(self, psd):
        """Detector should never raise on valid positive inputs."""
        f = np.linspace(1e-3, 1.0, len(psd))
        peak_f, peak_h = dsgbr_detector(f, psd)
        assert isinstance(peak_f, np.ndarray)
        assert isinstance(peak_h, np.ndarray)
        assert peak_f.shape == peak_h.shape

    @given(psd=_psd_strategy)
    @settings(max_examples=50, deadline=5000)
    def test_max_peaks_always_honored(self, psd):
        """Output should never exceed max_peaks."""
        f = np.linspace(1e-3, 1.0, len(psd))
        max_peaks = 5
        peak_f, _ = dsgbr_detector(f, psd, case_info={"max_peaks": max_peaks})
        assert peak_f.size <= max_peaks

    @given(psd=_psd_strategy)
    @settings(max_examples=50, deadline=5000)
    def test_output_always_sorted(self, psd):
        """peak_f should always be sorted ascending."""
        f = np.linspace(1e-3, 1.0, len(psd))
        peak_f, _ = dsgbr_detector(f, psd)
        if peak_f.size > 1:
            assert np.all(np.diff(peak_f) > 0)
