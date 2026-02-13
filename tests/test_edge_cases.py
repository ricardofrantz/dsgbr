"""Edge case tests for extreme and degenerate inputs."""

from __future__ import annotations

import numpy as np

from dsgbr import dsgbr_detector


class TestTinyInputs:
    """Very short arrays."""

    def test_single_point(self):
        f = np.array([0.5])
        p = np.array([1.0])
        peak_f, peak_h = dsgbr_detector(f, p)
        assert peak_f.size == 0
        assert peak_h.size == 0

    def test_two_points(self):
        f = np.array([0.1, 0.2])
        p = np.array([1.0, 5.0])
        peak_f, peak_h = dsgbr_detector(f, p)
        assert isinstance(peak_f, np.ndarray)
        assert isinstance(peak_h, np.ndarray)

    def test_three_points_with_peak(self):
        f = np.array([0.1, 0.2, 0.3])
        p = np.array([1.0, 10.0, 1.0])
        # Smoothing window = 3 equals data length, so smoothing may not apply
        peak_f, _ = dsgbr_detector(f, p, case_info={"smooth": "none", "RT": 1.2})
        assert isinstance(peak_f, np.ndarray)


class TestExtremeValues:
    """NaN, Inf, zeros, and large values."""

    def test_all_zeros(self):
        f = np.linspace(1e-3, 1.0, 100)
        p = np.zeros(100)
        peak_f, _ = dsgbr_detector(f, p)
        assert isinstance(peak_f, np.ndarray)

    def test_very_large_values(self):
        f = np.linspace(1e-3, 1.0, 200)
        p = np.ones(200) * 1e15
        p[100] = 1e20
        peak_f, _ = dsgbr_detector(
            f, p, case_info={"smooth": "none", "RT": 1.2, "baseline_window": 31}
        )
        assert isinstance(peak_f, np.ndarray)

    def test_near_epsilon_psd(self):
        f = np.linspace(1e-3, 1.0, 200)
        p = np.ones(200) * 1e-300
        p[100] = 1e-290
        peak_f, _ = dsgbr_detector(f, p)
        assert isinstance(peak_f, np.ndarray)

    def test_nan_in_psd(self):
        f = np.linspace(1e-3, 1.0, 200)
        p = np.ones(200)
        p[50] = np.nan
        # Should not crash, though results may be unreliable
        peak_f, _ = dsgbr_detector(f, p)
        assert isinstance(peak_f, np.ndarray)

    def test_inf_in_psd(self):
        f = np.linspace(1e-3, 1.0, 200)
        p = np.ones(200)
        p[50] = np.inf
        peak_f, _ = dsgbr_detector(f, p)
        assert isinstance(peak_f, np.ndarray)


class TestNoSmoothing:
    """Smoothing disabled edge cases."""

    def test_smooth_none_string(self):
        f = np.linspace(1e-3, 1.0, 100)
        p = np.ones(100)
        p[50] = 5.0
        peak_f, _ = dsgbr_detector(f, p, case_info={"smooth": "none", "RT": 1.2})
        assert isinstance(peak_f, np.ndarray)

    def test_smooth_empty_string(self):
        f = np.linspace(1e-3, 1.0, 100)
        p = np.ones(100)
        p[50] = 5.0
        peak_f, _ = dsgbr_detector(f, p, case_info={"smooth": ""})
        assert isinstance(peak_f, np.ndarray)
