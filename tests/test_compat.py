"""Tests for backward-compatibility layer."""

from __future__ import annotations

import warnings

import numpy as np

from dsgbr import DSGBR_PARAM_ALIASES, DetectionConfig, DSGBRDetectionConfig


class TestAliases:
    """DSGBR_PARAM_ALIASES and type aliases."""

    def test_param_aliases_keys(self):
        expected_keys = {"RT", "SW", "BWF", "DL", "DH", "SF", "MP"}
        assert set(DSGBR_PARAM_ALIASES.keys()) == expected_keys

    def test_param_aliases_values(self):
        assert DSGBR_PARAM_ALIASES["RT"] == "ratio_threshold"
        assert DSGBR_PARAM_ALIASES["SW"] == "smooth_window"
        assert DSGBR_PARAM_ALIASES["BWF"] == "baseline_window_frac"

    def test_dsgbr_detection_config_is_detection_config(self):
        assert DSGBRDetectionConfig is DetectionConfig
        cfg = DSGBRDetectionConfig(ratio_threshold=2.0)
        assert isinstance(cfg, DetectionConfig)


class TestDetectPeaksCaseAdaptive:
    """Deprecated detect_peaks_case_adaptive wrapper."""

    def test_deprecation_warning(self):
        from dsgbr import detect_peaks_case_adaptive

        f = np.linspace(1e-3, 1.0, 100)
        p = np.ones(100)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            detect_peaks_case_adaptive(f, p)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "dsgbr_detector" in str(w[0].message)

    def test_delegates_to_dsgbr_detector(self):
        from dsgbr import detect_peaks_case_adaptive, dsgbr_detector

        f = np.linspace(1e-3, 1.0, 500)
        p = np.ones(500)
        p[200] = 10.0

        cfg = {"smooth": "none", "RT": 1.2, "baseline_window": 61}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            compat_f, compat_h = detect_peaks_case_adaptive(f, p, case_info=cfg)

        direct_f, direct_h = dsgbr_detector(f, p, case_info=cfg)
        np.testing.assert_array_equal(compat_f, direct_f)
        np.testing.assert_array_equal(compat_h, direct_h)


class TestDSGBRModuleImport:
    """Backward-compatible import path via DSGBR.py shim."""

    def test_import_from_dsgbr_DSGBR(self):
        from dsgbr.DSGBR import dsgbr_detector

        assert callable(dsgbr_detector)

    def test_all_public_names_available(self):
        from dsgbr import DSGBR

        expected = {
            "DSGBR_PARAM_ALIASES",
            "DSGBRDetectionConfig",
            "DetectionConfig",
            "compute_support_series",
            "detect_peaks_case_adaptive",
            "dsgbr_detector",
            "find_nearest_frequency",
            "select_peaks_by_frequency_bands",
        }
        assert expected.issubset(set(dir(DSGBR)))

    def test_functional_through_shim(self):
        from dsgbr.DSGBR import DetectionConfig, dsgbr_detector

        cfg = DetectionConfig.from_case_info({"RT": 2.0})
        assert cfg.ratio_threshold == 2.0

        f = np.linspace(1e-3, 1.0, 100)
        p = np.ones(100)
        peak_f, _ = dsgbr_detector(f, p)
        assert isinstance(peak_f, np.ndarray)
