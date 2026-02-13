"""Tests for DetectionConfig construction, validation, and serialization."""

from __future__ import annotations

import pytest

from dsgbr import DetectionConfig, DSGBRDetectionConfig


class TestDefaults:
    """Default construction and basic properties."""

    def test_default_construction(self):
        cfg = DetectionConfig()
        assert cfg.ratio_threshold == 1.8
        assert cfg.smooth_window == 3
        assert cfg.max_peaks == 25
        assert cfg.n_bands == 10

    def test_frozen_immutability(self):
        cfg = DetectionConfig()
        with pytest.raises(AttributeError):
            cfg.ratio_threshold = 2.0  # type: ignore[misc]

    def test_dsgbr_detection_config_alias(self):
        assert DSGBRDetectionConfig is DetectionConfig


class TestValidation:
    """__post_init__ validation rules."""

    def test_ratio_threshold_below_one(self):
        with pytest.raises(ValueError, match=r"ratio_threshold must be >= 1\.0"):
            DetectionConfig(ratio_threshold=0.5)

    def test_ratio_threshold_exactly_one(self):
        cfg = DetectionConfig(ratio_threshold=1.0)
        assert cfg.ratio_threshold == 1.0

    def test_smooth_window_too_small(self):
        with pytest.raises(ValueError, match="smooth_window must be >= 3"):
            DetectionConfig(smooth_window=1)

    def test_smooth_window_even(self):
        with pytest.raises(ValueError, match="smooth_window must be odd"):
            DetectionConfig(smooth_window=4)

    def test_polyorder_exceeds_window(self):
        with pytest.raises(ValueError, match=r"smooth_polyorder.*must be.*< smooth_window"):
            DetectionConfig(smooth_window=3, smooth_polyorder=3)

    def test_max_peaks_zero(self):
        with pytest.raises(ValueError, match="max_peaks must be >= 1"):
            DetectionConfig(max_peaks=0)

    def test_n_bands_zero(self):
        with pytest.raises(ValueError, match="n_bands must be >= 1"):
            DetectionConfig(n_bands=0)


class TestFromCaseInfo:
    """from_case_info alias resolution and edge cases."""

    def test_short_aliases(self):
        cfg = DetectionConfig.from_case_info(
            {
                "RT": "2.2",
                "SW": "11",
                "BWF": "0.002",
                "DH": "5",
                "DL": "4",
                "SF": "0.01",
                "MP": "7",
            }
        )
        assert cfg.ratio_threshold == 2.2
        assert cfg.smooth_window == 11
        assert cfg.baseline_window_frac == 0.002
        assert cfg.distance_high == 5
        assert cfg.distance_low == 4
        assert cfg.switch_frequency == 0.01
        assert cfg.max_peaks == 7

    def test_long_names(self):
        cfg = DetectionConfig.from_case_info(
            {
                "ratio_threshold": 2.5,
                "smooth_window": 7,
            }
        )
        assert cfg.ratio_threshold == 2.5
        assert cfg.smooth_window == 7

    def test_none_returns_defaults(self):
        cfg = DetectionConfig.from_case_info(None)
        assert cfg == DetectionConfig()

    def test_empty_dict_returns_defaults(self):
        cfg = DetectionConfig.from_case_info({})
        assert cfg == DetectionConfig()

    def test_non_dict_returns_defaults(self):
        cfg = DetectionConfig.from_case_info("not a dict")
        assert cfg == DetectionConfig()

    def test_negative_baseline_window_ignored(self):
        cfg = DetectionConfig.from_case_info({"baseline_window": "-1"})
        assert cfg.baseline_window is None

    def test_invalid_value_skipped(self):
        cfg = DetectionConfig.from_case_info({"RT": "not_a_number"})
        assert cfg.ratio_threshold == 1.8  # default

    def test_legacy_prominence_alias(self):
        cfg = DetectionConfig.from_case_info({"prominence_window": 51})
        assert cfg.baseline_window == 51

    def test_legacy_two_pass_alias(self):
        cfg = DetectionConfig.from_case_info({"two_pass_fmin": 0.05})
        assert cfg.switch_frequency == 0.05


class TestToMetadata:
    """Serialization round-trip."""

    def test_round_trip(self):
        cfg = DetectionConfig(ratio_threshold=2.0, smooth_window=7)
        meta = cfg.to_metadata()
        assert meta["ratio_threshold"] == 2.0
        assert meta["smooth_window"] == 7
        assert isinstance(meta, dict)

    def test_all_fields_present(self):
        cfg = DetectionConfig()
        meta = cfg.to_metadata()
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(cfg)}
        assert set(meta.keys()) == field_names
