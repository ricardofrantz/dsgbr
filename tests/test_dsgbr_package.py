import numpy as np

from dsgbr import (
    DSGBRDetectionConfig,
    DetectionConfig,
    dsgbr_detector,
    select_peaks_by_frequency_bands,
)


def test_detection_config_aliases_and_normalization():
    cfg = DetectionConfig.from_case_info(
        {
            "RT": "2.2",
            "SW": "11",
            "BWF": "0.002",
            "DH": "5",
            "DL": "4",
            "SF": "0.01",
            "MP": "7",
            "baseline_window": "-1",
        }
    )

    assert isinstance(cfg, DSGBRDetectionConfig)
    assert cfg.ratio_threshold == 2.2
    assert cfg.smooth_window == 11
    assert cfg.baseline_window_frac == 0.002
    assert cfg.distance_high == 5
    assert cfg.distance_low == 4
    assert cfg.switch_frequency == 0.01
    assert cfg.max_peaks == 7
    assert cfg.baseline_window is None


def test_detector_detects_known_spectral_peaks_and_sorts():
    frequencies = np.linspace(1e-3, 1.0, 2048)
    psd = np.ones_like(frequencies)

    known_indices = np.array([110, 412, 1120, 1710], dtype=int)
    psd[known_indices] = np.array([12.0, 16.0, 9.5, 11.5])

    detector_cfg = {
        "smooth": "none",
        "ratio_threshold": 1.2,
        "baseline_window": 61,
        "max_peaks": 10,
    }
    peak_f, _ = dsgbr_detector(frequencies, psd, case_info=detector_cfg)
    support = dsgbr_detector(frequencies, psd, case_info=detector_cfg, return_support=True)[2]

    assert peak_f.size >= 4
    assert np.all(np.diff(peak_f) > 0)
    assert set(np.array([frequencies[idx] for idx in known_indices]).round(12)).issubset(
        set(np.round(peak_f, 12))
    )
    assert support["detector_config"]["ratio_threshold"] == 1.2
    assert support["accepted_indices"].dtype == int


def test_detector_respects_max_detected_peak_limit():
    frequencies = np.linspace(1e-3, 1.0, 500)
    psd = np.ones_like(frequencies)

    spike_indices = np.array([20, 60, 110, 180, 250, 330, 420])
    psd[spike_indices] = np.array([12, 11, 13, 10, 9.5, 9.7, 9.8])

    peak_f, _ = dsgbr_detector(
        frequencies,
        psd,
        case_info={"smooth": "none", "ratio_threshold": 1.2, "baseline_window": 61, "max_peaks": 3},
    )

    assert peak_f.size <= 3
    assert np.all(np.diff(peak_f) > 0)


def test_band_selection_is_deterministic_and_ordered():
    freqs = np.array([1e-4, 3e-4, 1e-3, 2.3e-3, 8e-3, 1.8e-2, 3.4e-2, 6e-2, 0.11, 0.28, 0.7])
    heights = np.array([1.0, 3.0, 0.8, 2.5, 0.7, 4.1, 2.7, 1.2, 3.3, 5.1, 2.2])

    selected_f, selected_h = select_peaks_by_frequency_bands(
        freqs,
        heights,
        max_peaks=4,
        strategy="equal",
        n_bands=3,
    )

    assert selected_f.size <= 4
    assert np.all(np.diff(selected_f) > 0)
    assert selected_f.size == len(selected_h)
