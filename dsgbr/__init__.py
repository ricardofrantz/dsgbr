"""DSGBR package public API."""

from .DSGBR import (
    DSGBR_PARAM_ALIASES,
    DetectionConfig,
    compute_support_series,
    detect_peaks_case_adaptive,
    dsgbr_detector,
    find_nearest_frequency,
    select_peaks_by_frequency_bands,
)

DSGBRDetectionConfig = DetectionConfig

__all__ = [
    "DSGBR_PARAM_ALIASES",
    "DSGBRDetectionConfig",
    "DetectionConfig",
    "compute_support_series",
    "detect_peaks_case_adaptive",
    "dsgbr_detector",
    "find_nearest_frequency",
    "select_peaks_by_frequency_bands",
]

__version__ = "0.1.0"
