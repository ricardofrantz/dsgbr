"""Detection configuration for the DSGBR peak detector.

This module defines :class:`DetectionConfig`, a frozen dataclass that holds
all tunable parameters for the five-stage DSGBR detection pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DetectionConfig:
    """Configuration for the DSGBR detector.

    Parameters are organized by function:

    - **SEARCH series** -- Peak detection smoothing parameters.
    - **BASELINE series** -- Baseline estimation parameters.
    - **Detection** -- Peak acceptance criteria.
    - **Spacing** -- Frequency-dependent peak separation rules.
    - **ULF guardrail** -- Ultra-low-frequency filtering.
    - **Selection** -- Final peak down-selection strategy.

    Parameters
    ----------
    smooth : str
        Smoothing algorithm for SEARCH series construction.
    smooth_window : int
        Window size for SEARCH series Savitzky-Golay smoothing (odd integer >= 3).
        Commonly abbreviated as **SW**.
    smooth_polyorder : int
        Polynomial order for Savitzky-Golay filtering (must be < ``smooth_window``).
    smooth_on_log : bool
        Apply smoothing to log10(PSD) instead of linear PSD.
    baseline_window : int or None
        Fixed window size for baseline smoothing.  Overrides
        ``baseline_window_frac`` when set.
    baseline_window_frac : float
        Baseline window as fraction of data length (e.g. 0.001 = 0.1%).
        Commonly abbreviated as **BWF**.
    baseline_on_log : bool
        Apply baseline smoothing to log10(SEARCH) instead of linear SEARCH.
    ratio_threshold : float
        Minimum SEARCH/BASELINE ratio for peak acceptance (>= 1.0).
        Commonly abbreviated as **RT**.
    switch_frequency : float
        Frequency threshold: f >= switch_frequency uses ``distance_high``,
        otherwise ``distance_low``.
    distance_low : int
        Minimum bin separation for peaks below ``switch_frequency``.
    distance_high : int
        Minimum bin separation for peaks at or above ``switch_frequency``.
    ulf_fmax : float
        Maximum frequency considered ultra-low-frequency (ULF) for special filtering.
    ulf_min_q : float
        Minimum Q-factor (f_center / FWHM) for ULF peak retention.
    ulf_max_points : int
        Maximum number of ULF peaks to retain (ranked by amplitude).
    max_peaks : int
        Maximum number of peaks to return from detection.
    band_strategy : str
        Strategy for allocating peaks across frequency bands:
        ``'proportional'`` or ``'equal'``.
    n_bands : int
        Number of logarithmic frequency bands for peak allocation.

    Examples
    --------
    >>> cfg = DetectionConfig(ratio_threshold=2.0, smooth_window=5)
    >>> cfg.ratio_threshold
    2.0

    >>> cfg = DetectionConfig.from_case_info({"RT": "2.2", "SW": "11"})
    >>> cfg.smooth_window
    11
    """

    # ==================== SEARCH SERIES PARAMETERS ====================
    smooth: str = "savgol"
    """Smoothing algorithm for SEARCH series construction."""

    smooth_window: int = 3
    """Window size for SEARCH series smoothing (odd integer >= 3)."""

    smooth_polyorder: int = 2
    """Polynomial order for Savitzky-Golay filtering (must be < smooth_window)."""

    smooth_on_log: bool = True
    """Apply smoothing to log10(PSD) instead of linear PSD."""

    # ==================== BASELINE SERIES PARAMETERS ====================
    baseline_window: int | None = None
    """Fixed window size for baseline smoothing (overrides baseline_window_frac)."""

    baseline_window_frac: float = 0.001
    """Baseline window as fraction of data length."""

    baseline_on_log: bool = True
    """Apply baseline smoothing to log10(SEARCH) instead of linear SEARCH."""

    # ==================== DETECTION PARAMETERS ====================
    ratio_threshold: float = 1.8
    """Minimum SEARCH/BASELINE ratio for peak acceptance (>= 1.0)."""

    # ==================== SPACING PARAMETERS ====================
    switch_frequency: float = 2e-2
    """Frequency threshold for spacing rule selection."""

    distance_low: int = 2
    """Minimum bin separation for peaks below switch_frequency."""

    distance_high: int = 1
    """Minimum bin separation for peaks at or above switch_frequency."""

    # ==================== ULF GUARDRAIL PARAMETERS ====================
    ulf_fmax: float = 1e-3
    """Maximum frequency considered ultra-low-frequency (ULF)."""

    ulf_min_q: float = 9.0
    """Minimum Q-factor (f_center/FWHM) for ULF peak retention."""

    ulf_max_points: int = 5
    """Maximum number of ULF peaks to retain (ranked by amplitude)."""

    # ==================== SELECTION PARAMETERS ====================
    max_peaks: int = 25
    """Maximum number of peaks to return from detection."""

    band_strategy: str = "proportional"
    """Strategy for allocating peaks across frequency bands."""

    n_bands: int = 10
    """Number of logarithmic frequency bands for peak allocation."""

    def __post_init__(self) -> None:
        """Validate parameter constraints after initialization.

        Raises
        ------
        ValueError
            If any parameter is out of its valid range.
        """
        if self.ratio_threshold < 1.0:
            msg = f"ratio_threshold must be >= 1.0, got {self.ratio_threshold}"
            raise ValueError(msg)
        if self.smooth_window < 3:
            msg = f"smooth_window must be >= 3, got {self.smooth_window}"
            raise ValueError(msg)
        if self.smooth_window % 2 == 0:
            msg = f"smooth_window must be odd, got {self.smooth_window}"
            raise ValueError(msg)
        if self.smooth_polyorder >= self.smooth_window:
            msg = (
                f"smooth_polyorder ({self.smooth_polyorder}) must be "
                f"< smooth_window ({self.smooth_window})"
            )
            raise ValueError(msg)
        if self.max_peaks < 1:
            msg = f"max_peaks must be >= 1, got {self.max_peaks}"
            raise ValueError(msg)
        if self.n_bands < 1:
            msg = f"n_bands must be >= 1, got {self.n_bands}"
            raise ValueError(msg)

    @classmethod
    def from_case_info(cls, case_info: Any | None) -> DetectionConfig:
        """Construct a :class:`DetectionConfig` from a dict with alias support.

        Parameters
        ----------
        case_info : dict or None
            Dictionary of parameter values, optionally using short aliases
            (``RT``, ``SW``, ``BWF``, ``DH``, ``DL``, ``SF``, ``MP``).

        Returns
        -------
        DetectionConfig
            Validated configuration instance.

        Examples
        --------
        >>> cfg = DetectionConfig.from_case_info({"RT": "2.0", "SW": "7"})
        >>> cfg.ratio_threshold
        2.0
        """
        if not isinstance(case_info, dict) or not case_info:
            return cls()

        def _convert(value: Any, dtype: type) -> Any:
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
        """Serialize all fields to a plain dictionary.

        Returns
        -------
        dict
            All configuration fields as a JSON-serializable dictionary.
        """
        return asdict(self)
