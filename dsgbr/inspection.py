"""Minimal inspection helpers for DSGBR prototyping.

The DSGBR sandbox keeps plotting utilities intentionally tiny so the package
remains focused on detection.  Only ``_psd_limits_within_band`` is exported; it
mirrors the helper in ``core.inspection`` but lives here to avoid a dependency on
the full preprocessing pipeline.
"""

from __future__ import annotations

import numpy as np


def _psd_limits_within_band(
    st: np.ndarray,
    psd: np.ndarray,
    *,
    x_min: float,
    x_max: float,
) -> tuple[float, float]:
    """Return PSD-driven y-limits restricted to the requested frequency band."""

    mask = (st >= x_min) & (st <= x_max) & (psd > 0) & np.isfinite(psd)
    if mask.any():
        y_min = float(np.min(psd[mask]))
        y_max = float(np.max(psd[mask]))
    else:
        y_min, y_max = 1e-16, 1.0

    y_min = max(y_min, 1e-16)
    y_max = max(y_max * 1.10, y_min * 1.2)
    return y_min, y_max


__all__ = ["_psd_limits_within_band"]
