"""Apply calibration: raw scene DN -> calibrated radiance.

Inverse of the forward model:
    radiance = (DN - dark_level) / (response * scale)

scale (DN per radiance unit) comes from flat frames at KNOWN radiance, like a
lab integrating sphere. Never reads the sensor config or the truth file.
"""
from typing import Sequence

import numpy as np


def estimate_radiometric_scale(
    radiance_levels: Sequence[float],
    flats: Sequence[np.ndarray],
    dark_level_dn: np.ndarray,
    full_scale_dn: float = 4095.0,
    max_fraction: float = 0.85,
) -> float:
    """Least-squares slope (through the origin) of median signal DN vs known radiance."""
    xs, ys = [], []
    for lev, stack in zip(radiance_levels, flats):
        stack = np.asarray(stack)
        if stack.ndim != 3:
            raise ValueError("each flat entry must be an (N, H, W) stack")
        raw_mean = stack.mean(axis=0, dtype=np.float64)       # float64: no uint16 wrap
        if np.median(raw_mean) > max_fraction * full_scale_dn:
            continue                                          # near saturation: skip
        xs.append(float(lev))
        ys.append(float(np.median(raw_mean - dark_level_dn)))

    if len(xs) < 2:
        raise ValueError(f"need at least 2 usable flat levels, got {len(xs)}")

    x = np.array(xs)
    y = np.array(ys)
    scale = float(np.sum(x * y) / np.sum(x * x))
    if scale <= 0:
        raise ValueError("non-positive radiometric scale")
    return scale


def apply_calibration(
    dn: np.ndarray,
    dark_level_dn: np.ndarray,
    response: np.ndarray,
    scale: float,
    min_response: float = 0.5,
) -> np.ndarray:
    """Convert raw DN to calibrated radiance. Pixels with response < min_response -> NaN."""
    dn = np.asarray(dn)
    if not (dn.shape == dark_level_dn.shape == response.shape):
        raise ValueError(
            f"shape mismatch: dn {dn.shape}, dark {dark_level_dn.shape}, "
            f"response {response.shape}"
        )
    if scale <= 0:
        raise ValueError("scale must be positive")

    signal = dn.astype(np.float64) - dark_level_dn            # float64 BEFORE subtracting
    radiance = np.full(dn.shape, np.nan, dtype=np.float64)
    ok = response >= min_response
    radiance[ok] = signal[ok] / (response[ok] * scale)
    return radiance