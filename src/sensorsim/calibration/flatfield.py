"""Flat-field calibration: per-pixel relative response from flat frames only.

Model per pixel:  signal_dn(level k) = a_xy * m_k
where m_k is the median signal over all pixels at level k, and a_xy is the
pixel's relative response, normalised to median 1.
a_xy = (1 + PRNU) * (1 + column gain) in the forward model.

Never reads the sensor config or the truth file.
"""
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class FlatFieldCalibration:
    response: np.ndarray       # (H, W) total relative response, median 1
    column_gain: np.ndarray    # (W,) per-column part, median 1
    prnu: np.ndarray           # (H, W) pixel-only part = response / column_gain
    n_levels_used: int


def estimate_flatfield(
    levels: Sequence[np.ndarray],
    dark_level_dn: np.ndarray,
    full_scale_dn: float = 4095.0,
    max_fraction: float = 0.85,
) -> FlatFieldCalibration:
    """levels: list of flat-frame stacks, each (N, H, W). dark_level_dn: (H, W)."""
    num = np.zeros(dark_level_dn.shape, dtype=np.float64)
    den = 0.0
    used = 0

    for stack in levels:
        stack = np.asarray(stack)
        if stack.ndim != 3:
            raise ValueError("each level must be an (N, H, W) stack")

        raw_mean = stack.mean(axis=0, dtype=np.float64)   # float64: no uint16 wrap
        if np.median(raw_mean) > max_fraction * full_scale_dn:
            continue                                       # near saturation: skip

        s = raw_mean - dark_level_dn                       # signal above dark, DN
        m = np.median(s)
        if m <= 0:
            continue
        num += s * m
        den += m * m
        used += 1

    if used < 2:
        raise ValueError(f"need at least 2 usable light levels, got {used}")

    a = num / den                                          # least squares through origin
    response = a / np.median(a)

    column_gain = np.median(response, axis=0)
    column_gain = column_gain / np.median(column_gain)
    prnu = response / column_gain[None, :]

    return FlatFieldCalibration(
        response=response, column_gain=column_gain, prnu=prnu, n_levels_used=used,
    )