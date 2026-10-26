"""Photon transfer curve: estimate gain K (e-/DN) from flat frames only.

Model in DN:  var_dn = mean_dn / K + intercept
    slope = 1/K ; intercept = read-noise variance + quantisation (1/12 DN^2)

Never reads the sensor config or the truth file.
"""
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PTCResult:
    gain_e_per_dn: float        # K
    gain_stderr: float          # 1-sigma uncertainty on K
    read_noise_e: float         # from the intercept, quantisation removed
    r2: float                   # quality of the straight-line fit
    n_levels_used: int
    mean_dn: np.ndarray         # (n_levels_used,) signal above dark level
    var_dn: np.ndarray          # (n_levels_used,) temporal variance


def _robust_pixel_mask(mean_img, var_img, k=5.0):
    """Keep pixels that look normal: drops dead, stuck and hot pixels."""
    med_m = np.median(mean_img)
    mad_m = 1.4826 * np.median(np.abs(mean_img - med_m))
    med_v = np.median(var_img)
    ok_mean = np.abs(mean_img - med_m) <= max(k * mad_m, 1e-6)
    ok_var = var_img <= k * med_v
    return ok_mean & ok_var


def estimate_gain(
    levels: Sequence[np.ndarray],
    dark_level_dn: np.ndarray,
    full_scale_dn: float = 4095.0,
    max_fraction: float = 0.85,
) -> PTCResult:
    """levels: list of flat-frame stacks, each (N, H, W), one stack per light level."""
    means, variances = [], []
    for stack in levels:
        stack = np.asarray(stack)
        if stack.ndim != 3 or stack.shape[0] < 2:
            raise ValueError("each level must be an (N, H, W) stack with N >= 2")

        # float64 inside the reduction avoids uint16 wrap-around
        raw_mean = stack.mean(axis=0, dtype=np.float64)
        if np.median(raw_mean) > max_fraction * full_scale_dn:
            continue  # too close to saturation: variance is no longer shot noise

        mean_img = raw_mean - dark_level_dn
        var_img = stack.var(axis=0, ddof=1, dtype=np.float64)
        keep = _robust_pixel_mask(mean_img, var_img)
        means.append(mean_img[keep].mean())
        variances.append(var_img[keep].mean())

    if len(means) < 3:
        raise ValueError(f"need at least 3 usable light levels, got {len(means)}")

    m = np.array(means)
    v = np.array(variances)
    (slope, intercept), cov = np.polyfit(m, v, 1, cov=True)
    if slope <= 0:
        raise ValueError("non-positive PTC slope: frames are not shot-noise limited")

    gain = 1.0 / slope
    gain_err = np.sqrt(cov[0, 0]) / slope**2
    read_var_dn = max(intercept - 1.0 / 12.0, 0.0)   # remove rounding noise
    pred = slope * m + intercept
    r2 = 1.0 - np.sum((v - pred) ** 2) / np.sum((v - v.mean()) ** 2)

    return PTCResult(
        gain_e_per_dn=float(gain),
        gain_stderr=float(gain_err),
        read_noise_e=float(gain * np.sqrt(read_var_dn)),
        r2=float(r2),
        n_levels_used=len(m),
        mean_dn=m,
        var_dn=v,
    )