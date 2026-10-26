"""Defect detection from calibration frames only (dark + flat stacks).

Rules:
  constant          : DN identical in every frame of every stack (dead / stuck)
  hot               : dark level far above its column's typical dark level
  response_outlier  : flat-field response far from its column's typical response

Never reads the sensor config or the truth file.
"""
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class DefectMasks:
    constant: np.ndarray          # (H, W) bool: dead or stuck
    hot: np.ndarray               # (H, W) bool
    response_outlier: np.ndarray  # (H, W) bool

    @property
    def combined(self) -> np.ndarray:
        return self.constant | self.hot | self.response_outlier


def _robust_sigma(x: np.ndarray) -> float:
    """Sigma from the median absolute deviation (ignores outliers)."""
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def detect_defects(
    darks: np.ndarray,
    flats: Sequence[np.ndarray],
    dark_level_dn: np.ndarray,
    response: np.ndarray,
    k: float = 6.0,
    min_sigma_dn: float = 0.1,
    min_sigma_resp: float = 1e-4,
) -> DefectMasks:
    """darks: (N, H, W). flats: list of (N, H, W). dark_level_dn, response: (H, W)."""
    darks = np.asarray(darks)
    if darks.ndim != 3:
        raise ValueError("darks must be an (N, H, W) stack")
    shape = darks.shape[1:]
    if dark_level_dn.shape != shape or response.shape != shape:
        raise ValueError(
            f"shape mismatch: darks {shape}, dark_level {dark_level_dn.shape}, "
            f"response {response.shape}"
        )
    if len(flats) == 0:
        raise ValueError("need at least one flat stack")

    # 1) constant pixels: max == min across frames, in EVERY stack
    constant = darks.max(axis=0) == darks.min(axis=0)
    for stack in flats:
        stack = np.asarray(stack)
        if stack.ndim != 3 or stack.shape[1:] != shape:
            raise ValueError("every flat stack must be (N, H, W) matching the darks")
        constant &= stack.max(axis=0) == stack.min(axis=0)

    # 2) hot pixels: remove each column's typical dark level (column offsets), then outliers
    resid = dark_level_dn - np.median(dark_level_dn, axis=0)[None, :]
    sigma_dn = max(_robust_sigma(resid), min_sigma_dn)
    hot = resid > k * sigma_dn

    # 3) response outliers: remove each column's typical response (column gain), then outliers
    r = response / np.median(response, axis=0)[None, :]
    sigma_r = max(_robust_sigma(r), min_sigma_resp)
    response_outlier = np.abs(r - np.median(r)) > k * sigma_r

    return DefectMasks(constant=constant, hot=hot, response_outlier=response_outlier)