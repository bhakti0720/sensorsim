"""Dark-frame calibration: estimate dark level and read noise from frames only.

Never reads the sensor config or the truth file.

Units: everything here is in DN. Read noise in electrons needs the gain K,
which comes from photon_transfer.py.

Limit: with one integration time, offset and dark signal cannot be separated,
so we estimate their sum (dark_level_dn).
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DarkCalibration:
    dark_level_dn: np.ndarray    # (H, W) per-pixel mean of dark frames (offset + dark signal)
    read_noise_dn: np.ndarray    # (H, W) per-pixel temporal std of dark frames
    n_frames: int

    @property
    def read_noise_dn_median(self) -> float:
        # median is robust to hot/dead/stuck pixels
        return float(np.median(self.read_noise_dn))

    @property
    def dark_level_dn_median(self) -> float:
        return float(np.median(self.dark_level_dn))


def estimate_dark(frames: np.ndarray) -> DarkCalibration:
    """Estimate dark level and read noise from a stack of dark frames.

    frames: (N, H, W), any numeric dtype (uint16 is fine), N >= 2.
    """
    frames = np.asarray(frames)
    if frames.ndim != 3:
        raise ValueError(f"frames must be (N, H, W), got shape {frames.shape}")
    n = frames.shape[0]
    if n < 2:
        raise ValueError("need at least 2 dark frames to estimate noise")

    # dtype=float64 inside the reduction avoids uint16 wrap-around.
    dark_level = frames.mean(axis=0, dtype=np.float64)
    read_noise = frames.std(axis=0, ddof=1, dtype=np.float64)

    return DarkCalibration(dark_level_dn=dark_level, read_noise_dn=read_noise, n_frames=n)