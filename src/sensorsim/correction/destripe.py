"""Scene-based column destriping (fallback when flat-field calibration is missing or stale).

Column profile = median over rows of each column (ignoring invalid pixels).
Stripe estimate = profile minus its running median over neighbouring columns.
Real vertical structure wider than ~window/2 survives; 1-column features do not.
"""
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def destripe(
    image: np.ndarray,
    mode: str = "offset",
    window: int = 31,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Remove column stripes. Returns a float64 copy; NaN pixels stay NaN.

    mode: "offset" subtracts the stripe estimate, "gain" divides by it.
    valid: optional (H, W) bool, False = ignore this pixel when estimating stripes.
    """
    img = np.asarray(image, dtype=np.float64)              # float64: no uint16 wrap
    if img.ndim != 2:
        raise ValueError(f"image must be 2-D, got shape {img.shape}")
    if mode not in ("offset", "gain"):
        raise ValueError(f"mode must be 'offset' or 'gain', got {mode!r}")
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer >= 3")
    if window > img.shape[1]:
        raise ValueError("window must not exceed the image width")

    use = np.isfinite(img)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != img.shape:
            raise ValueError(f"shape mismatch: image {img.shape}, valid {valid.shape}")
        use &= valid
    work = np.where(use, img, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)    # all-NaN columns are handled below
        profile = np.nanmedian(work, axis=0)
        padded = np.pad(profile, window // 2, mode="reflect")
        smooth = np.nanmedian(sliding_window_view(padded, window), axis=1)

    if mode == "offset":
        est = profile - smooth
        est = np.where(np.isfinite(est), est, 0.0)
        return img - est[None, :]

    good = np.isfinite(profile) & np.isfinite(smooth) & (profile > 0) & (smooth > 0)
    factor = np.where(good, smooth / np.where(good, profile, 1.0), 1.0)
    return img * factor[None, :]