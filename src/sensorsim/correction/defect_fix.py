"""Defect interpolation: replace flagged pixels with the mean of valid 8-neighbours.

Clusters with no valid neighbour are filled over several passes, outside in.
Non-finite pixels (NaN) are always treated as invalid.
Vectorised NumPy; this is the baseline for the C++ port.
"""
import numpy as np


def fix_defects(image: np.ndarray, mask: np.ndarray, max_iter: int = 50) -> np.ndarray:
    """Return a float64 copy of image with masked (and non-finite) pixels interpolated.

    image: (H, W). mask: (H, W) bool, True = defective.
    Valid pixels are returned exactly unchanged.
    """
    img = np.asarray(image, dtype=np.float64)             # float64: no uint16 wrap
    mask = np.asarray(mask, dtype=bool)
    if img.ndim != 2:
        raise ValueError(f"image must be 2-D, got shape {img.shape}")
    if mask.shape != img.shape:
        raise ValueError(f"shape mismatch: image {img.shape}, mask {mask.shape}")

    valid = ~mask & np.isfinite(img)
    if not valid.any():
        raise ValueError("no valid pixels to interpolate from")

    h, w = img.shape
    out = np.where(valid, img, 0.0)

    for _ in range(max_iter):
        if valid.all():
            break
        pv = np.pad(valid.astype(np.float64), 1)           # 1 = usable neighbour
        pi = np.pad(out, 1)                                # invalid pixels hold 0
        s = np.zeros((h, w))
        c = np.zeros((h, w))
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                s += pi[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] * pv[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
                c += pv[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        fill = ~valid & (c > 0)
        out[fill] = s[fill] / c[fill]
        valid |= fill

    if not valid.all():
        raise ValueError("could not fill all pixels within max_iter passes")
    return out