"""Optics stage: Gaussian PSF blur applied to radiance (before any noise)."""
import numpy as np
from scipy import ndimage


def gaussian_kernel(sigma_px: float) -> np.ndarray:
    """Normalised 2D Gaussian kernel, radius = ceil(3*sigma), so size is odd."""
    if sigma_px <= 0:
        raise ValueError("sigma_px must be > 0")
    radius = int(np.ceil(3.0 * sigma_px))
    ax = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma_px**2))
    return kernel / kernel.sum()          # sums to 1: blur must not change total radiance


def apply_psf(radiance: np.ndarray, sigma_px: float) -> np.ndarray:
    """Blur a 2D radiance image with a Gaussian PSF. sigma_px == 0 means no blur."""
    if radiance.ndim != 2:
        raise ValueError("expected a 2D (single-band) array")
    if sigma_px == 0:
        return radiance.copy()
    return ndimage.convolve(radiance.astype(np.float64),
                            gaussian_kernel(sigma_px), mode="reflect")