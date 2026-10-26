import numpy as np
from scipy import ndimage
from sensorsim.forward.optics import gaussian_kernel, apply_psf


def test_kernel_normalised_symmetric_odd():
    k = gaussian_kernel(0.5)
    assert np.isclose(k.sum(), 1.0)
    assert k.shape[0] == k.shape[1] and k.shape[0] % 2 == 1
    assert np.allclose(k, k.T) and np.allclose(k, k[::-1, ::-1])
    assert k.argmax() == k.size // 2          # peak at the centre


def test_constant_image_unchanged():
    img = np.full((32, 32), 80.0)
    assert np.allclose(apply_psf(img, 0.5), 80.0)   # also checks the edges


def test_impulse_preserves_energy_and_width():
    img = np.zeros((41, 41)); img[20, 20] = 1.0
    out = apply_psf(img, 2.0)
    assert np.isclose(out.sum(), 1.0)
    row = out[20]; x = np.arange(41) - 20
    std = np.sqrt((row * x**2).sum() / row.sum())
    assert abs(std - 2.0) / 2.0 < 0.03


def test_matches_scipy_gaussian_filter():
    rng = np.random.default_rng(0)
    img = rng.random((64, 64))
    ref = ndimage.gaussian_filter(img, sigma=0.5, truncate=3.0, mode="reflect")
    assert np.allclose(apply_psf(img, 0.5), ref, atol=1e-12)