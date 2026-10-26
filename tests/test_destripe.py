import numpy as np
import pytest

from sensorsim.correction.destripe import destripe

H, W = 128, 100


def _noise(seed=0, sigma=1.0):
    return np.random.default_rng(seed).normal(0, sigma, size=(H, W))


def _col_rms_error(out, clean):
    """RMS of per-column mean error: a measure of leftover stripes."""
    return float(np.sqrt(np.mean((out.mean(axis=0) - clean.mean(axis=0)) ** 2)))


def test_offset_stripes_are_removed():
    clean = 100.0 + _noise(0)
    stripes = np.random.default_rng(1).normal(0, 1.0, size=W)
    out = destripe(clean + stripes[None, :], mode="offset")
    before = _col_rms_error(clean + stripes[None, :], clean)
    assert _col_rms_error(out, clean) < 0.3 * before


def test_gain_stripes_are_removed():
    clean = 100.0 + _noise(2)
    gains = 1.0 + np.random.default_rng(3).normal(0, 0.02, size=W)
    striped = clean * gains[None, :]
    out = destripe(striped, mode="gain")
    assert _col_rms_error(out, clean) < 0.3 * _col_rms_error(striped, clean)


def test_stripe_free_image_barely_changes():
    img = 100.0 + _noise(4)
    out = destripe(img, mode="offset")
    assert np.sqrt(np.mean((out - img) ** 2)) < 0.2     # only column-median noise (~0.1)


def test_smooth_gradient_is_preserved_in_interior():
    x = np.arange(W)
    img = 100.0 + 0.5 * x[None, :] + _noise(5)          # real brightness slope along columns
    out = destripe(img, mode="offset", window=31)
    inner = slice(16, W - 16)                           # away from reflected edges
    assert np.abs(out.mean(axis=0)[inner] - img.mean(axis=0)[inner]).max() < 0.4


def test_wide_bright_band_is_preserved():
    img = 100.0 + _noise(6)
    img[:, 30:70] += 20.0                               # 40-column real feature
    out = destripe(img, mode="offset", window=31)
    assert abs(out[:, 45:56].mean() - img[:, 45:56].mean()) < 0.3


def test_nan_pixels_do_not_poison_and_stay_nan():
    img = 100.0 + _noise(7)
    rng = np.random.default_rng(8)
    bad = rng.random(img.shape) < 0.01
    img[bad] = np.nan
    out = destripe(img, mode="offset")
    assert np.array_equal(np.isnan(out), np.isnan(img))
    assert np.isfinite(out[~bad]).all()


def test_valid_mask_excludes_pixels_from_estimate():
    clean = 100.0 + _noise(9)
    stripes = np.random.default_rng(10).normal(0, 1.0, size=W)
    img = clean + stripes[None, :]
    img[5, 40] = 5000.0                                  # a defect that would skew nothing (median)
    valid = np.ones(img.shape, dtype=bool)
    valid[5, 40] = False
    out = destripe(img, mode="offset", valid=valid)
    assert _col_rms_error(out[np.arange(H) != 5], clean[np.arange(H) != 5]) < 0.5


def test_rejects_bad_input():
    img = 100.0 + _noise(11)
    with pytest.raises(ValueError):
        destripe(img[0], mode="offset")                          # not 2-D
    with pytest.raises(ValueError):
        destripe(img, mode="sideways")                           # bad mode
    with pytest.raises(ValueError):
        destripe(img, window=30)                                 # even window
    with pytest.raises(ValueError):
        destripe(img, window=W + 2)                              # wider than the image
    with pytest.raises(ValueError):
        destripe(img, valid=np.ones((3, 3), dtype=bool))         # shape mismatch