import numpy as np
import pytest

from sensorsim.calibration.dark import estimate_dark


def _make_frames(n, shape=(64, 64), sigma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    true_level = 100.0 + rng.normal(0, 3.0, size=shape)   # fixed per-pixel pattern
    frames = true_level + rng.normal(0, sigma, size=(n, *shape))
    return true_level, frames


def test_dark_level_error_matches_theory():
    true_level, frames = _make_frames(n=200, sigma=1.0)
    cal = estimate_dark(frames)
    rms = np.sqrt(np.mean((cal.dark_level_dn - true_level) ** 2))
    expected = 1.0 / np.sqrt(200)
    assert rms == pytest.approx(expected, rel=0.1)


def test_read_noise_recovered():
    _, frames = _make_frames(n=200, sigma=1.0)
    cal = estimate_dark(frames)
    assert cal.read_noise_dn_median == pytest.approx(1.0, rel=0.03)


def test_more_frames_means_smaller_error():
    true_level, f_small = _make_frames(n=10, sigma=1.0, seed=1)
    _, f_big = _make_frames(n=100, sigma=1.0, seed=1)
    # same true_level (same seed for pattern), different noise draw size
    true_big, f_big = _make_frames(n=100, sigma=1.0, seed=1)
    e_small = np.sqrt(np.mean((estimate_dark(f_small).dark_level_dn - true_level) ** 2))
    e_big = np.sqrt(np.mean((estimate_dark(f_big).dark_level_dn - true_big) ** 2))
    assert e_big < e_small


def test_uint16_input_does_not_wrap():
    rng = np.random.default_rng(2)
    frames = rng.integers(95, 105, size=(20, 16, 16)).astype(np.uint16)
    cal = estimate_dark(frames)
    assert 95 <= cal.dark_level_dn.min() and cal.dark_level_dn.max() <= 105
    assert cal.dark_level_dn.dtype == np.float64


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        estimate_dark(np.zeros((1, 8, 8)))      # only one frame
    with pytest.raises(ValueError):
        estimate_dark(np.zeros((8, 8)))         # not a stack