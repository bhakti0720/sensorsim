import numpy as np
import pytest
from sensorsim.forward.noise import add_shot_noise, add_read_noise, apply_noise

SHAPE = (256, 256)   # 65,536 samples -> variance estimate good to ~0.5%


@pytest.mark.parametrize("mean", [20.0, 1000.0, 19000.0])
def test_shot_noise_variance_equals_mean(mean):
    out = add_shot_noise(np.full(SHAPE, mean), np.random.default_rng(0))
    assert abs(out.mean() - mean) / mean < 0.01
    assert abs(out.var() / mean - 1.0) < 0.03       # the Day 3-4 gate


def test_read_noise_std_matches_config():
    out = add_read_noise(np.zeros(SHAPE), 20.0, np.random.default_rng(1))
    assert abs(out.mean()) < 0.5
    assert abs(out.std() - 20.0) / 20.0 < 0.02


def test_total_variance_is_shot_plus_read():
    S, D, sig = np.full(SHAPE, 480.0), np.full(SHAPE, 20.0), 20.0
    out = apply_noise(S, D, sig, np.random.default_rng(2))
    assert abs(out.var() - (500.0 + sig**2)) / 900.0 < 0.03


def test_reproducible_with_same_seed():
    S, D = np.full((32, 32), 300.0), np.full((32, 32), 0.1)
    a = apply_noise(S, D, 20.0, np.random.default_rng(7))
    b = apply_noise(S, D, 20.0, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_rejects_negative_or_nan():
    with pytest.raises(ValueError):
        add_shot_noise(np.array([1.0, -1.0]), np.random.default_rng(0))
    with pytest.raises(ValueError):
        add_shot_noise(np.array([1.0, np.nan]), np.random.default_rng(0))