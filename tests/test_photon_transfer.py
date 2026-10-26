import numpy as np
import pytest

from sensorsim.calibration.photon_transfer import estimate_gain

K_TRUE, OFFSET, READ_E = 22.0, 100.0, 20.0
SHAPE = (64, 64)


def _make_level(signal_e, n_frames, rng, prnu):
    s = signal_e * (1.0 + prnu)
    e = rng.poisson(s, size=(n_frames, *SHAPE)) + rng.normal(0, READ_E, size=(n_frames, *SHAPE))
    return np.clip(np.round(e / K_TRUE + OFFSET), 0, 4095).astype(np.uint16)


def _make_levels(signals_e, n_frames=20, seed=0):
    rng = np.random.default_rng(seed)
    prnu = rng.normal(0, 0.01, size=SHAPE)
    return [_make_level(s, n_frames, rng, prnu) for s in signals_e]


DARK = np.full(SHAPE, OFFSET)
SIGNALS = np.linspace(500, 60000, 10)


def test_gain_recovered_within_2_percent():
    res = estimate_gain(_make_levels(SIGNALS), DARK)
    assert res.gain_e_per_dn == pytest.approx(K_TRUE, rel=0.02)
    assert res.r2 > 0.999


def test_read_noise_recovered():
    res = estimate_gain(_make_levels(SIGNALS), DARK)
    assert res.read_noise_e == pytest.approx(READ_E, rel=0.1)


def test_saturated_level_is_dropped():
    signals = np.append(SIGNALS, 100000.0)      # clips at 4095 DN
    res = estimate_gain(_make_levels(signals), DARK)
    assert res.n_levels_used == len(SIGNALS)
    assert res.gain_e_per_dn == pytest.approx(K_TRUE, rel=0.02)


def test_defect_pixels_do_not_bias_gain():
    levels = _make_levels(SIGNALS)
    for lv in levels:
        lv[:, 5, 5] = 0         # dead pixel
        lv[:, 9, 9] = 4000      # stuck pixel
    res = estimate_gain(levels, DARK)
    assert res.gain_e_per_dn == pytest.approx(K_TRUE, rel=0.02)


def test_rejects_too_few_levels():
    with pytest.raises(ValueError):
        estimate_gain(_make_levels(SIGNALS[:2]), DARK)