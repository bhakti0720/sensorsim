import numpy as np
import pytest

from sensorsim.calibration.flatfield import estimate_flatfield

K, OFFSET, READ_E = 22.0, 100.0, 20.0
SHAPE = (128, 64)
DARK = np.full(SHAPE, OFFSET)


def _truth(seed=0):
    rng = np.random.default_rng(seed)
    prnu = rng.normal(0, 0.01, size=SHAPE)
    col = rng.normal(0, 0.005, size=SHAPE[1])
    resp = (1 + prnu) * (1 + col[None, :])
    return resp / np.median(resp), col


def _levels(resp, signals_e, n_frames=20, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for s in signals_e:
        e = rng.poisson(s * resp, size=(n_frames, *SHAPE)) \
            + rng.normal(0, READ_E, size=(n_frames, *SHAPE))
        out.append(np.clip(np.round(e / K + OFFSET), 0, 4095).astype(np.uint16))
    return out


SIGNALS = np.linspace(2000, 60000, 8)


def test_response_recovered():
    resp, _ = _truth()
    cal = estimate_flatfield(_levels(resp, SIGNALS), DARK)
    rms = np.sqrt(np.mean((cal.response - resp) ** 2))
    assert rms < 0.002          # PRNU itself is 0.01, so error is >5x smaller
    assert np.median(cal.response) == pytest.approx(1.0, abs=1e-9)


def test_column_and_pixel_parts_multiply_back():
    resp, _ = _truth()
    cal = estimate_flatfield(_levels(resp, SIGNALS), DARK)
    assert np.allclose(cal.prnu * cal.column_gain[None, :], cal.response)


def test_column_gain_recovered_loosely():
    resp, col = _truth()
    cal = estimate_flatfield(_levels(resp, SIGNALS), DARK)
    true_col = (1 + col) / np.median(1 + col)
    rms = np.sqrt(np.mean((cal.column_gain - true_col) ** 2))
    assert rms < 0.003          # split is statistical, so looser than the product


def test_saturated_level_is_dropped():
    resp, _ = _truth()
    signals = np.append(SIGNALS, 100000.0)     # clips at 4095 DN
    cal = estimate_flatfield(_levels(resp, signals), DARK)
    assert cal.n_levels_used == len(SIGNALS)
    assert np.sqrt(np.mean((cal.response - resp) ** 2)) < 0.002


def test_dead_pixel_does_not_disturb_neighbours():
    resp, _ = _truth()
    levels = _levels(resp, SIGNALS)
    for lv in levels:
        lv[:, 5, 5] = 0
    cal = estimate_flatfield(levels, DARK)
    ok = np.ones(SHAPE, dtype=bool)
    ok[5, 5] = False
    assert np.sqrt(np.mean((cal.response - resp)[ok] ** 2)) < 0.002
    assert cal.response[5, 5] < 0.5


def test_rejects_bad_input():
    resp, _ = _truth()
    with pytest.raises(ValueError):
        estimate_flatfield(_levels(resp, SIGNALS[:1]), DARK)   # one level only
    with pytest.raises(ValueError):
        estimate_flatfield([np.zeros(SHAPE)], DARK)            # not a stack