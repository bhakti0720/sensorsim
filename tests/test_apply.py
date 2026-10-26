import numpy as np
import pytest

from sensorsim.calibration.apply import apply_calibration, estimate_radiometric_scale

SHAPE = (64, 48)
SCALE = 8.25                      # DN per radiance unit


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    dark = 100.0 + rng.normal(0, 3.0, size=SHAPE)
    resp = 1.0 + rng.normal(0, 0.012, size=SHAPE)
    resp = resp / np.median(resp)
    return rng, dark, resp


def test_noiseless_roundtrip_is_exact():
    rng, dark, resp = _setup()
    L = rng.uniform(10, 250, size=SHAPE)
    dn = dark + SCALE * resp * L                      # float, no noise, no rounding
    out = apply_calibration(dn, dark, resp, SCALE)
    assert np.allclose(out, L, rtol=1e-12)


def test_uint16_input_does_not_wrap():
    rng, dark, resp = _setup()
    dn = np.full(SHAPE, 50, dtype=np.uint16)          # below the dark level
    out = apply_calibration(dn, dark, resp, SCALE)
    assert np.all(out < 0)                            # negative, not ~65000 wrap-around
    assert out.dtype == np.float64


def test_low_response_pixels_become_nan():
    rng, dark, resp = _setup()
    resp[3, 4] = 0.0                                  # dead pixel
    resp[7, 8] = 0.2                                  # stuck-like pixel
    dn = dark + SCALE * resp * 100.0
    out = apply_calibration(dn, dark, resp, SCALE)
    assert np.isnan(out[3, 4]) and np.isnan(out[7, 8])
    assert np.isfinite(out).sum() == out.size - 2


def test_scale_recovered_from_noisy_flats():
    rng, dark, resp = _setup()
    levels = np.linspace(20, 400, 8)
    flats = []
    for lev in levels:
        mean = dark + SCALE * resp * lev
        flats.append(np.round(mean + rng.normal(0, 1.0, size=(20, *SHAPE))).astype(np.uint16))
    s = estimate_radiometric_scale(levels, flats, dark)
    assert s == pytest.approx(SCALE, rel=0.005)


def test_saturated_level_is_dropped_from_scale():
    rng, dark, resp = _setup()
    levels = np.array([20, 100, 200, 300, 2000.0])    # last one clips at 4095 DN
    flats = []
    for lev in levels:
        mean = dark + SCALE * resp * lev
        dn = np.clip(np.round(mean + rng.normal(0, 1.0, size=(10, *SHAPE))), 0, 4095)
        flats.append(dn.astype(np.uint16))
    s = estimate_radiometric_scale(levels, flats, dark)
    assert s == pytest.approx(SCALE, rel=0.005)


def test_rejects_bad_input():
    rng, dark, resp = _setup()
    with pytest.raises(ValueError):
        apply_calibration(np.zeros((4, 4)), dark, resp, SCALE)       # shape mismatch
    with pytest.raises(ValueError):
        apply_calibration(np.zeros(SHAPE), dark, resp, -1.0)         # bad scale
    with pytest.raises(ValueError):
        estimate_radiometric_scale([50.0], [np.zeros((5, *SHAPE))], dark)   # one level only