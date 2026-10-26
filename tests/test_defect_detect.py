import numpy as np
import pytest

from sensorsim.calibration.dark import estimate_dark
from sensorsim.calibration.flatfield import estimate_flatfield
from sensorsim.correction.defect_detect import detect_defects

SHAPE = (64, 48)
SCALE = 8.25
LEVELS = [20.0, 60.0, 120.0, 200.0, 300.0]


def _frames(seed=0, n=30, dead=(), stuck=(), hot=(), stuck_dn=2000):
    rng = np.random.default_rng(seed)
    dark = 100.0 + rng.normal(0, 0.3, size=SHAPE) + rng.normal(0, 3.0, size=SHAPE[1])[None, :]
    resp = 1.0 + rng.normal(0, 0.01, size=SHAPE)
    resp = resp / np.median(resp)
    for p in hot:
        dark[p] += 90.0                                   # fixed excess, visible in every frame

    def make(mean):
        f = np.round(mean + rng.normal(0, 1.0, size=(n, *SHAPE)))
        for p in dead:
            f[:, p[0], p[1]] = 0
        for p in stuck:
            f[:, p[0], p[1]] = stuck_dn
        return f.astype(np.uint16)

    darks = make(dark)
    flats = [make(dark + SCALE * resp * lev) for lev in LEVELS]
    return darks, flats


def _run(**kw):
    darks, flats = _frames(**kw)
    dark_cal = estimate_dark(darks)
    ff = estimate_flatfield(flats, dark_cal.dark_level_dn)
    return detect_defects(darks, flats, dark_cal.dark_level_dn, ff.response)


def _mask(points):
    m = np.zeros(SHAPE, dtype=bool)
    for p in points:
        m[p] = True
    return m


def test_clean_sensor_has_no_detections():
    masks = _run()
    assert masks.combined.sum() == 0


def test_dead_and_stuck_found_as_constant():
    dead, stuck = [(3, 4), (20, 30)], [(10, 10), (50, 5)]
    masks = _run(dead=dead, stuck=stuck)
    assert np.array_equal(masks.constant, _mask(dead + stuck))


def test_stuck_pixel_with_normal_looking_value_is_found():
    masks = _run(stuck=[(7, 7)], stuck_dn=105)            # looks like a normal dark pixel
    assert masks.constant[7, 7]


def test_hot_pixels_found():
    hot = [(5, 5), (40, 12), (60, 40)]
    masks = _run(hot=hot)
    assert masks.hot[tuple(np.array(hot).T)].all()
    assert masks.constant.sum() == 0


def test_combined_equals_planted_set_exactly():
    dead, stuck, hot = [(3, 4)], [(10, 10)], [(5, 5), (40, 12)]
    masks = _run(dead=dead, stuck=stuck, hot=hot)
    assert np.array_equal(masks.combined, _mask(dead + stuck + hot))


def test_rejects_bad_input():
    darks, flats = _frames()
    dl = np.zeros(SHAPE)
    with pytest.raises(ValueError):
        detect_defects(darks[0], flats, dl, np.ones(SHAPE))        # not a stack
    with pytest.raises(ValueError):
        detect_defects(darks, flats, np.zeros((4, 4)), np.ones(SHAPE))   # shape mismatch
    with pytest.raises(ValueError):
        detect_defects(darks, [], dl, np.ones(SHAPE))              # no flats
        