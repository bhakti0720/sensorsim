import numpy as np
import pytest

from sensorsim.correction.defect_fix import fix_defects


def _ramp(h=20, w=24):
    y, x = np.mgrid[0:h, 0:w]
    return 3.0 * x + 2.0 * y + 10.0


def test_isolated_defect_on_linear_ramp_is_exact():
    img = _ramp()
    mask = np.zeros(img.shape, dtype=bool)
    mask[8, 9] = True
    bad = img.copy()
    bad[8, 9] = 9999.0
    out = fix_defects(bad, mask)
    assert out[8, 9] == pytest.approx(img[8, 9], abs=1e-9)   # symmetric neighbours -> exact


def test_valid_pixels_are_unchanged():
    img = _ramp()
    mask = np.zeros(img.shape, dtype=bool)
    mask[5, 5] = mask[12, 3] = True
    out = fix_defects(img, mask)
    assert np.array_equal(out[~mask], img[~mask])


def test_cluster_is_filled_from_the_outside_in():
    img = np.full((15, 15), 7.0)
    mask = np.zeros(img.shape, dtype=bool)
    mask[4:9, 4:9] = True                                    # 5x5 hole, centre has no valid neighbour
    bad = img.copy()
    bad[mask] = np.nan
    out = fix_defects(bad, mask)
    assert np.isfinite(out).all()
    assert np.allclose(out, 7.0)


def test_edge_and_corner_defects_use_available_neighbours():
    img = np.full((10, 10), 5.0)
    mask = np.zeros(img.shape, dtype=bool)
    mask[0, 0] = mask[0, 5] = mask[9, 9] = True
    out = fix_defects(img, mask)
    assert np.allclose(out, 5.0)


def test_nan_pixels_are_treated_as_defects_even_if_unmasked():
    img = np.full((8, 8), 4.0)
    img[3, 3] = np.nan
    out = fix_defects(img, np.zeros(img.shape, dtype=bool))
    assert np.isfinite(out).all() and out[3, 3] == pytest.approx(4.0)


def test_sparse_valid_pixels_spread_everywhere():
    img = np.full((11, 11), 0.0)
    img[5, 5] = 3.5
    mask = np.ones(img.shape, dtype=bool)
    mask[5, 5] = False
    out = fix_defects(img, mask)
    assert np.allclose(out, 3.5)


def test_uint16_input_returns_float64():
    img = np.full((6, 6), 100, dtype=np.uint16)
    mask = np.zeros(img.shape, dtype=bool)
    mask[2, 2] = True
    out = fix_defects(img, mask)
    assert out.dtype == np.float64 and out[2, 2] == pytest.approx(100.0)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        fix_defects(np.zeros((4, 4)), np.zeros((3, 3), dtype=bool))      # shape mismatch
    with pytest.raises(ValueError):
        fix_defects(np.zeros(5), np.zeros(5, dtype=bool))                # not 2-D
    with pytest.raises(ValueError):
        fix_defects(np.zeros((4, 4)), np.ones((4, 4), dtype=bool))       # nothing valid