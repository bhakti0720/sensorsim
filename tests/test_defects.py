import numpy as np
from sensorsim.forward.defects import (make_column_errors, apply_column_gain,
    apply_column_offset, make_defect_maps, apply_hot_pixels, apply_dead_stuck)


def test_column_errors_statistics():
    g, o = make_column_errors(100000, 0.01, 1.0, np.random.default_rng(0))
    assert g.shape == o.shape == (100000,)
    assert abs(g.std() - 0.01) < 2e-4 and abs(o.std() - 1.0) < 0.02


def test_column_gain_is_constant_down_each_column():
    sig = np.random.default_rng(1).uniform(100, 5000, (64, 32))
    g = np.random.default_rng(2).normal(0, 0.01, 32)
    ratio = apply_column_gain(sig, g) / sig
    assert np.allclose(ratio, 1.0 + g[None, :])          # identical in every row
    assert np.allclose(ratio.std(axis=0), 0.0)


def test_column_offset_in_electrons():
    e = np.zeros((10, 5)); off = np.array([1.0, -1.0, 0.0, 2.0, 0.5])
    out = apply_column_offset(e, off, 22.0)
    assert np.allclose(out, off[None, :] * 22.0)


def test_defect_masks_exclusive_and_rates():
    m = make_defect_maps((1000, 1000), 0.0005, 0.0005, 0.0002, 12, np.random.default_rng(3))
    assert not (m["dead"] & m["hot"]).any()
    assert not (m["dead"] & m["stuck"]).any() and not (m["hot"] & m["stuck"]).any()
    for key, rate in [("dead", 5e-4), ("hot", 5e-4), ("stuck", 2e-4)]:
        expected = rate * 1e6
        assert abs(m[key].sum() - expected) < 4 * np.sqrt(expected)   # 4 sigma of Poisson counting noise

def test_reproducible():
    a = make_defect_maps((100, 100), 0.01, 0.01, 0.01, 12, np.random.default_rng(5))
    b = make_defect_maps((100, 100), 0.01, 0.01, 0.01, 12, np.random.default_rng(5))
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_hot_pixels_add_only_at_mask():
    hot = np.zeros((4, 4), bool); hot[1, 2] = True
    out = apply_hot_pixels(np.full((4, 4), 0.1), hot, 2000.0)
    assert np.isclose(out[1, 2], 2000.1) and np.isclose(out.sum() - 2000.0, 1.6)


def test_dead_and_stuck_applied_after_adc():
    dn = np.full((4, 4), 300, dtype=np.uint16)
    maps = {"dead": np.zeros((4, 4), bool), "stuck": np.zeros((4, 4), bool),
            "stuck_dn": np.full((4, 4), 1234)}
    maps["dead"][0, 0] = True; maps["stuck"][3, 3] = True
    out = apply_dead_stuck(dn, maps)
    assert out.dtype == np.uint16 and out[0, 0] == 0 and out[3, 3] == 1234
    assert (out == 300).sum() == 14                      # everything else untouched