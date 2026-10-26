"""The C++ port must match the NumPy version. Skipped if not built."""
import numpy as np
import pytest

cpp = pytest.importorskip("sensorsim._fixcore")
from sensorsim.correction.defect_fix import fix_defects as fix_py  # noqa: E402


def _same(img, mask, **kw):
    a = fix_py(img, mask, **kw)
    b = cpp.fix_defects(img, mask, **kw)
    np.testing.assert_allclose(b, a, rtol=1e-12, atol=0)
    return a, b


def _img(seed=0, shape=(64, 64)):
    return 20 + 30 * np.random.default_rng(seed).random(shape)


def test_random_sparse_mask():
    rng = np.random.default_rng(1)
    img = _img()
    _same(img, rng.random(img.shape) < 0.02)


def test_random_dense_mask_with_clusters():
    rng = np.random.default_rng(2)
    img = _img(3)
    _same(img, rng.random(img.shape) < 0.3)


def test_cluster_needs_several_passes():
    img = _img(4)
    mask = np.zeros(img.shape, bool)
    mask[20:29, 30:39] = True          # 9x9 block: 5 passes
    _same(img, mask)


def test_corner_and_edge_pixels():
    img = _img(5)
    mask = np.zeros(img.shape, bool)
    mask[0, 0] = mask[0, -1] = mask[-1, 0] = mask[-1, -1] = True
    mask[0, 10] = mask[30, 0] = True
    _same(img, mask)


def test_nan_is_treated_as_invalid():
    img = _img(6)
    img[10, 10] = np.nan
    img[11, 10] = np.nan
    mask = np.zeros(img.shape, bool)
    mask[40, 40] = True
    a, b = _same(img, mask)
    assert np.all(np.isfinite(b))


def test_valid_pixels_unchanged():
    img = _img(7)
    mask = np.random.default_rng(8).random(img.shape) < 0.05
    out = cpp.fix_defects(img, mask)
    np.testing.assert_array_equal(out[~mask], img[~mask])


def test_other_input_dtypes():
    mask = np.random.default_rng(9).random((32, 32)) < 0.05
    for dt in (np.float32, np.uint16):
        img = (_img(10, (32, 32)) * 10).astype(dt)
        _same(img, mask)


def test_errors_match():
    img = _img()
    ok = np.zeros(img.shape, bool)
    ok[5, 5] = True
    with pytest.raises(ValueError):
        cpp.fix_defects(img, np.ones(img.shape, bool))        # no valid pixels
    with pytest.raises(ValueError):
        cpp.fix_defects(img, ok[:10])                         # shape mismatch
    with pytest.raises(ValueError):
        cpp.fix_defects(img[0], ok[0])                        # not 2-D
    block = np.zeros(img.shape, bool)
    block[20:40, 20:40] = True
    with pytest.raises(ValueError):
        cpp.fix_defects(img, block, max_iter=2)               # too few passes
    with pytest.raises(ValueError):
        fix_py(img, block, max_iter=2)