import numpy as np
from sensorsim.forward.detector import (responsivity, make_prnu_map,make_dsnu_map, signal_electrons,dark_electrons)

K, BITS, OFF, LMAX, T = 22.0, 12, 100, 484.0, 0.0015


def test_lmax_hits_full_scale():
    R = responsivity(K, BITS, OFF, LMAX, T)
    e = signal_electrons(np.array([LMAX]), R, T, 0.0)
    assert np.isclose(e[0] / K + OFF, 4095)


def test_shot_snr_at_lref_near_published():
    R = responsivity(K, BITS, OFF, LMAX, T)
    snr = np.sqrt(signal_electrons(108.0, R, T, 0.0))
    assert abs(snr - 142) / 142 < 0.05


def test_prnu_statistics_and_reproducibility():
    g = make_prnu_map((512, 512), 0.01, np.random.default_rng(1))
    assert abs(g.mean()) < 1e-3 and abs(g.std() - 0.01) < 5e-4
    g2 = make_prnu_map((512, 512), 0.01, np.random.default_rng(1))
    assert np.array_equal(g, g2)


def test_signal_scales_linearly_and_prnu_is_multiplicative():
    R = responsivity(K, BITS, OFF, LMAX, T)
    L = np.full((4, 4), 100.0)
    g = np.full((4, 4), 0.02)
    assert np.allclose(signal_electrons(2 * L, R, T, g),
                       2 * signal_electrons(L, R, T, g))
    assert np.allclose(signal_electrons(L, R, T, g),
                       1.02 * signal_electrons(L, R, T, 0.0))


def test_dark_electrons_nonnegative_and_mean():
    delta = make_dsnu_map((256, 256), 5.0, np.random.default_rng(2))
    d = dark_electrons(50.0, delta, T)
    assert (d >= 0).all() and abs(d.mean() - 50.0 * T) < 1e-3
    