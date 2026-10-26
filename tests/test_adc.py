import numpy as np
from sensorsim.forward.adc import electrons_to_dn, saturation_mask

K, OFF, BITS = 22.0, 100, 12


def test_known_values():
    e = np.array([0.0, 220.0, 2200.0])
    assert np.array_equal(electrons_to_dn(e, K, OFF, BITS), [100, 110, 200])


def test_clipping_and_dtype():
    out = electrons_to_dn(np.array([-1e6, 1e9]), K, OFF, BITS)
    assert out.dtype == np.uint16
    assert out[0] == 0 and out[1] == 4095


def test_round_trip_within_half_dn():
    e = np.random.default_rng(0).uniform(1000, 80000, 10000)
    back = (electrons_to_dn(e, K, OFF, BITS).astype(float) - OFF) * K
    assert np.abs(back - e).max() <= K / 2 + 1e-9


def test_quantisation_noise_is_one_over_sqrt12():
    e = np.random.default_rng(1).uniform(1000, 80000, 200000)
    err_dn = electrons_to_dn(e, K, OFF, BITS) - (e / K + OFF)
    assert abs(err_dn.std() - 1 / np.sqrt(12)) / (1 / np.sqrt(12)) < 0.02


def test_monotonic():
    e = np.linspace(-500, 100000, 5000)
    assert (np.diff(electrons_to_dn(e, K, OFF, BITS).astype(int)) >= 0).all()


def test_saturation_mask():
    dn = np.array([0, 100, 4094, 4095], dtype=np.uint16)
    assert saturation_mask(dn, BITS).tolist() == [False, False, False, True]