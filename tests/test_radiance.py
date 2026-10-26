import numpy as np
import pytest
from sensorsim.scene.radiance import reflectance_to_radiance


def test_known_value():
    # rho=0.3, Esun=1500, sun straight overhead, mean distance
    # by hand: 0.3 * 1500 / pi = 143.239
    out = reflectance_to_radiance(np.array([0.3]), 1500.0, 0.0)
    assert out[0] == pytest.approx(143.239, abs=0.01)


def test_zero_reflectance_gives_zero():
    out = reflectance_to_radiance(np.zeros((4, 4)), 1500.0, 30.0)
    assert (out == 0).all()


def test_lower_sun_means_less_radiance():
    high_sun = reflectance_to_radiance(np.array([0.3]), 1500.0, 20.0)
    low_sun = reflectance_to_radiance(np.array([0.3]), 1500.0, 60.0)
    assert low_sun[0] < high_sun[0]