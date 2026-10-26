"""Detector stage: radiance -> signal electrons and dark electrons (no noise yet)."""
import numpy as np


def responsivity(gain_e_per_dn, bits, offset_dn, l_max, t_int_s):
    """R in electrons per (radiance unit * second), so l_max hits ADC full scale."""
    full_scale_dn = (2**bits - 1) - offset_dn
    return gain_e_per_dn * full_scale_dn / (l_max * t_int_s)


def make_prnu_map(shape, sigma_prnu, rng):
    """Fixed per-pixel multiplicative gain error g ~ N(0, sigma). Pixel gain = 1 + g."""
    return rng.normal(0.0, sigma_prnu, size=shape)


def make_dsnu_map(shape, dsnu_e_per_s, rng):
    """Fixed per-pixel additive dark-rate offset delta (electrons/s)."""
    return rng.normal(0.0, dsnu_e_per_s, size=shape)


def signal_electrons(l_blur, resp, t_int_s, prnu_g):
    """Mean signal electrons S = R * L_blur * t_int * (1 + g)."""
    return resp * l_blur * t_int_s * (1.0 + prnu_g)


def dark_electrons(dark_e_per_s, dsnu_delta, t_int_s):
    """Mean dark electrons D = (d0 + delta) * t_int, never negative."""
    return np.clip(dark_e_per_s + dsnu_delta, 0.0, None) * t_int_s