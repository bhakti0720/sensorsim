"""Defect and stripe stage. Defect maps and column errors are FIXED per detector (truth)."""
import numpy as np


def make_column_errors(ncols, gain_sigma, offset_sigma_dn, rng):
    """One gain error (fraction) and one offset error (DN) per column."""
    return rng.normal(0.0, gain_sigma, ncols), rng.normal(0.0, offset_sigma_dn, ncols)


def apply_column_gain(signal_e, col_gain):
    """Signal electrons times (1 + gain error) of its column."""
    return signal_e * (1.0 + col_gain[None, :])


def apply_column_offset(electrons, col_offset_dn, gain_e_per_dn):
    """Add a per-column offset, given in DN, as electrons (before the ADC)."""
    return electrons + col_offset_dn[None, :] * gain_e_per_dn


def make_defect_maps(shape, dead_rate, hot_rate, stuck_rate, bits, rng):
    """Mutually exclusive boolean masks plus the fixed DN of each stuck pixel."""
    u = rng.random(shape)                      # one draw per pixel -> classes cannot overlap
    dead = u < dead_rate
    hot = (u >= dead_rate) & (u < dead_rate + hot_rate)
    stuck = (u >= dead_rate + hot_rate) & (u < dead_rate + hot_rate + stuck_rate)
    stuck_dn = rng.integers(0, 2**bits, size=shape)
    return {"dead": dead, "hot": hot, "stuck": stuck, "stuck_dn": stuck_dn}


def apply_hot_pixels(dark_e, hot_mask, hot_excess_e):
    """Add excess electrons at hot pixels (before noise, so shot noise applies)."""
    return dark_e + hot_mask * hot_excess_e


def apply_dead_stuck(dn, maps):
    """Force dead pixels to 0 and stuck pixels to their fixed value (after the ADC)."""
    out = dn.copy()
    out[maps["dead"]] = 0
    out[maps["stuck"]] = maps["stuck_dn"][maps["stuck"]].astype(out.dtype)
    return out