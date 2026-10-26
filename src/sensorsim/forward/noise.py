"""Noise stage: Poisson shot noise on (signal + dark), then Gaussian read noise. Units: electrons."""
import numpy as np


def add_shot_noise(mean_electrons, rng):
    """Draw Poisson samples around the mean electron count (signal + dark)."""
    mean_electrons = np.asarray(mean_electrons, dtype=np.float64)
    if not np.all(np.isfinite(mean_electrons)) or np.any(mean_electrons < 0):
        raise ValueError("mean electrons must be finite and >= 0 (mask nodata first)")
    return rng.poisson(mean_electrons).astype(np.float64)


def add_read_noise(electrons, read_noise_e, rng):
    """Add zero-mean Gaussian read noise with std read_noise_e (electrons rms)."""
    return electrons + rng.normal(0.0, read_noise_e, size=electrons.shape)


def apply_noise(signal_e, dark_e, read_noise_e, rng):
    """Full noise chain: E = Poisson(S + D) + N(0, read_noise^2)."""
    return add_read_noise(add_shot_noise(signal_e + dark_e, rng), read_noise_e, rng)