"""ADC stage: electrons -> integer DN. Last step of the forward model that uses electrons."""
import numpy as np


def electrons_to_dn(electrons, gain_e_per_dn, offset_dn, bits):
    """DN = clip(round(E / K + offset), 0, 2**bits - 1), returned as uint16."""
    if bits > 16:
        raise ValueError("uint16 output supports up to 16 bits")
    max_dn = 2**bits - 1
    dn = np.rint(np.asarray(electrons, dtype=np.float64) / gain_e_per_dn + offset_dn)
    return np.clip(dn, 0, max_dn).astype(np.uint16)


def saturation_mask(dn, bits):
    """True where the pixel hit full scale (information lost)."""
    return dn >= (2**bits - 1)