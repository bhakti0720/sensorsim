import numpy as np


def reflectance_to_radiance(reflectance, esun_w_m2_um, sun_zenith_deg, u_factor=1.0):
    """Convert TOA reflectance to TOA radiance.

    reflectance      : array, unitless, 0..1
    esun_w_m2_um     : solar irradiance of the band, W / m^2 / um (from metadata)
    sun_zenith_deg   : solar zenith angle in degrees (from metadata)
    u_factor         : Earth-Sun distance correction (from metadata), 1.0 = mean distance

    returns radiance in W / m^2 / sr / um
    """
    cos_theta = np.cos(np.deg2rad(sun_zenith_deg))
    return reflectance * esun_w_m2_um * cos_theta * u_factor / np.pi