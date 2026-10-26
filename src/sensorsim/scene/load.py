import numpy as np
import rasterio
from rasterio.warp import transform
from rasterio.windows import Window


def load_reflectance_crop(path, lon, lat, size, quantification_value, add_offset):
    """Read a square crop of one Sentinel-2 L1C band and return TOA reflectance.

    returns (reflectance, valid_mask)
      reflectance : float array, unitless, clipped at 0 (dark water can go slightly below 0)
      valid_mask  : True where the pixel has data (DN 0 means nodata)
    """
    with rasterio.open(path) as src:
        xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
        row, col = src.index(xs[0], ys[0])
        window = Window(col - size // 2, row - size // 2, size, size)
        dn = src.read(1, window=window)

    valid_mask = dn > 0
    reflectance = (dn.astype(np.float64) + add_offset) / quantification_value
    reflectance = np.clip(reflectance, 0.0, None)
    return reflectance, valid_mask