"""Print the lat/lon extent of the B04 tile, so we pick a crop inside it."""
from pathlib import Path

import rasterio
from rasterio.warp import transform
import yaml

cfg = yaml.safe_load(open(r"configs\scene_nyc.yaml"))
path = (Path("data") / (cfg["product"] + ".SAFE") / "GRANULE" / cfg["granule"]
        / "IMG_DATA" / cfg["band_file"])

with rasterio.open(path) as src:
    b = src.bounds
    xs, ys = [b.left, b.right], [b.bottom, b.top]
    lons, lats = transform(src.crs, "EPSG:4326", xs, ys)
    print("crs        ", src.crs)
    print("size       ", src.width, "x", src.height)
    print("lon range  ", min(lons), "to", max(lons))
    print("lat range  ", min(lats), "to", max(lats))