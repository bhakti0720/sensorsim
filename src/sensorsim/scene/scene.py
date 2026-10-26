from pathlib import Path
import yaml

from sensorsim.scene.load import load_reflectance_crop
from sensorsim.scene.radiance import reflectance_to_radiance


def load_scene(scene_cfg_path, data_dir="data"):
    """Load a scene config and return the true radiance crop.

    returns (radiance, valid_mask)
      radiance   : float array, W / m^2 / sr / um   (this is the ground truth)
      valid_mask : True where the pixel has data
    """
    cfg = yaml.safe_load(open(scene_cfg_path))
    path = (Path(data_dir) / (cfg["product"] + ".SAFE") / "GRANULE" / cfg["granule"]
            / "IMG_DATA" / cfg["band_file"])

    refl, valid = load_reflectance_crop(
        path, cfg["crop"]["lon"], cfg["crop"]["lat"], cfg["crop"]["size"],
        cfg["reflectance"]["quantification_value"], cfg["reflectance"]["add_offset"])

    r = cfg["radiance"]
    radiance = reflectance_to_radiance(
        refl, r["esun_w_m2_um"], r["sun_zenith_deg"], r["u_factor"])
    return radiance, valid