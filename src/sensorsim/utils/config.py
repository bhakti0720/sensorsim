from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class SensorConfig:
    bits: int
    gain_e_per_dn: float
    offset_dn: float
    read_noise_e: float
    dark_current_e_per_s: float
    dsnu_e_per_s: float
    sigma_prnu: float
    t_int_s: float
    psf_sigma_px: float
    l_ref: float
    l_max: float


@dataclass(frozen=True)
class DefectConfig:
    dead_rate: float
    hot_rate: float
    stuck_rate: float
    hot_excess_e: float


@dataclass(frozen=True)
class StripeConfig:
    col_gain_sigma: float
    col_offset_sigma_dn: float


@dataclass(frozen=True)
class Config:
    seed: int
    sensor: SensorConfig
    defects: DefectConfig
    stripes: StripeConfig


def load_config(path):
    with open(Path(path), "r") as f:
        raw = yaml.safe_load(f)
    return Config(
        seed=raw["seed"],
        sensor=SensorConfig(**raw["sensor"]),
        defects=DefectConfig(**raw["defects"]),
        stripes=StripeConfig(**raw["stripes"]),
    )