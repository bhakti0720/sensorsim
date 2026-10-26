from sensorsim.utils.config import load_config  # pyright: ignore[reportMissingImports]
from sensorsim.utils.seeding import make_rng


def test_load_config():
    cfg = load_config("configs/sensor_baseline.yaml")
    assert cfg.sensor.bits == 12
    assert cfg.sensor.gain_e_per_dn > 0


def test_same_seed_same_numbers():
    a = make_rng(42).normal(size=5)
    b = make_rng(42).normal(size=5)
    assert (a == b).all()