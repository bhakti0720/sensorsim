from pathlib import Path

import numpy as np
import yaml

from sensorsim.pipeline import run_pipeline

CFG_PATH = Path(__file__).resolve().parents[1] / "configs" / "sensor_baseline.yaml"


def _cfg():
    return yaml.safe_load(open(CFG_PATH))


def _radiance():
    rng = np.random.default_rng(0)
    return 20 + 30 * rng.random((64, 64))


def test_pipeline_reproducible():
    a = run_pipeline(_cfg(), _radiance(), n_dark=20, n_levels=8, n_per_level=10)
    b = run_pipeline(_cfg(), _radiance(), n_dark=20, n_levels=8, n_per_level=10)
    assert a == b


def test_pipeline_recovers_gain():
    out = run_pipeline(_cfg(), _radiance(), n_dark=20, n_levels=8, n_per_level=10)
    assert abs(out["k_err_pct"]) < 2.0
    assert out["flat_response_rms_err"] < 0.01