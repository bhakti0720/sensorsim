"""Experiment 09: sensitivity sweep on the water scene.

Axes: read noise, bit depth, defect rate. Each point averaged over seeds.
Only edits cfg and calls run_pipeline.
"""
import copy
import json
from typing import cast

import numpy as np
import yaml

from sensorsim.pipeline import run_pipeline

BASE = yaml.safe_load(open(r"configs\sensor_baseline.yaml"))
L = np.load(r"results\scene_water_truth_radiance.npy")
SEEDS = [42, 43, 44]
KEYS = ["rmse_l1_fixed", "k_err_pct", "flat_response_rms_err",
        "det_precision", "det_recall", "det_f1", "n_true_defects"]


def set_read_noise(cfg, v):
    cfg["sensor"]["read_noise_e"] = float(v)


def set_bits(cfg, v):
    # keep full-well electrons fixed: K scales so only quantisation changes
    cfg["sensor"]["bits"] = int(v)
    cfg["sensor"]["gain_e_per_dn"] = BASE["sensor"]["gain_e_per_dn"] * 2 ** (BASE["sensor"]["bits"] - int(v))


def set_defect_factor(cfg, v):
    for k in ("dead_rate", "hot_rate", "stuck_rate"):
        cfg["defects"][k] = BASE["defects"][k] * float(v)


AXES = {
    "read_noise_e": ([5, 10, 20, 40, 80], set_read_noise),
    "bits": ([8, 10, 12, 14], set_bits),
    "defect_factor": ([1, 4, 16, 40], set_defect_factor),
}

results = {}
for axis, (values, setter) in AXES.items():
    print(f"\n=== {axis} ===")
    print(f"{'value':>8s}{'RMSE':>9s}{'K err %':>10s}{'flat err':>10s}{'prec':>7s}{'recall':>8s}{'F1':>7s}{'defects':>9s}")
    rows = []
    assert isinstance(values, (list, tuple))
    for v in values:
        runs = []
        for seed in SEEDS:
            cfg = copy.deepcopy(BASE)
            cfg["seed"] = seed
            setter(cfg, v)
            runs.append(run_pipeline(cfg, L))
        row = {k: float(np.mean([r[k] for r in runs])) for k in KEYS}
        row["rmse_std"] = float(np.std([r["rmse_l1_fixed"] for r in runs]))
        row["value"] = v
        rows.append(row)
        print(f"{v:>8}{row['rmse_l1_fixed']:>9.4f}{row['k_err_pct']:>10.4f}"
              f"{row['flat_response_rms_err']:>10.5f}{row['det_precision']:>7.3f}"
              f"{row['det_recall']:>8.3f}{row['det_f1']:>7.3f}{row['n_true_defects']:>9.0f}")
    results[axis] = rows

json.dump(results, open(r"results\sweep_water.json", "w"), indent=2)