"""Experiment 10: where does detection break, and how many frames are needed?

Axes: hot-pixel excess, number of dark frames, frames per flat level.
Water scene, 3 seeds per point.
"""
import copy
import json

import numpy as np
import yaml

from sensorsim.pipeline import run_pipeline

BASE = yaml.safe_load(open(r"configs\sensor_baseline.yaml"))
L = np.load(r"results\scene_water_truth_radiance.npy")
SEEDS = [42, 43, 44]
KEYS = ["rmse_l1_fixed", "k_err_pct", "flat_response_rms_err",
        "det_precision", "det_recall", "det_f1"]


def run_axis(name, values, make_call):
    print(f"\n=== {name} ===")
    print(f"{'value':>8s}{'RMSE':>9s}{'|K err| %':>11s}{'flat err':>10s}{'prec':>7s}{'recall':>8s}{'F1':>7s}")
    rows = []
    for v in values:
        runs = []
        for seed in SEEDS:
            cfg = copy.deepcopy(BASE)
            cfg["seed"] = seed
            runs.append(make_call(cfg, v))
        row = {k: float(np.mean([r[k] for r in runs])) for k in KEYS}
        row["abs_k_err_pct"] = float(np.mean([abs(r["k_err_pct"]) for r in runs]))
        row["value"] = v
        rows.append(row)
        print(f"{v:>8}{row['rmse_l1_fixed']:>9.4f}{row['abs_k_err_pct']:>11.4f}"
              f"{row['flat_response_rms_err']:>10.5f}{row['det_precision']:>7.3f}"
              f"{row['det_recall']:>8.3f}{row['det_f1']:>7.3f}")
    return rows


def hot_call(cfg, v):
    cfg["defects"]["hot_excess_e"] = float(v)
    return run_pipeline(cfg, L)


def dark_call(cfg, v):
    return run_pipeline(cfg, L, n_dark=int(v))


def flat_call(cfg, v):
    return run_pipeline(cfg, L, n_per_level=int(v))


results = {
    "hot_excess_e": run_axis("hot_excess_e", [2000, 500, 200, 100, 50, 20, 10], hot_call),
    "n_dark": run_axis("n_dark", [50, 20, 10, 5, 3], dark_call),
    "n_per_level": run_axis("n_per_level", [30, 10, 5, 3], flat_call),
}
json.dump(results, open(r"results\sweep_detection_water.json", "w"), indent=2)