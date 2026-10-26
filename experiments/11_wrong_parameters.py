"""Experiment 11: how do deliberately wrong calibration values reach the image?

Both scenes, one seed (the perturbation is deterministic).
"""
import json

import numpy as np
import yaml

from sensorsim.pipeline import run_pipeline

cfg = yaml.safe_load(open(r"configs\sensor_baseline.yaml"))
scenes = {
    "urban": np.load(r"results\scene_nyc_truth_radiance.npy"),
    "water": np.load(r"results\scene_water_truth_radiance.npy"),
}

AXES = {
    "dark_offset_dn": [0, 0.25, 0.5, 1, 2, 5],
    "scale_err": [0, 0.001, 0.005, 0.01, 0.02, 0.05],
    "response_err_rms": [0, 0.0005, 0.001, 0.005, 0.01, 0.014],
}

results = {}
for axis, values in AXES.items():
    print(f"\n=== {axis} ===")
    print(f"{'value':>9s}" + "".join(f"{n + ' RMSE':>13s}{n + ' bias':>13s}" for n in scenes))
    rows = []
    for v in values:
        row = {"value": v}
        line = f"{v:>9}"
        for name, L in scenes.items():
            out = run_pipeline(cfg, L, perturb={axis: v})
            row[name] = {"rmse": out["rmse_l1_fixed"], "bias": out["bias_fixed"],
                         "mean": out["mean_radiance"]}
            line += f"{out['rmse_l1_fixed']:>13.4f}{out['bias_fixed']:>13.4f}"
        rows.append(row)
        print(line)
    results[axis] = rows

json.dump(results, open(r"results\wrong_parameters.json", "w"), indent=2)