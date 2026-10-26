"""Experiment 07: the same pipeline on the urban and water scenes."""
import json

import numpy as np
import yaml

from sensorsim.pipeline import run_pipeline

cfg = yaml.safe_load(open(r"configs\sensor_baseline.yaml"))
scenes = {
    "urban (NYC)": r"results\scene_nyc_truth_radiance.npy",
    "water (LI Sound)": r"results\scene_water_truth_radiance.npy",
}

results = {}
for name, path in scenes.items():
    out = run_pipeline(cfg, np.load(path))
    out["rmse_pct_of_mean"] = 100 * out["rmse_l1_fixed"] / out["mean_radiance"]
    results[name] = out

keys = ["mean_radiance", "rmse_l1_fixed", "rmse_pct_of_mean", "bias_fixed",
        "k_err_pct", "flat_response_rms_err", "det_f1", "n_true_defects"]
print(f"{'metric':24s}" + "".join(f"{n:>20s}" for n in results))
for k in keys:
    print(f"{k:24s}" + "".join(f"{results[n][k]:>20.5f}" for n in results))

json.dump(results, open(r"results\two_scenes.json", "w"), indent=2)