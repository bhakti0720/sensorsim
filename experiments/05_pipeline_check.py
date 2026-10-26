"""Experiment 05: run_pipeline on the real NYC scene; must match 04."""
import json

import numpy as np
import yaml

from sensorsim.pipeline import run_pipeline

cfg = yaml.safe_load(open(r"configs\sensor_baseline.yaml"))
L_true = np.load(r"results\scene_nyc_truth_radiance.npy")

out = run_pipeline(cfg, L_true)
for k, v in out.items():
    print(f"{k:24s} {v}")

json.dump(out, open(r"results\pipeline_check_nyc.json", "w"), indent=2)