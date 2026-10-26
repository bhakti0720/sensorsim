"""Experiment 08: does scene-based destriping help on smooth water?"""
import numpy as np
import yaml

from sensorsim.pipeline import run_pipeline
from sensorsim.correction.destripe import destripe

cfg = yaml.safe_load(open(r"configs\sensor_baseline.yaml"))
L = np.load(r"results\scene_water_truth_radiance.npy")

metrics, im = run_pipeline(cfg, L, return_images=True)
img, truth, valid = im["l1_fixed"], im["truth_blur"], im["valid"]


def score(x):
    r = np.where(valid, x - truth, np.nan)
    rmse = float(np.sqrt(np.nanmean(r ** 2)))
    stripe = float(np.nanstd(np.nanmean(r, axis=0)))   # spread of column means
    return rmse, stripe


print(f"{'variant':20s}{'RMSE':>10s}{'stripe RMS':>12s}")
print(f"{'no destripe':20s}{score(img)[0]:>10.4f}{score(img)[1]:>12.4f}")
for mode in ("offset", "gain"):
    out = destripe(img, mode=mode, valid=valid)   # CHECK: return type
    r, s = score(out)
    print(f"{'destripe ' + mode:20s}{r:>10.4f}{s:>12.4f}")