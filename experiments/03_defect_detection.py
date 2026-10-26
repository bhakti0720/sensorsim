"""Experiment 03: score calibration-time defect detection against the truth masks.

Detection uses frames only. The truth file is used here for scoring.
"""
import json
from pathlib import Path

import numpy as np
import yaml

from sensorsim.forward.simulator import (
    make_sensor_state, simulate_dark_frames, simulate_flat_frames,
)
from sensorsim.calibration.dark import estimate_dark
from sensorsim.calibration.flatfield import estimate_flatfield
from sensorsim.correction.defect_detect import detect_defects

N_DARK = 50
N_PER_LEVEL = 30
LEVEL_FRACTIONS = np.linspace(0.03, 0.80, 10)

cfg = yaml.safe_load(open(r"configs\sensor_baseline.yaml"))
truth = np.load(r"results\truth_nyc.npz")
shape = truth["prnu_g"].shape

# ---- calibration frames (the only thing detection may see) ----
state = make_sensor_state(shape, cfg)
darks = simulate_dark_frames(state, cfg, N_DARK)
levels, flats = simulate_flat_frames(state, cfg, LEVEL_FRACTIONS, N_PER_LEVEL)

# ---- calibrate + detect (no config, no truth) ----
dark_cal = estimate_dark(darks)
ff = estimate_flatfield(flats, dark_cal.dark_level_dn)
masks = detect_defects(darks, flats, dark_cal.dark_level_dn, ff.response)

# ---- one type per pixel, constant first, then hot, then response outlier ----
pred_const = masks.constant
pred_hot = masks.hot & ~pred_const
pred_resp = masks.response_outlier & ~pred_const & ~pred_hot
pred_all = masks.combined

true_dead_stuck = truth["dead"].astype(bool) | truth["stuck"].astype(bool)
true_hot = truth["hot"].astype(bool) & ~true_dead_stuck
true_all = true_dead_stuck | true_hot


def prf(pred, true):
    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


results = {
    "dead+stuck (constant rule)": prf(pred_const, true_dead_stuck),
    "hot (dark-level rule)": prf(pred_hot, true_hot),
    "all defects (combined)": prf(pred_all, true_all),
}

print(f"image: {shape[0]}x{shape[1]} = {shape[0] * shape[1]} pixels")
print(f"true defects: dead {int(truth['dead'].sum())}, stuck {int(truth['stuck'].sum())}, "
      f"hot {int(truth['hot'].sum())}  "
      f"(all, overlaps merged: {int(true_all.sum())}, {100 * true_all.mean():.3f} %)")
print(f"response-outlier rule adds {int(pred_resp.sum())} pixels beyond the other two rules\n")

hdr = f"{'Detector':<28}{'TP':>6}{'FP':>6}{'FN':>6}{'Precision':>11}{'Recall':>9}{'F1':>8}"
print(hdr)
print("-" * len(hdr))
for name, r in results.items():
    print(f"{name:<28}{r['tp']:>6}{r['fp']:>6}{r['fn']:>6}"
          f"{r['precision']:>11.4f}{r['recall']:>9.4f}{r['f1']:>8.4f}")

# ---- where were the misses and false alarms? ----
miss = np.argwhere(~pred_all & true_all)
fa = np.argwhere(pred_all & ~true_all)
if len(miss):
    print(f"\nmissed pixels (first 10): {miss[:10].tolist()}")
if len(fa):
    print(f"false alarms (first 10): {fa[:10].tolist()}")

# ---- save ----
out = Path("results"); out.mkdir(exist_ok=True)
np.save(out / "defect_mask_detected.npy", pred_all)       # input for the fix step
json.dump({"n_pixels": int(true_all.size), "n_true_defects": int(true_all.sum()),
           "response_outlier_extra": int(pred_resp.sum()), "scores": results},
          open(out / "defect_detection_nyc.json", "w"), indent=2)