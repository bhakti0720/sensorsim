"""Experiment 02: score the full radiometric calibration on the NYC scene.

Calibration uses frames only. The truth file and config are used here for
scoring and for the oracle row only.

Two references:
  vs sharp truth    : includes the PSF blur (which no radiometric step can undo)
  vs blurred truth  : truth passed through the sensor PSF -> isolates radiometric error
"""
import json
from pathlib import Path

import numpy as np
import yaml

from sensorsim.forward.optics import apply_psf
from sensorsim.forward.simulator import (
    make_sensor_state, simulate_dark_frames, simulate_flat_frames,
)
from sensorsim.calibration.dark import estimate_dark
from sensorsim.calibration.flatfield import estimate_flatfield
from sensorsim.calibration.apply import estimate_radiometric_scale, apply_calibration

N_DARK = 50
N_PER_LEVEL = 30
LEVEL_FRACTIONS = np.linspace(0.03, 0.80, 10)

cfg = yaml.safe_load(open(r"configs\sensor_baseline.yaml"))
truth = np.load(r"results\truth_nyc.npz")
dn = np.load(r"results\scene_nyc_L0.npy")
L_true = np.load(r"results\scene_nyc_truth_radiance.npy")
shape = L_true.shape

# reference scenes (scoring only)
L_blur = apply_psf(L_true, cfg["sensor"]["psf_sigma_px"])

# ---- calibration frames (the only thing calibration may see) ----
state = make_sensor_state(shape, cfg)
darks = simulate_dark_frames(state, cfg, N_DARK)
levels, flats = simulate_flat_frames(state, cfg, LEVEL_FRACTIONS, N_PER_LEVEL)

# ---- calibrate (no config, no truth) ----
dark_cal = estimate_dark(darks)
ff = estimate_flatfield(flats, dark_cal.dark_level_dn)
scale = estimate_radiometric_scale(levels, flats, dark_cal.dark_level_dn)

dn_f = dn.astype(np.float64)                      # float64 BEFORE any subtraction
dark_level = dark_cal.dark_level_dn

# ---- the methods ----
est = {
    "No correction": dn_f / scale,
    "Offset-only": (dn_f - dark_level) / scale,
    "Full pipeline": apply_calibration(dn, dark_level, ff.response, scale),
}

# ---- oracle (true parameters): scoring only, shows the noise floor ----
off_true = float(truth["offset_dn"])
dark_true = off_true + np.asarray(truth["col_offset_dn"])[None, :]
resp_true = (1.0 + truth["prnu_g"]) * (1.0 + np.asarray(truth["col_gain"]))[None, :]
resp_true = resp_true / np.median(resp_true)
scale_true = (2 ** cfg["sensor"]["bits"] - 1 - off_true) / cfg["sensor"]["l_max"]
est["Oracle (floor)"] = (dn_f - dark_true) / (resp_true * scale_true)

# ---- pixels to score: exclude defects, saturated, nodata/NaN ----
defect = np.zeros(shape, dtype=bool)
for key in truth.files:
    if any(s in key for s in ("dead", "hot", "stuck")) and key != "stuck_dn":
        defect |= truth[key].astype(bool)
saturated = dn >= 2 ** cfg["sensor"]["bits"] - 1
good = ~defect & ~saturated & np.isfinite(L_true) & np.isfinite(est["Full pipeline"])


def score(e, ref):
    err = (e - ref)[good]
    t = ref[good]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    err_img = np.where(good, e - ref, np.nan)
    col_mean_err = np.nanmean(err_img, axis=0)          # one number per column
    return {
        "rmse": rmse,
        "rmse_pct": 100 * rmse / float(t.mean()),
        "bias": float(err.mean()),
        "snr_db": float(10 * np.log10(np.sum(t ** 2) / np.sum(err ** 2))),
        "stripe_rms": float(np.nanstd(col_mean_err)),
    }


res_sharp = {name: score(e, L_true) for name, e in est.items()}
res_blur = {name: score(e, L_blur) for name, e in est.items()}


def print_table(title, results):
    print(title)
    hdr = f"{'Method':<16}{'RMSE':>9}{'RMSE %':>9}{'Bias':>9}{'SNR dB':>9}{'Stripe RMS':>13}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in results.items():
        print(f"{name:<16}{r['rmse']:>9.3f}{r['rmse_pct']:>9.2f}{r['bias']:>+9.3f}"
              f"{r['snr_db']:>9.2f}{r['stripe_rms']:>13.4f}")
    full, none, oracle = results["Full pipeline"], results["No correction"], results["Oracle (floor)"]
    print(f"RMSE improvement, full vs none: {100 * (1 - full['rmse'] / none['rmse']):.1f} %")
    print(f"RMSE of full pipeline vs oracle floor: x{full['rmse'] / oracle['rmse']:.3f}\n")


print(f"scored pixels: {good.sum()} of {good.size} "
      f"(defects excluded: {defect.sum()}, saturated excluded: {saturated.sum()})")
print(f"mean true radiance (scored pixels): {L_true[good].mean():.2f}")
print(f"radiometric scale: estimated {scale:.4f} DN/unit, true {scale_true:.4f}, "
      f"error {100 * (scale - scale_true) / scale_true:+.3f} %\n")
print_table("=== A) vs SHARP truth (includes PSF blur error) ===", res_sharp)
print_table("=== B) vs BLURRED truth (radiometric + noise error only) ===", res_blur)

# ---- save ----
out = Path("results"); out.mkdir(exist_ok=True)
np.save(out / "scene_nyc_L1_full.npy", est["Full pipeline"])      # input for Day 7 defect work
np.save(out / "scene_nyc_truth_blurred.npy", L_blur)              # reference for Day 7 scoring
json.dump({"scale_est": scale, "scale_true": scale_true,
           "n_scored": int(good.sum()),
           "vs_sharp_truth": res_sharp, "vs_blurred_truth": res_blur},
          open(out / "baselines_nyc.json", "w"), indent=2)