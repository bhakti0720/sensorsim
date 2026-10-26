"""Experiment 01: can calibration recover the sensor parameters from frames only?

Calibration functions get frames only. The truth file is used here for scoring.
"""
import json
from pathlib import Path

import numpy as np
import yaml

from sensorsim.forward.simulator import (
    make_sensor_state, simulate_dark_frames, simulate_flat_frames,
)
from sensorsim.calibration.dark import estimate_dark
from sensorsim.calibration.photon_transfer import estimate_gain
from sensorsim.calibration.flatfield import estimate_flatfield

N_DARK = 50
N_PER_LEVEL = 30
LEVEL_FRACTIONS = np.linspace(0.03, 0.80, 10)   # fractions of l_max
GATE_K_PCT = 3.0                                # gate: K within a few %
GATE_FF_PCT = 0.5                               # gate: flat-field RMS error (% of response)

cfg = yaml.safe_load(open(r"configs\sensor_baseline.yaml"))
truth = np.load(r"results\truth_nyc.npz")
print("truth keys:", sorted(truth.files))

shape = truth["prnu_g"].shape
state = make_sensor_state(shape, cfg)           # same seed -> same fixed pattern as truth

# ---- generate calibration frames (this is all calibration is allowed to see)
darks = simulate_dark_frames(state, cfg, N_DARK)
levels, flats = simulate_flat_frames(state, cfg, LEVEL_FRACTIONS, N_PER_LEVEL)

# ---- calibrate (no config, no truth)
dark_cal = estimate_dark(darks)
ptc = estimate_gain(flats, dark_cal.dark_level_dn)
ff = estimate_flatfield(flats, dark_cal.dark_level_dn)

# ---- score against truth
K_true = float(truth["gain_e_per_dn"])
off_true = float(truth["offset_dn"])
read_true = float(truth["read_noise_e"])

# per-pixel defect mask from truth (dead / hot / stuck), used only to exclude them from scoring
defect = np.zeros(shape, dtype=bool)
for key in truth.files:
    if any(s in key for s in ("dead", "hot", "stuck")) and key != "stuck_dn":
        defect |= truth[key].astype(bool)

# true dark level per pixel = offset + column offset (dark current is ~0.075 e-, negligible)
dark_true = off_true + np.asarray(truth["col_offset_dn"])[None, :]
dark_err = (dark_cal.dark_level_dn - dark_true)[~defect]

K_err_pct = 100 * (ptc.gain_e_per_dn - K_true) / K_true
read_err_pct = 100 * (ptc.read_noise_e - read_true) / read_true

# true relative response = (1 + prnu) * (1 + column gain), normalised to median 1
resp_true = (1.0 + truth["prnu_g"]) * (1.0 + np.asarray(truth["col_gain"]))[None, :]
resp_true = resp_true / np.median(resp_true)

ff_err = (ff.response - resp_true)[~defect]
ff_rms = float(np.sqrt(np.mean(ff_err ** 2)))
nonuni_before = float(np.std(resp_true[~defect]))   # what the sensor does without correction
col_true = np.median(resp_true, axis=0)
col_true = col_true / np.median(col_true)
col_rms = float(np.sqrt(np.mean((ff.column_gain - col_true) ** 2)))

# read noise from darks: remove rounding noise (1/12 DN^2) before converting to electrons
rn_dn = dark_cal.read_noise_dn_median
rn_dark_e = ptc.gain_e_per_dn * np.sqrt(max(rn_dn ** 2 - 1.0 / 12.0, 0.0))

print("\n=== Dark calibration ===")
print(f"frames: {dark_cal.n_frames}")
print(f"dark level RMS error (non-defect px): {np.sqrt(np.mean(dark_err**2)):.3f} DN "
      f"(theory ~0.14), bias {dark_err.mean():+.3f} DN")
print(f"read noise from darks (median): {rn_dn:.3f} DN "
      f"= {rn_dn * ptc.gain_e_per_dn:.2f} e- raw, {rn_dark_e:.2f} e- after removing "
      f"rounding noise (true {read_true:.1f} e-)")

print("\n=== Photon transfer ===")
print(f"levels used: {ptc.n_levels_used} of {len(LEVEL_FRACTIONS)}, R2 = {ptc.r2:.5f}")
print(f"K estimated: {ptc.gain_e_per_dn:.3f} +/- {ptc.gain_stderr:.3f}  true: {K_true:.3f}  "
      f"error: {K_err_pct:+.2f} %")
print(f"read noise from PTC: {ptc.read_noise_e:.2f} e-  true: {read_true:.2f}  "
      f"error: {read_err_pct:+.1f} %")

print("\n=== Flat-field ===")
print(f"levels used: {ff.n_levels_used}")
print(f"true non-uniformity (std of response): {100 * nonuni_before:.3f} %")
print(f"response RMS error after calibration:  {100 * ff_rms:.3f} %   "
      f"(improvement x{nonuni_before / ff_rms:.0f})")
print(f"column gain RMS error: {100 * col_rms:.3f} %")

passed = abs(K_err_pct) <= GATE_K_PCT and 100 * ff_rms <= GATE_FF_PCT
print(f"\nGATE (K within {GATE_K_PCT}% and flat-field RMS within {GATE_FF_PCT}%): "
      f"{'PASS' if passed else 'FAIL'}")

out = Path("results"); out.mkdir(exist_ok=True)
json.dump({
    "K_est": ptc.gain_e_per_dn, "K_true": K_true, "K_err_pct": K_err_pct,
    "read_noise_ptc_e": ptc.read_noise_e, "read_noise_true_e": read_true,
    "dark_rms_err_dn": float(np.sqrt(np.mean(dark_err ** 2))),
    "ff_rms_err": ff_rms, "nonuniformity_before": nonuni_before,
    "col_gain_rms_err": col_rms,
    "n_dark": N_DARK, "n_per_level": N_PER_LEVEL, "gate_pass": bool(passed),
}, open(out / "calibration_recovery.json", "w"), indent=2)