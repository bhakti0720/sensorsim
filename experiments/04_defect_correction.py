"""Experiment 04: score defect fixing and destriping on the NYC scene.

Calibration and detection use frames only. Truth is used here for scoring.
Everything is scored against the BLURRED truth over ALL pixels (defect pixels included).
"""
import json
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt

from sensorsim.forward.optics import apply_psf
from sensorsim.forward.simulator import (
    make_sensor_state, simulate_dark_frames, simulate_flat_frames,
)
from sensorsim.calibration.dark import estimate_dark
from sensorsim.calibration.flatfield import estimate_flatfield
from sensorsim.calibration.apply import estimate_radiometric_scale, apply_calibration
from sensorsim.correction.defect_detect import detect_defects
from sensorsim.correction.defect_fix import fix_defects
from sensorsim.correction.destripe import destripe

N_DARK = 50
N_PER_LEVEL = 30
LEVEL_FRACTIONS = np.linspace(0.03, 0.80, 10)

cfg = yaml.safe_load(open(r"configs\sensor_baseline.yaml"))
truth = np.load(r"results\truth_nyc.npz")
dn = np.load(r"results\scene_nyc_L0.npy")
L_true = np.load(r"results\scene_nyc_truth_radiance.npy")
shape = L_true.shape
L_blur = apply_psf(L_true, cfg["sensor"]["psf_sigma_px"])          # scoring reference

# ---- calibration frames -> calibrate -> detect (frames only) ----
state = make_sensor_state(shape, cfg)
darks = simulate_dark_frames(state, cfg, N_DARK)
levels, flats = simulate_flat_frames(state, cfg, LEVEL_FRACTIONS, N_PER_LEVEL)

dark_cal = estimate_dark(darks)
ff = estimate_flatfield(flats, dark_cal.dark_level_dn)
scale = estimate_radiometric_scale(levels, flats, dark_cal.dark_level_dn)
masks = detect_defects(darks, flats, dark_cal.dark_level_dn, ff.response)

# ---- the two calibrated images ----
full = apply_calibration(dn, dark_cal.dark_level_dn, ff.response, scale)       # NaN at dead/stuck
offset_only = (dn.astype(np.float64) - dark_cal.dark_level_dn) / scale         # float64 first

m_const = masks.constant | ~np.isfinite(full)           # dead + stuck (+ anything NaN)
m_all = masks.combined | ~np.isfinite(full)             # everything detected

# ---- table 1 variants (full calibration) ----
t1 = {
    "Nodata = 0 (no fix)": np.nan_to_num(full, nan=0.0),
    "Nodata = scene mean": np.where(np.isfinite(full), full, np.nanmean(full)),
    "Fix dead/stuck only": fix_defects(full, m_const),
    "Fix all detected": fix_defects(full, m_all),
}
t1["Fix all + destripe"] = destripe(t1["Fix all detected"], mode="gain", valid=~m_all)

# ---- table 2 variants (NO flat-field: destriper as a fallback) ----
off_fixed = fix_defects(offset_only, m_all)
t2 = {
    "Offset-only + fix": off_fixed,
    "  + destripe (gain)": destripe(off_fixed, mode="gain", valid=~m_all),
    "  + destripe (offset)": destripe(off_fixed, mode="offset", valid=~m_all),
    "Full cal + fix (ref)": t1["Fix all detected"],
}

# ---- scoring ----
true_defect = (truth["dead"].astype(bool) | truth["stuck"].astype(bool)
               | truth["hot"].astype(bool))
ok = np.isfinite(L_true)


def score(e):
    fin = ok & np.isfinite(e)
    err = np.where(fin, e - L_blur, np.nan)
    d = err[fin & true_defect]
    c = err[fin & ~true_defect]
    return {
        "rmse": float(np.sqrt(np.nanmean(err ** 2))),
        "rmse_pct": float(100 * np.sqrt(np.nanmean(err ** 2)) / L_blur[fin].mean()),
        "bias": float(np.nanmean(err)),
        "rmse_defect_px": float(np.sqrt(np.mean(d ** 2))),
        "rmse_clean_px": float(np.sqrt(np.mean(c ** 2))),
        "stripe_rms": float(np.nanstd(np.nanmean(err, axis=0))),
    }


def table(title, variants):
    res = {k: score(v) for k, v in variants.items()}
    print(title)
    hdr = (f"{'Variant':<24}{'RMSE':>8}{'RMSE %':>8}{'Bias':>8}"
           f"{'Defect px':>11}{'Clean px':>10}{'Stripe':>9}")
    print(hdr)
    print("-" * len(hdr))
    for k, r in res.items():
        print(f"{k:<24}{r['rmse']:>8.3f}{r['rmse_pct']:>8.2f}{r['bias']:>+8.3f}"
              f"{r['rmse_defect_px']:>11.3f}{r['rmse_clean_px']:>10.3f}{r['stripe_rms']:>9.4f}")
    print()
    return res


print(f"true defects: {int(true_defect.sum())}  detected: {int(masks.combined.sum())}  "
      f"NaN after calibration: {int((~np.isfinite(full)).sum())}\n")
r1 = table("=== Table 1: full calibration + correction (vs blurred truth, all pixels) ===", t1)
r2 = table("=== Table 2: no flat-field, destriper as fallback ===", t2)

z, m = r1["Nodata = 0 (no fix)"], r1["Nodata = scene mean"]
f_dead, f_all = r1["Fix dead/stuck only"], r1["Fix all detected"]
print(f"RMSE improvement of fixing (dead/stuck) vs zero-fill:  {100 * (1 - f_dead['rmse'] / z['rmse']):.1f} %")
print(f"RMSE improvement of fixing (dead/stuck) vs mean-fill:  {100 * (1 - f_dead['rmse'] / m['rmse']):.1f} %")
print(f"Fix-all vs fix-dead/stuck-only RMSE: {f_all['rmse']:.4f} vs {f_dead['rmse']:.4f} "
      f"({'fix-all better' if f_all['rmse'] < f_dead['rmse'] else 'dead/stuck-only better'})")

# ---- figure ----
out = Path("results"); out.mkdir(exist_ok=True)
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
ax[0].imshow(t1["Nodata = 0 (no fix)"] - L_blur, cmap="coolwarm", vmin=-10, vmax=10)
ax[0].set_title("Error: nodata = 0 (no fix)")
ax[1].imshow(t1["Fix all detected"] - L_blur, cmap="coolwarm", vmin=-10, vmax=10)
ax[1].set_title("Error: defects interpolated")
for a in ax[:2]:
    a.axis("off")
for name in ("Offset-only + fix", "  + destripe (gain)", "Full cal + fix (ref)"):
    ax[2].plot(np.nanmean(t2[name] - L_blur, axis=0), lw=0.8, label=name.strip())
ax[2].set_title("Column-mean error"); ax[2].set_xlabel("column"); ax[2].legend()
plt.tight_layout(); plt.savefig(out / "fig_04_defect_correction.png", dpi=150)

np.save(out / "scene_nyc_L1_corrected.npy", t1["Fix dead/stuck only"])
json.dump({"table1": r1, "table2": r2,
           "n_true_defects": int(true_defect.sum()), "n_detected": int(masks.combined.sum())},
          open(out / "correction_nyc.json", "w"), indent=2)