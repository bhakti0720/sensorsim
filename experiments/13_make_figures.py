"""Experiment 13: make the README figures from saved result files.

No simulation here. Inputs: results/*.json. Outputs: results/fig_*.png.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

R = Path("results")
sweep = json.load(open(R / "sweep_water.json"))
det = json.load(open(R / "sweep_detection_water.json"))
wrong = json.load(open(R / "wrong_parameters.json"))
bench = json.load(open(R / "benchmark_cpp.json"))
cfg = yaml.safe_load(open(r"configs\sensor_baseline.yaml"))
WATER_PIXELS = 256 * 256


def col(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(R / name, dpi=130)
    plt.close(fig)
    print("saved", R / name)


# ---- Figure 1: sensitivity sweep (water scene, mean of 3 seeds) ----
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))

rows = sweep["read_noise_e"]
x, y, e = col(rows, "value"), col(rows, "rmse_l1_fixed"), col(rows, "rmse_std")
ax[0].errorbar(x, y, yerr=e, marker="o", capsize=3, label="measured")
slope, icpt = np.polyfit(x ** 2, y ** 2, 1)
xs = np.linspace(x.min(), x.max(), 200)
ax[0].plot(xs, np.sqrt(icpt + slope * xs ** 2), "--", label="quadrature fit")
ax[0].set(xlabel="read noise (e-)", ylabel="RMSE vs blurred truth", title="Read noise")
ax[0].legend(fontsize=8)

rows = sweep["bits"]
ax[1].semilogy(col(rows, "value"), col(rows, "rmse_l1_fixed"), marker="o")
ax[1].set(xlabel="ADC bits (full-well electrons held fixed)", ylabel="RMSE",
          title="Bit depth")

rows = sweep["defect_factor"]
pct = col(rows, "n_true_defects") / WATER_PIXELS * 100
ax[2].plot(pct, col(rows, "rmse_l1_fixed"), marker="o")
ax[2].set(xlabel="defective pixels (%)", ylabel="RMSE",
          title=f"Defect rate (min F1 = {col(rows, 'det_f1').min():.2f})")
save(fig, "fig_09_sensitivity.png")

# ---- Figure 2: calibration accuracy vs number of frames ----
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))

rows = det["n_per_level"]
n = col(rows, "value")
ax[0].loglog(n, col(rows, "abs_k_err_pct"), marker="o")
ax[0].set(xlabel="frames per light level", ylabel="|K error| (%)", title="Gain K")

flat = col(rows, "flat_response_rms_err")
ref = flat[np.argmax(n)] * np.sqrt(n.max() / n)
ax[1].loglog(n, flat, marker="o", label="measured")
ax[1].loglog(n, ref, "--", label="1/sqrt(N) from N = max")
ax[1].set(xlabel="frames per light level", ylabel="flat-field RMS error",
          title="Flat-field (PRNU)")
ax[1].legend(fontsize=8)

rows = det["n_dark"]
ax[2].semilogx(col(rows, "value"), col(rows, "rmse_l1_fixed"), marker="o")
ax[2].set(xlabel="dark frames", ylabel="RMSE vs blurred truth", title="Dark frames")
save(fig, "fig_10_calibration_vs_frames.png")

# ---- Figure 3: where hot-pixel detection breaks ----
fig, ax = plt.subplots(figsize=(6, 4))
rows = det["hot_excess_e"]
x = col(rows, "value")
for key, label in (("det_recall", "recall"), ("det_precision", "precision"),
                   ("det_f1", "F1")):
    ax.semilogx(x, col(rows, key), marker="o", label=label)
thr = 6 * cfg["sensor"]["read_noise_e"] / np.sqrt(50)
ax.axvline(thr, color="gray", linestyle=":",
           label=f"6 sigma / sqrt(50 darks) = {thr:.0f} e-")
ax.invert_xaxis()
ax.set(xlabel="hot-pixel excess charge (e-)", ylabel="score", ylim=(0, 1.05),
       title="Hot-pixel detection limit (water, 3 seeds)")
ax.legend(fontsize=8, loc="lower left")
save(fig, "fig_11_detection_limit.png")

# ---- Figure 4: wrong calibration parameters ----
titles = {"dark_offset_dn": "Dark level error (DN)",
          "scale_err": "Radiometric scale error (fraction)",
          "response_err_rms": "Response map error (rms fraction)"}
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for a, (axis, rows) in zip(ax, wrong.items()):
    x = col(rows, "value")
    for scene, color in (("urban", "tab:red"), ("water", "tab:blue")):
        a.plot(x, [r[scene]["rmse"] for r in rows], marker="o", color=color,
               label=f"{scene} RMSE")
        a.plot(x, [abs(r[scene]["bias"]) for r in rows], "--", color=color,
               label=f"{scene} |bias|")
    a.set(xlabel=titles[axis], ylabel="radiance units")
    a.legend(fontsize=7)
save(fig, "fig_12_wrong_parameters.png")

# ---- Figure 5: C++ vs NumPy ----
fig, ax = plt.subplots(figsize=(7, 4))
names = [r["case"] for r in bench]
sp = [r["speedup"] for r in bench]
ax.barh(names[::-1], sp[::-1])
for i, v in enumerate(sp[::-1]):
    ax.text(v + 0.1, i, f"{v:.1f}x", va="center", fontsize=8)
ax.set(xlabel="speedup over vectorised NumPy", title="C++ fix_defects")
save(fig, "fig_13_cpp_speedup.png")