"""End-to-end run: simulate -> calibrate -> detect -> fix -> score.

One call = one full experiment. The sweep only edits cfg and calls this.
Calibration uses frames only. Truth (state) is used ONLY for scoring.
cfg is the plain dict from yaml.safe_load.

perturb (optional dict) deliberately corrupts the ESTIMATED calibration
before it is applied, to study error propagation:
  dark_offset_dn   : added to the estimated dark level (DN)
  scale_err        : fractional error on the radiometric scale
  response_err_rms : random per-pixel error on the response map
"""
from __future__ import annotations

import numpy as np

from sensorsim.forward.optics import apply_psf
from sensorsim.forward.simulator import (
    make_sensor_state,
    expose,
    simulate_dark_frames,
    simulate_flat_frames,
)
from sensorsim.calibration.dark import estimate_dark
from sensorsim.calibration.photon_transfer import estimate_gain
from sensorsim.calibration.flatfield import estimate_flatfield
from sensorsim.calibration.apply import (
    estimate_radiometric_scale,
    apply_calibration,
)
from sensorsim.correction.defect_detect import detect_defects
from sensorsim.correction.defect_fix import fix_defects


def _rmse(a, b, valid):
    d = a[valid].astype(np.float64) - b[valid].astype(np.float64)
    return float(np.sqrt(np.mean(d * d)))


def _prf(pred, truth):
    tp = float(np.sum(pred & truth))
    fp = float(np.sum(pred & ~truth))
    fn = float(np.sum(~pred & truth))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def run_pipeline(cfg, radiance, n_dark=50, n_levels=10, n_per_level=30,
                 return_images=False, perturb=None):
    """Run the full pipeline once; return a flat dict of metrics."""
    radiance = np.asarray(radiance, dtype=np.float64)
    shape = radiance.shape
    s = cfg["sensor"]
    perturb = perturb or {}

    # ---- 1. Forward model: sensor, scene, calibration frames ----
    state = make_sensor_state(shape, cfg)
    scene_dn = expose(radiance, state, cfg, 0)
    darks = simulate_dark_frames(state, cfg, n_dark)
    level_fractions = np.linspace(0.03, 0.80, n_levels)
    levels, flats = simulate_flat_frames(state, cfg, level_fractions, n_per_level)

    # ---- 2. Calibration (frames only) ----
    dark = estimate_dark(darks)
    ptc = estimate_gain(flats, dark.dark_level_dn)
    flat = estimate_flatfield(flats, dark.dark_level_dn)
    scale = estimate_radiometric_scale(levels, flats, dark.dark_level_dn)

    # ---- 2b. Optional deliberate corruption of the estimates ----
    dark_level_used = dark.dark_level_dn + perturb.get("dark_offset_dn", 0.0)
    scale_used = scale * (1.0 + perturb.get("scale_err", 0.0))
    response_used = flat.response
    if perturb.get("response_err_rms", 0.0) > 0:
        prng = np.random.default_rng([cfg["seed"], 9])
        response_used = flat.response * (
            1.0 + perturb["response_err_rms"] * prng.standard_normal(flat.response.shape))

    l1 = apply_calibration(scene_dn, dark_level_used, response_used, scale_used)

    # ---- 3. Detect and fix (dead/stuck only, as found best) ----
    masks = detect_defects(darks, flats, dark.dark_level_dn, flat.response)
    fix_mask = masks.constant | ~np.isfinite(l1)
    l1_fixed = fix_defects(l1, fix_mask)

    # ---- 4. Scoring (truth used only here) ----
    truth_blur = apply_psf(radiance, s["psf_sigma_px"])
    dead = state.defects["dead"].astype(bool)
    stuck = state.defects["stuck"].astype(bool)
    hot = state.defects["hot"].astype(bool)
    true_defects = dead | stuck | hot
    saturated = scene_dn >= (2 ** s["bits"] - 1)
    valid = np.isfinite(l1_fixed) & ~saturated

    precision, recall, f1 = _prf(masks.combined, true_defects)

    # true relative response, built the same way save_truth builds gain_total
    good = ~true_defects
    resp_true = (1 + state.prnu_g) * (1 + state.col_gain)[None, :]
    resp_true = resp_true / np.median(resp_true[good])
    resp_err = float(np.sqrt(np.mean((flat.response[good] - resp_true[good]) ** 2)))

    k_est = float(ptc.gain_e_per_dn)   # use whichever field name worked for you
    k_true = float(s["gain_e_per_dn"])

    out = {
        "rmse_l1_fixed": _rmse(l1_fixed, truth_blur, valid),
        "bias_fixed": float(np.mean(l1_fixed[valid] - truth_blur[valid])),
        "mean_radiance": float(np.mean(truth_blur[valid])),
        "k_est": k_est,
        "k_true": k_true,
        "k_err_pct": 100.0 * (k_est - k_true) / k_true,
        "flat_response_rms_err": resp_err,
        "det_precision": precision,
        "det_recall": recall,
        "det_f1": f1,
        "n_true_defects": int(true_defects.sum()),
        "n_saturated": int(saturated.sum()),
        "n_dark": n_dark,
        "n_levels": n_levels,
        "n_per_level": n_per_level,
    }
    if return_images:
        return out, {"l1_fixed": l1_fixed, "truth_blur": truth_blur, "valid": valid}
    return out