# Physics-Based Push-Broom Sensor Simulator and Radiometric Calibration Pipeline

*Final project plan: portfolio project for the GalaxEye Image Processing Intern application*

## 1. Project summary

Simulate a push-broom satellite sensor, generate raw data with known errors, calibrate and correct that data using only *estimated* parameters, then measure how close the result is to the known truth.

**In scope:** forward sensor model, calibration estimation, defect and stripe correction, one sensitivity study, one C++ module.

**Out of scope (state this in the README):** DEM orthorectification, RPC models, and atmospheric correction (L1C to L2A).

**Why this scope:** the project is deliberately narrow so it can be finished, tested and explained line by line. Sentinel-2 L1C is already orthorectified, and a simulator with known truth makes radiometric errors measurable, which is the strongest part of the story.

## 2. How it maps to the job description

| Resume point | Covered? | Notes |
| --- | --- | --- |
| Radiometric correction (dark, flat, gain/offset), Python/C++ | Yes | Dark frames, photon transfer curve, flat-field, plus a C++ module |
| Geometric correction / orthorectification | No | Out of scope |
| Sensor simulator (PSF, quantisation, shot/read noise, SNR, MTF) | Mostly | MTF50 is optional; drop "MTF" from the resume if you skip it |
| Defect and stripe correction with metrics | Yes | Precision, recall, F1, RMSE |
| Atmospheric correction (L1C to L2A) | No | Out of scope |
| Validation metrics, unit tests, documentation | Mostly | Geometric accuracy not covered |
| C++ routines with measured speedup | Yes | One function, benchmarked against vectorised NumPy |

## 3. End-to-end flow

1. **Scene:** load a Sentinel-2 L1C crop (TOA reflectance) and convert it to TOA radiance.
2. **Forward model:** optics blur, electrons, PRNU, dark signal, shot and read noise, ADC, defects, column stripes. Output is simulated L0 (raw DN) plus saved truth parameters.
3. **Calibration frames:** the simulator also generates 50+ dark frames and flat frames at 8+ light levels, each with its own noise.
4. **Calibration:** estimate offset, dark current, read noise, gain K (photon transfer curve) and the PRNU map from those frames only. Never read the simulator config.
5. **Apply:** convert scene DN to calibrated radiance (roughly an L1 radiometric product).
6. **Correction:** detect defects, interpolate them, destripe.
7. **Metrics against truth:** RMSE, bias, SNR, F1, calibration error.
8. **Experiments:** calibration recovery, defect correction, sensitivity sweep.
9. **Output:** results table, figures, README.

## 4. Folder structure

```
sensorsim/
├── README.md              # pipeline diagram, results table, assumptions, limitations
├── pyproject.toml
├── configs/
│   ├── sensor_baseline.yaml     # all sensor params, citations in comments
│   └── experiments/             # sweep configs
├── data/                  # gitignored; download instructions only
├── src/sensorsim/
│   ├── scene/       load.py, radiance.py
│   ├── forward/     optics.py, detector.py, noise.py, defects.py, adc.py, simulator.py
│   ├── calibration/ dark.py, flatfield.py, photon_transfer.py, apply.py
│   ├── correction/  defect_detect.py, defect_fix.py, destripe.py
│   ├── metrics/     radiometric.py, defects.py, image_quality.py
│   ├── cpp/         one function + pybind11
│   └── utils/       config, seeding, logging
├── experiments/     01_calibration_recovery.py, 02_defect_correction.py, 03_sensitivity_sweep.py
├── tests/           pytest (+ GoogleTest for C++)
├── notebooks/       figures only, no logic
└── results/         figures and tables (committed), run metadata
```

**Rule:** all logic lives in `src/`. Experiments and notebooks only call it.

## 5. Forward model

Work in **electrons**, not DN, until the ADC stage.

1. **Radiance from truth:** L = rho \* E\_sun \* cos(theta\_s) / (pi \* d^2), where rho is Sentinel-2 L1C TOA reflectance per band.
2. **Optics:** L\_blur = L convolved with the PSF. Start with a Gaussian whose width is a fraction of a pixel, since the reference is already blurred.
3. **Signal electrons:** S = R \* L\_blur \* t\_int \* (1 + g), where g \~ N(0, sigma\_PRNU) is multiplicative and fixed per detector.
4. **Dark signal:** D = (d0 + delta) \* t\_int, where delta is DSNU, additive and fixed per detector.
5. **Shot and dark noise:** E \~ Poisson(S + D).
6. **Read noise:** E = E + N(0, sigma\_read^2).
7. **ADC:** DN = clip(round(E / K + offset), 0, 2^bits - 1), with K in electrons per DN.
8. **Defects:** dead pixels (DN = 0), hot pixels (high dark current), stuck pixels (constant value).
9. **Stripes:** in a push-broom sensor each detector element produces one image column, so a gain or offset error in one detector appears as a vertical stripe. Model stripes as per-column gain and offset, not as random lines painted on.

## 6. Inverse pipeline (estimate, never peek)

| Step | Input | What it estimates |
| --- | --- | --- |
| Dark calibration | 50+ dark frames | per-pixel offset, dark current, read noise |
| Photon transfer curve | flat frames at 8+ light levels | gain K (slope of variance vs. mean) |
| Flat-field | flat frames | per-pixel PRNU map |
| Apply | scene L0 | DN to calibrated radiance |
| Defect detection | dark and flat frames, then scene | defect mask |
| Correction | mask, scene | interpolated, destriped image |

Calibration frames carry their own noise, so the estimates are imperfect, as in real life.

## 7. Validation plan

- **Baselines in every table:** no correction, offset-only, full pipeline.
- **Radiometric:** RMSE, bias and SNR against the known truth radiance.
- **Calibration accuracy:** estimated K and PRNU against true values, as a function of the number of calibration frames.
- **Defects:** precision, recall and F1 at several defect rates.
- **Sensitivity sweep (3 axes):** read noise, bit depth, defect percentage against residual error.
- **Calibration-error plot:** feed deliberately wrong parameters and show how the error propagates.
- **Scenes:** 2 (urban and water). Phrase claims as "validated on urban and water scenes".
- **Optional if time allows:** MTF50 via slanted edge, Spectral Angle Mapper (SAM) for spectral fidelity, a downsampling step to model detector sampling.

## 8. Schedule (about 11 days)

| Days | Deliverable | Gate to pass |
| --- | --- | --- |
| 1-2 | Repo skeleton, config YAML, load crop, reflectance to radiance | Radiance values in a plausible range |
| 3-4 | Full forward model and tests | Shot-noise variance matches the mean (in electrons) |
| 5-6 | Dark, photon transfer and flat-field calibration | K and PRNU recovered within a few % of truth |
| 7 | Defect detection, interpolation, destriping | F1 and RMSE reported vs. no correction |
| 8 | Sensitivity sweep and plots | Reproducible from YAML and seed |
| 9-10 | C++ module via pybind11 (e.g. defect interpolation), benchmark | Output matches Python; speedup measured vs. vectorised NumPy |
| 11 | README, figures, results table, test cleanup | Runs from the README alone |

**Cut to save time:** denoising module, 4 scenes reduced to 2, five experiment scripts reduced to three, MTF50 made optional, geometry module dropped.

**If time gets tight:** simplify the C++ benchmark before dropping simulator or calibration validation.

## 9. Breakpoints to watch

1. **Simulator sanity first.** If simulated noise does not match theory (variance equals mean for shot noise, in electrons), everything downstream is meaningless. Test this before writing any correction code.
2. **Units.** Keep radiance, electrons and DN in clearly named variables. Unit mix-ups are the most common silent bug.
3. **Sentinel-2 scaling.** Check the scale factor and the baseline offset in the metadata, since it changed in 2022.
4. **Clouds and nodata.** Mask them before computing any metric.
5. **Seeds and truth.** Save the seed and true parameters with every run.
6. **No cheating.** Calibration must estimate from frames, not read the config.
7. **Saturation.** Saturated pixels cannot be recovered. Exclude them from error metrics or report them separately.
8. **Sensor parameters.** Use values from a published sensor and cite them in the YAML comments.
9. **C++ scope.** One small, correct, tested function with an honest benchmark against vectorised NumPy.
10. **Order of operations.** Forward and inverse steps must mirror each other; blur before or after noise changes results a lot.

## 10. Speed tips

- Use a small crop (about 512 x 512) so every run takes seconds.
- Write the test right after each module.
- Pick the C++ function early.
- Use generated boilerplate for config loading and plotting, but write the physics and validation yourself so you can explain every line.
- Freeze the scope once you start. New ideas go in a "future work" section of the README.

## 11. Resume bullets (fill in numbers only from real runs)

- Built a modular Python/C++ pipeline performing radiometric correction (dark subtraction, flat-field, gain/offset estimation), validated against known ground-truth radiance with residual error of \[X\]%.
- Developed a physics-based push-broom sensor simulator (PSF blur, PRNU, dark current, shot and read noise, quantisation) to study how sensor parameters affect SNR.
- Detected and corrected defective pixels and column striping, reaching F1 of \[X\] and improving RMSE by \[X\]%.
- Wrote a C++ \[function\] with pybind11, achieving \[X\]x speedup over vectorised NumPy.
- Defined validation metrics and wrote unit tests and documentation so the pipeline can be reused.

**Honesty notes:**

- Say "validated against known ground-truth radiance", not "validated against a reference product", because validation is against simulator truth.
- Do not claim orthorectification or atmospheric correction unless you build them.
- List signal processing, optics, linear algebra and probability coursework explicitly on the resume.

## 12. Public repo checklist

- README with pipeline diagram, results table, assumptions and limitations
- Out-of-scope items stated plainly
- Unit tests that pass from a clean clone
- At least three result figures (calibration recovery, defect correction, sensitivity sweep)
- Config YAML with citations for every sensor parameter
- One command to reproduce each experiment from config and seed

## 13. Optional extension

A minimal geometry module using per-line shifts estimated independently with phase correlation. It shows registration understanding without needing a DEM, and is the natural next step if you finish early.
