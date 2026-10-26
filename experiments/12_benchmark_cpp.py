"""Experiment 12: benchmark the C++ fix_defects against vectorised NumPy.

Checks the outputs agree on every case before timing it.
"""
import json
import time
from pathlib import Path

import numpy as np

from sensorsim.correction.defect_fix import fix_defects as fix_py
from sensorsim._fixcore import fix_defects as fix_cpp


def median_time(f, img, mask, reps):
    f(img, mask)                               # warm-up
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        f(img, mask)
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


cases = []
rng = np.random.default_rng(0)

real_img = Path(r"results\scene_nyc_L1_full.npy")
real_mask = Path(r"results\defect_mask_detected.npy")
if real_img.exists() and real_mask.exists():
    cases.append(("NYC 512 real mask", np.load(real_img),
                  np.load(real_mask).astype(bool), 20))

for size, reps in ((512, 20), (2048, 5)):
    img = 20 + 30 * rng.random((size, size))
    for rate in (0.0012, 0.01, 0.05):
        cases.append((f"{size}x{size} random {rate * 100:.2f}%", img,
                      rng.random((size, size)) < rate, reps))

print(f"{'case':28s}{'defects':>9s}{'numpy ms':>11s}{'C++ ms':>10s}{'speedup':>9s}")
rows = []
for name, img, mask, reps in cases:
    a = fix_py(img, mask)
    b = fix_cpp(img, mask)
    assert np.allclose(a, b, rtol=1e-12, atol=0), f"outputs differ on {name}"
    tp = median_time(fix_py, img, mask, reps)
    tc = median_time(fix_cpp, img, mask, reps)
    rows.append({"case": name, "n_defects": int(mask.sum()),
                 "numpy_ms": tp * 1e3, "cpp_ms": tc * 1e3, "speedup": tp / tc})
    print(f"{name:28s}{int(mask.sum()):>9d}{tp * 1e3:>11.2f}{tc * 1e3:>10.3f}{tp / tc:>8.1f}x")

json.dump(rows, open(r"results\benchmark_cpp.json", "w"), indent=2)