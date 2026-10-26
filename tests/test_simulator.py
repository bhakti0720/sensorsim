from pathlib import Path
import numpy as np
import yaml
from sensorsim.forward.simulator import (make_sensor_state, expose, simulate_dark_frames,
                                         simulate_flat_frames)

CFG = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs" / "sensor_baseline.yaml"))
K = CFG["sensor"]["gain_e_per_dn"]
SHAPE = (32, 32)


def good(state):
    m = state.defects
    return ~(m["dead"] | m["hot"] | m["stuck"])


def test_same_seed_same_frame_different_tag_differs():
    st = make_sensor_state(SHAPE, CFG)
    L = np.full(SHAPE, 100.0)
    assert np.array_equal(expose(L, st, CFG, 5), expose(L, st, CFG, 5))
    assert not np.array_equal(expose(L, st, CFG, 5), expose(L, st, CFG, 6))


def test_defects_are_fixed_pattern_across_frames():
    st = make_sensor_state((256, 256), CFG)
    d = st.defects
    assert d["hot"].sum() > 0 and d["dead"].sum() > 0 and d["stuck"].sum() > 0
    L = np.full((256, 256), 0.0)
    for tag in (1, 2):
        dn = expose(L, st, CFG, tag)
        assert (dn[d["dead"]] == 0).all()
        assert (dn[d["stuck"]] == d["stuck_dn"][d["stuck"]]).all()
        ref = np.median(dn[good(st)])
        assert (dn[d["hot"]].astype(float) - ref > 60).all()     # ~90 DN excess


def test_dark_frames_offset_and_noise():
    st = make_sensor_state(SHAPE, CFG)
    f = simulate_dark_frames(st, CFG, 100).astype(np.float64)    # float before statistics!
    g = good(st)
    assert abs(f[:, g].mean() - CFG["sensor"]["offset_dn"]) < 0.5
    expected_std = np.sqrt((CFG["sensor"]["read_noise_e"] / K) ** 2 + 1 / 12)
    pooled = np.sqrt(f.var(axis=0, ddof=1)[g].mean())
    assert abs(pooled - expected_std) / expected_std < 0.05


def test_flat_mean_matches_theory():
    st = make_sensor_state(SHAPE, CFG)
    lev, stacks = simulate_flat_frames(st, CFG, [0.5], 20)
    expected = CFG["sensor"]["offset_dn"] + 0.5 * (2 ** CFG["sensor"]["bits"] - 1 - CFG["sensor"]["offset_dn"])
    assert abs(stacks[0].astype(float)[:, good(st)].mean() - expected) / expected < 0.005


def test_photon_transfer_slope_is_one_over_K():
    """The exact measurement calibration will do later, run here on known truth."""
    st = make_sensor_state(SHAPE, CFG)
    _, stacks = simulate_flat_frames(st, CFG, [0.2, 0.8], 200)
    g = good(st)
    m, v = [], []
    for s in stacks:
        s = s.astype(np.float64)
        m.append(s.mean(axis=0)[g].mean())
        v.append(s.var(axis=0, ddof=1)[g].mean())
    slope = (v[1] - v[0]) / (m[1] - m[0])
    assert abs(slope - 1 / K) / (1 / K) < 0.03