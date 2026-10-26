"""Forward model driver: optics -> detector -> noise -> ADC -> defects. Saves the truth."""
from dataclasses import dataclass
import numpy as np

from sensorsim.forward.optics import apply_psf
from sensorsim.forward.detector import (responsivity, make_prnu_map, make_dsnu_map,
                                        signal_electrons, dark_electrons)
from sensorsim.forward.noise import apply_noise
from sensorsim.forward.adc import electrons_to_dn
from sensorsim.forward.defects import (make_column_errors, apply_column_gain,
                                       apply_column_offset, make_defect_maps,
                                       apply_hot_pixels, apply_dead_stuck)


@dataclass
class SensorState:
    """Fixed per-detector flaws. This is the ground truth for calibration."""
    shape: tuple
    resp: float            # electrons per (radiance unit * s)
    prnu_g: np.ndarray     # per-pixel gain error
    dsnu: np.ndarray       # per-pixel dark-rate offset (e-/s)
    col_gain: np.ndarray   # per-column gain error
    col_offset_dn: np.ndarray
    defects: dict          # dead / hot / stuck masks + stuck_dn


def make_sensor_state(shape, cfg):
    s, d, st = cfg["sensor"], cfg["defects"], cfg["stripes"]
    rng = np.random.default_rng([cfg["seed"], 0])          # stream 0 = fixed pattern
    prnu = make_prnu_map(shape, s["sigma_prnu"], rng)
    dsnu = make_dsnu_map(shape, s["dsnu_e_per_s"], rng)
    cg, co = make_column_errors(shape[1], st["col_gain_sigma"], st["col_offset_sigma_dn"], rng)
    maps = make_defect_maps(shape, d["dead_rate"], d["hot_rate"], d["stuck_rate"], s["bits"], rng)
    resp = responsivity(s["gain_e_per_dn"], s["bits"], s["offset_dn"], s["l_max"], s["t_int_s"])
    return SensorState(shape, resp, prnu, dsnu, cg, co, maps)


def expose(radiance, state, cfg, frame_tag):
    """One frame: true radiance -> raw DN (uint16). frame_tag picks independent noise."""
    s, d = cfg["sensor"], cfg["defects"]
    if radiance.shape != state.shape:
        raise ValueError("radiance shape must match the sensor state")
    rng = np.random.default_rng([cfg["seed"], 1, frame_tag])   # stream 1 = noise

    l_blur = apply_psf(radiance, s["psf_sigma_px"])                          # optics
    sig = signal_electrons(l_blur, state.resp, s["t_int_s"], state.prnu_g)   # PRNU
    sig = apply_column_gain(sig, state.col_gain)                             # stripes (gain)
    dark = dark_electrons(s["dark_current_e_per_s"], state.dsnu, s["t_int_s"])
    dark = apply_hot_pixels(dark, state.defects["hot"], d["hot_excess_e"])
    e = apply_noise(sig, dark, s["read_noise_e"], rng)                       # shot + read
    e = apply_column_offset(e, state.col_offset_dn, s["gain_e_per_dn"])      # stripes (offset)
    dn = electrons_to_dn(e, s["gain_e_per_dn"], s["offset_dn"], s["bits"])   # ADC
    return apply_dead_stuck(dn, state.defects)                               # dead / stuck


def simulate_dark_frames(state, cfg, n):
    zero = np.zeros(state.shape)
    return np.stack([expose(zero, state, cfg, 1000 + i) for i in range(n)])


def simulate_flat_frames(state, cfg, level_fractions, n_per_level):
    """Uniform light at fractions of l_max. Returns (radiance levels, list of frame stacks)."""
    l_max = cfg["sensor"]["l_max"]
    levels = np.array(level_fractions) * l_max
    stacks = []
    for k, lev in enumerate(levels):
        flat = np.full(state.shape, lev)
        stacks.append(np.stack([expose(flat, state, cfg, 2000 + 100 * k + j)
                                for j in range(n_per_level)]))
    return levels, stacks


def save_truth(state, cfg, path):
    """Save everything calibration must be judged against."""
    s = cfg["sensor"]
    np.savez_compressed(
        path, seed=cfg["seed"], gain_e_per_dn=s["gain_e_per_dn"], offset_dn=s["offset_dn"],
        read_noise_e=s["read_noise_e"], resp=state.resp, prnu_g=state.prnu_g, dsnu=state.dsnu,
        col_gain=state.col_gain, col_offset_dn=state.col_offset_dn,
        gain_total=(1 + state.prnu_g) * (1 + state.col_gain)[None, :],
        dead=state.defects["dead"], hot=state.defects["hot"],
        stuck=state.defects["stuck"], stuck_dn=state.defects["stuck_dn"])