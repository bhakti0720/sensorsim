"""Experiment 06: load the water scene, check it, save its truth radiance."""
import numpy as np
import matplotlib.pyplot as plt

from sensorsim.scene.scene import load_scene

L, valid = load_scene(r"configs\scene_water.yaml")

print("shape                ", L.shape)
if L.size == 0:
    raise SystemExit("Empty crop: the centre is outside the tile. Move lon/lat.")

print("valid fraction       ", float(valid.mean()))
print("radiance min/mean/max", float(np.nanmin(L)), float(np.nanmean(L)), float(np.nanmax(L)))
print("NaN count            ", int(np.isnan(L).sum()))
print("above l_max 484      ", int((L > 484).sum()))
print("std / mean           ", float(np.nanstd(L) / np.nanmean(L)))

np.save(r"results\scene_water_truth_radiance.npy", L)

fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
im = ax[0].imshow(L, cmap="gray")
ax[0].set_title("Water scene: true radiance")
plt.colorbar(im, ax=ax[0], fraction=0.046)
ax[1].hist(L[valid].ravel(), bins=80)
ax[1].set_title("Radiance histogram")
plt.tight_layout()
plt.savefig(r"results\fig_06_water_scene.png", dpi=120)
plt.show()