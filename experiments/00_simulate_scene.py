from pathlib import Path
import numpy as np
import yaml
import matplotlib.pyplot as plt
from sensorsim.scene.scene import load_scene
from sensorsim.forward.simulator import make_sensor_state, expose, save_truth

cfg = yaml.safe_load(open(r"configs\sensor_baseline.yaml"))
L, valid = load_scene(r"configs\scene_nyc.yaml")

state = make_sensor_state(L.shape, cfg)
dn = expose(L, state, cfg, frame_tag=0)

out = Path("results"); out.mkdir(exist_ok=True)
np.save(out / "scene_nyc_L0.npy", dn)
np.save(out / "scene_nyc_truth_radiance.npy", L)
save_truth(state, cfg, out / "truth_nyc.npz")

# expected DN if the sensor were perfect and noiseless
K, off = cfg["sensor"]["gain_e_per_dn"], cfg["sensor"]["offset_dn"]
ideal = off + L / cfg["sensor"]["l_max"] * (2 ** cfg["sensor"]["bits"] - 1 - off)

print("DN min/mean/max:", dn.min(), dn.mean(), dn.max())
print("ideal DN min/mean/max:", ideal.min(), ideal.mean(), ideal.max())
print("saturated pixels:", (dn >= 4095).sum())
print("mean error vs ideal (DN):", (dn.astype(float) - ideal).mean())

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(L, cmap="gray"); ax[0].set_title("True radiance")
ax[1].imshow(dn, cmap="gray"); ax[1].set_title("Simulated raw DN (L0)")
ax[2].imshow(dn.astype(float) - ideal, cmap="coolwarm", vmin=-60, vmax=60)
ax[2].set_title("Raw minus ideal (DN)")
for a in ax: a.axis("off")
plt.tight_layout(); plt.savefig(out / "fig_00_simulated_scene.png", dpi=150)