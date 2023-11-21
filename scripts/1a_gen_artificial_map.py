import json
import numpy as np
from brainspace.mesh import mesh_io
from scipy.spatial.distance import cdist


config_file = "/fs04/kg98/vbarnes/HeteroModes/scripts/config.json"
with open(config_file, encoding="UTF-8") as f:
    config = json.load(f)

space = config["space"]
den = config["den"]
surf_type = config["surf"]
hemi = config["hemi"]
atlas = config["atlas"]
n_modes = config["n_modes"]
mask_medial = config["mask_medial"]
emode_dir = config["emode_dir"]

surf_dir = "/fs03/kg98/vbarnes/surfaces"
surf = mesh_io.read_surface(f"{surf_dir}/atlas-{atlas}_space-{space}_den-{den}_"
                            f"surf-{surf_type}_hemi-{hemi}_surface.vtk")
medial = np.loadtxt(f"{surf_dir}/atlas-{atlas}_space-{space}_den-{den}_hemi-{hemi}_"
                    f"medialMask.txt").astype(bool)

# ===== White noise map =====
# hetero_map = np.random.normal(0, 1, surf.Points.shape[0])

# ===== Gradient map based on x, y, and z coordinates =====
# map = np.where((surf.Points[:, 2] < -55) & (surf.Points[:, 2] > 55), 2.0, 1.0)
# np.savetxt(f"./data/hetero-DVpatches_surf-{surf_type}.txt", hetero_map)

# ===== Map with 2 guassian kernels =====
kernel_radius = 750
kernel_intensity = 1
x1 = cdist([[-55, 25, 20]], surf.Points) # [ml, ap, dv]
x2 = cdist([[-55, -75, 10]], surf.Points)
hetero_map = (np.exp(-x1**2 / kernel_radius) * kernel_intensity).flatten() + (np.exp(-x2**2 / kernel_radius) * kernel_intensity).flatten()
np.savetxt(f"./data/hetero-patches2_surf-{surf_type}.txt", hetero_map)

# plot_surf_template(hetero_map, "fsLR", "32k", hemi="L")
# plt.savefig("brain.png")
