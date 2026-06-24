import json
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import nibabel as nib
from scipy.stats import spearmanr, rankdata

from brainspace.utils.parcellation import reduce_by_labels
from neuromodes.io import fetch_surf
from neuromodes.eigen import EigenSolver
from nsbutils.utils import unmask, unparcellate
from nsbutils.plotting_pyvista import plot_surf, plot_surf_video

from heteromodes.utils import get_project_root, load_hmap

PROJ_DIR = get_project_root()

# Load data
density = "32k"
mesh, medmask = fetch_surf(density=density)
mesh_plot = {"v": mesh.vertices, "t": mesh.faces} # for plotting
hetero_label = "SAaxis"
aniso_label = "SAaxis"
map = load_hmap(hetero_label, density=density)

# Load HCP-MMP1 parcellation
parc_file = f"{PROJ_DIR}/data/parcellations/parc-hcpmmp1_space-fsLR_den-{density}_hemi-L.label.gii"
parc = nib.load(parc_file)

# Create mask for V1
label_to_key = {lab.label: lab.key for lab in parc.labeltable.labels}
V1_mask = parc.darrays[0].data == label_to_key['L_V1_ROI']

# Load visual hierarchy oi labels
vis_hierarchy_file = f"{PROJ_DIR}/data/parcellations/17_visual_cortical_hierarchy_rois.npy"
vis_hierarchy_roi_labels = np.load(vis_hierarchy_file).astype(int) - 1
print(np.unique_values(vis_hierarchy_roi_labels))

# Parcellate myelinmap (proxy for hierarchy)
myelinmap = load_hmap("myelinmap", density=density)
t1t2_vals_parc = reduce_by_labels(myelinmap[medmask], parc.darrays[0].data[medmask])
print(t1t2_vals_parc.shape)
t1t2_vals_hierarchy = t1t2_vals_parc[vis_hierarchy_roi_labels]
print(t1t2_vals_hierarchy)

# Simulation parameters
dt = 1e-4
nt = 1250
r = 28.9
gamma = 116
stimulation_amplitude = 20.0
stimulation_indices = [10, 20]

n_modes = 5000

# Create a 10 ms external input with amplitude 20.0 to V1
ext_input = np.zeros((mesh.vertices.shape[0], nt))
ext_input[V1_mask, stimulation_indices[0]:stimulation_indices[1]] = stimulation_amplitude
ext_input = ext_input[medmask, :]
print(ext_input.shape)

alpha = 0 #-1.6
beta = 0 #36

# Run wave model and parcellate
print("Running wave model...")
t1 = time.time()
solver = EigenSolver(
    surf=mesh,
    mask=medmask,
    hetero=map,
    alpha=alpha,
    aniso_map=map,
    beta=beta
).solve(n_modes=n_modes, seed=365)
print(f"Finished running wave model in {(time.time() - t1)/3600:.2f} hours.")
neural = solver.simulate_waves(
    ext_input=ext_input,
    nt=nt,
    dt=dt,
    r=r,
    gamma=gamma,
)
neural_parc = reduce_by_labels(
    neural, parc.darrays[0].data[medmask], axis=1
)

print("Saving results...")
config = {
    "hetero_label": "SAaxis",
    "aniso_label": "SAaxis",
    "alpha": alpha,
    "beta": beta,
    "r": r,
    "gamma": gamma,
    "nt": nt,
    "dt": dt,
    "stimulation_amplitude": stimulation_amplitude,
    "stimulation_indices": stimulation_indices,
    "n_modes": n_modes,
}
with open(f"{PROJ_DIR}/scripts/figures/aniso/fig2/config_alpha-{alpha:.1f}_beta-{beta:.1f}.json", "w") as f:
    json.dump(config, f, indent=4)

np.save(f"{PROJ_DIR}/scripts/figures/aniso/fig2/neural_alpha-{alpha:.1f}_beta-{beta:.1f}.npy", neural)
