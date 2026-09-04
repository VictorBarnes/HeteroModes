#%% Imports and globals
import json
import h5py
import numpy as np
from heteromodes.utils import get_project_root
from nsbutils.plotting_pyvista import plot_surf
import matplotlib.pyplot as plt
import nibabel as nib
from neuromodes.io import read_surf
from nsbutils.utils import unmask

PROJ_DIR = get_project_root()
RESULTS_DIR = PROJ_DIR / "results" / "model_rest" / "human" / "hcp-ep"
ID = 1

#%% Load FC for each model and compute node-level accuracy
hmap_labels = ["None", "myelinmap"]
hmap_config = PROJ_DIR / "data" / "heteromaps" / "human" / "heteromaps_config.json"
cohorts = ["hc", "scz"]

node_fc_acc = {}
for cohort in cohorts:
    n_subj = 45 if cohort == "hc" else 74
    emp_fc_file = f"{PROJ_DIR}/data/empirical/human/hcp-ep_run-1-2_nsubj-{n_subj}_des-fc_space-fsLR_den-4k_hemi-L_nt-820.h5"
    with h5py.File(emp_fc_file, "r") as f:
        fc_emp = f["group_fc"][:]
    for hmap_label in hmap_labels:
        # Get cache key for best run
        if hmap_label != "None":
            hmap = f"hcpep-{cohort}-{hmap_label}"
        else:
            hmap = hmap_label
        best_json = f"{RESULTS_DIR}/{cohort}/{ID}/hetero-{hmap}_aniso-None/best.json"
        with open(best_json) as f:
            best = json.load(f)
        cache_key = best["cache_key"]

        # Load FC from cache
        eval_dir = f"{RESULTS_DIR}/{cohort}/{ID}/hetero-{hmap}_aniso-None/evals"
        cache_file = f"{eval_dir}/{cache_key}_model_outputs.npz"

        fc_model = np.load(cache_file)["fc"]
        node_fc_acc[(cohort, hmap_label)] = np.array(
            [np.corrcoef(fc_emp[i, :], fc_model[i, :])[0, 1] for i in range(fc_emp.shape[0])]
        )


#%% Plot node-level accuracy on brain

surf = read_surf(
    f"{PROJ_DIR}/data/empirical/human/space-fsLR_den-4k_hemi-L_desc-midthickness.surf.gii"
)
surf = {"vertices": surf.vertices, "faces": surf.faces}
medmask = nib.load(
    f"{PROJ_DIR}/data/empirical/human/hcp-ep_space-fsLR_den-4k_hemi-L_desc-nomedialwall.func.gii"
).darrays[0].data.astype(bool)

node_fc_acc_stacked = np.stack(
    [node_fc_acc[(cohort, hmap_label)] for cohort in cohorts for hmap_label in hmap_labels], 
    axis=0
)
clims = np.min(node_fc_acc_stacked), np.max(node_fc_acc_stacked)
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
for i, cohort in enumerate(cohorts):
    for j, hmap_label in enumerate(hmap_labels):
        acc = node_fc_acc[(cohort, hmap_label)]
        plot_surf(
            surf={"lh": surf},
            data={"lh": unmask(acc, medmask)},
            rois={"lh": medmask},
            views=["lateral", "medial"],
            ax=axs[i, j],
            cmap="RdBu_r",
            clim=clims
        )


