#%%
import seaborn as sns
import h5py
import numpy as np
import matplotlib.pyplot as plt
import json
from heteromodes.utils import get_project_root
import nibabel as nib

PROJ_DIR = get_project_root()

id = 55
n_nulls = 500
hmap_label = "myelinmap"

#%%
# Load homogeneous and myelinmap results
hmap_labels = ["None", hmap_label]
edge_fc_model, node_fc_model, fcd_model, obj_model = {}, {}, {}, {}
for hmap_label in hmap_labels:
    file = f"{PROJ_DIR}/results/human/model_rest/group/id-{id}/crossval/{hmap_label}/best_model.h5"

    with h5py.File(file, 'r') as f:
        edge_fc_model[hmap_label] = np.mean(np.array(f['results']['edge_fc_corr']).flatten())
        node_fc_model[hmap_label] = np.mean(np.array(f['results']['node_fc_corr']).flatten())
        fcd_model[hmap_label] = np.mean(np.array(f['results']['fcd_ks']).flatten())

    obj_model[hmap_label] = edge_fc_model[hmap_label] + node_fc_model[hmap_label] + 1 - fcd_model[hmap_label]

## Load null results
edge_fc_null, node_fc_null, fcd_null, obj_null = [], [], [], []
# Loop through null files
for i in range(n_nulls):
    # Append edge fc, node fc and fcd
    file = f"{PROJ_DIR}/results/human/model_rest/group/id-{id}/crossval/{hmap_label}/nulls/null-{i}/best_model.h5"

    try:
        with h5py.File(file, 'r') as f:
            edge_fc_null.append(np.mean(np.array(f['results']['edge_fc_corr']).flatten()))
            node_fc_null.append(np.mean(np.array(f['results']['node_fc_corr']).flatten()))
            fcd_null.append(np.mean(np.array(f['results']['fcd_ks']).flatten()))
    except:
        print(f"File {file} not found")
        continue

# Convert to numpy arrays and average across runs/folds (each row)
edge_fc_null = np.array(edge_fc_null)
node_fc_null = np.array(node_fc_null)
fcd_null = np.array(fcd_null)
obj_null = edge_fc_null + node_fc_null + 1 - fcd_null

#%%
# Plot distributions of edge fc, node fc and fcd. Plot vertical line marking homogeneous and myelinmap models
fig, axs = plt.subplots(1, 4, figsize=(20, 3))

edge_fc_null_plot = edge_fc_null[:n_nulls] - edge_fc_model["None"]
node_fc_null_plot = node_fc_null[:n_nulls] - node_fc_model["None"]
fcd_null_plot = fcd_null[:n_nulls] - fcd_model["None"]
obj_null_plot = obj_null[:n_nulls] - obj_model["None"]

p_edge = np.sum(edge_fc_null_plot > (edge_fc_model[hmap_label] - edge_fc_model["None"])) / len(edge_fc_null_plot)
axs[0].hist(edge_fc_null_plot, bins=30, alpha=0.5, color='gray', label='Null')
axs[0].axvline(edge_fc_model[hmap_label] - edge_fc_model["None"], color='red', label="T1w/T2w")
axs[0].set_title(f"$r_{{edge}}$ difference\n(p = {p_edge:.2e})", fontsize=16)
axs[0].set_xlabel(f"$r_{{edge}}$ difference\n(heterogeneous - homogeneous)")
axs[0].set_ylabel("Count")
axs[0].legend()

p_node = np.sum(node_fc_null_plot > (node_fc_model[hmap_label] - node_fc_model["None"])) / len(node_fc_null_plot)
axs[1].hist(node_fc_null_plot.flatten(), bins=30, alpha=0.5, color='gray', label='Null')
axs[1].axvline(node_fc_model[hmap_label] - node_fc_model["None"], color='red', label="T1w/T2w")
axs[1].set_title(f"$r_{{node}}$ difference\n(p = {p_node:.2e})", fontsize=16)
axs[1].set_xlabel(f"$r_{{node}}$ difference\n(heterogeneous - homogeneous)")
axs[1].set_ylabel("Count")
axs[1].legend()

p_fcd = np.sum(fcd_null_plot > (fcd_model[hmap_label] - fcd_model["None"])) / len(fcd_null_plot)
axs[2].hist(fcd_null_plot, bins=30, alpha=0.5, color='gray', label='Null')
axs[2].axvline(fcd_model[hmap_label] - fcd_model["None"], color='red', label="T1w/T2w")
axs[2].set_title(f"$FCD_{{KS}}$ difference\n(p = {p_fcd:.2e})", fontsize=16)
axs[2].set_xlabel(f"$FCD_{{KS}}$ difference\n(heterogeneous - homogeneous)")
import matplotlib as mpl
axs[2].xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.3f'))
axs[2].set_ylabel("Count")
axs[2].legend()

p_obj = np.sum(obj_null_plot > (obj_model[hmap_label]) - obj_model["None"]) / len(obj_null_plot)
axs[3].hist(obj_null_plot.flatten(), bins=30, alpha=0.5, color='gray', label='Null')
axs[3].axvline(obj_model[hmap_label] - obj_model["None"], color='red', label='T1w/T2w')
axs[3].set_title(f"Overall fit difference\n(p = {p_obj:.2e})", fontsize=16)
axs[3].set_xlabel("Combined metric difference\n(heterogeneous - homogeneous)")
axs[3].set_ylabel("Count")
axs[3].legend()

#%%
null_maps = np.load(f"{PROJ_DIR}/data/nulls/human/data-{hmap_labels[1]}_space-fsLR_den-4k_hemi-L_nmodes-500_nnulls-1000_nulls_resample-True.npy")
medmask = nib.load(f"{PROJ_DIR}/data/empirical/human/space-fsLR_den-4k_hemi-L_desc-nomedialwall.func.gii").darrays[0].data.astype(bool)
print(null_maps.shape)

#%%
corrs = np.corrcoef(null_maps[:500, medmask])
triu_inds = np.triu_indices(corrs.shape[0], k=1)

# Plot histogram of correlations
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(corrs[triu_inds].flatten(), bins=30, alpha=0.5, color='gray', label='Null correlations')
ax.axvline(np.mean(corrs), color='blue', label='Mean')
ax.set_title(f"Correlation across nulls", fontsize=16)
ax.set_xlabel("Correlation")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.show()

#%%
# from neuromaps.datasets import fetch_fslr
# from nsbtools.plotting import plot_brain
# import matplotlib as mpl

# mpl.rcParams['figure.dpi'] = 300

# surf = fetch_fslr("4k")["inflated"][0]
# n_nulls_plot = 3

# fig, axs = plt.subplots(n_nulls_plot, 1, figsize=(3, n_nulls_plot * 1.25))
# fig.subplots_adjust(wspace=0.01, hspace=0.01)
# for i in range(n_nulls_plot):
#     color_range = [np.nanpercentile(null_maps[i, medmask], 5), np.nanpercentile(null_maps[i, medmask], 95)]
#     plot_brain(surf,null_maps[i, :], color_range=color_range, cbar=True,
#                     cbar_kws=dict(fontsize=25, aspect=20, pad=0.01, shrink=1.0, n_ticks=2), cmap="turbo", ax=axs[i])
    