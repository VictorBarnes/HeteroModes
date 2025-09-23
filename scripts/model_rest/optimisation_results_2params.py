# %%
import os
import json
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="white")

PROJ_DIR = "/fs04/kg98/vbarnes/HeteroModes"
with open(f"{PROJ_DIR}/scripts/model_rest/results_config.json", "r") as f:
    config = json.load(f)

# id = 0
# hmap_labels = ["None", "myelinmap"]
# hmap_labels_plotting = ["None", "T1w/T2w"]
species = "macaque"
id = 3
hmap_label = "myelinmap"
evaluation = "fit"
param1 = "alpha"        # if using alpha, it must be param1
param2 = "r"

default_params = {
    'alpha': 0.0,
    'r': 18.0,
    'beta': 1.0,
    'gamma': 0.116
}

results_dir = f"{PROJ_DIR}/results/{species}/model_rest/group/id-{id}/{evaluation}"

# %%
param_combs = pd.read_csv(f"{results_dir}/{hmap_label}/parameter_combinations.csv")
param1_vals = param_combs[param1].unique()
param2_vals = param_combs[param2].unique().round(3)

edge_fc_land = np.full((len(param1_vals), len(param2_vals)), np.nan)
node_fc_land = np.full((len(param1_vals), len(param2_vals)), np.nan)
fcd_land = np.full((len(param1_vals), len(param2_vals)), np.nan)
combined_land = np.full((len(param1_vals), len(param2_vals)), np.nan)

for i, p1 in enumerate(param1_vals):
    for j, p2 in enumerate(param2_vals):        
        # Update parameters based on param1 and param2
        params = default_params.copy()
        params[param1] = p1
        params[param2] = p2
        
        # Extracdefault_paramsvidual parameters for file naming
        alpha = params['alpha']
        beta = params['beta']
        r = params['r']
        gamma = params['gamma']
        
        file = f"{results_dir}/{hmap_label}/model_alpha-{alpha}_r-{r}_gamma-{gamma:.3f}_beta-{beta}.h5"
        if not os.path.exists(file):
            print(f"File {file} does not exist, skipping...")
            continue

        # Load the HDF5 file
        with h5py.File(file, "r") as f:
            edge_fc_land[i, j] = np.mean(f['results']['edge_fc_corr'][:]) if evaluation == "crossval" else f['results']['edge_fc_corr'][()]
            node_fc_land[i, j] = np.mean(f['results']['node_fc_corr'][:]) if evaluation == "crossval" else f['results']['node_fc_corr'][()]
            fcd_land[i, j] = np.mean(f['results']['fcd_ks'][:]) if evaluation == "crossval" else f['results']['fcd_ks'][()]
            combined_land[i, j] = edge_fc_land[i, j] + node_fc_land[i, j] + 1 - fcd_land[i, j]

# Insert homogeneous results
if param1 == 'alpha':
    edge_fc_hom = np.full(len(param2_vals), np.nan)
    node_fc_hom = np.full(len(param2_vals), np.nan)
    fcd_hom = np.full(len(param2_vals), np.nan)
    combined_hom = np.full(len(param2_vals), np.nan)
    for j, p2 in enumerate(param2_vals):
        params = default_params.copy()
        params[param2] = p2
        
        # Extracdefault_paramsvidual parameters for file naming
        alpha = params['alpha']
        beta = params['beta']
        r = params['r']
        gamma = params['gamma']
        
        file = f"{results_dir}/None/model_alpha-{alpha}_r-{r}_gamma-{gamma:.3f}_beta-{beta}.h5"
        with h5py.File(file, 'r') as f:
            edge_fc_hom[j] = np.mean(f['results']['edge_fc_corr'][:]) if evaluation == "crossval" else f['results']['edge_fc_corr'][()]
            node_fc_hom[j] = np.mean(f['results']['node_fc_corr'][:]) if evaluation == "crossval" else f['results']['node_fc_corr'][()]
            fcd_hom[j] = np.mean(f['results']['fcd_ks'][:]) if evaluation == "crossval" else f['results']['fcd_ks'][()]
            combined_hom[j] = edge_fc_hom[j] + node_fc_hom[j] + 1 - fcd_hom[j]

    hom_ind = len(param1_vals) // 2
    edge_fc_land = np.insert(edge_fc_land, hom_ind, edge_fc_hom, axis=0)
    node_fc_land = np.insert(node_fc_land, hom_ind, node_fc_hom, axis=0)
    fcd_land = np.insert(fcd_land, hom_ind, fcd_hom, axis=0)
    combined_land = np.insert(combined_land, hom_ind, combined_hom, axis=0)

    param1_vals = np.insert(param1_vals, hom_ind, 0.0)

# %%
# Plot heatmap
fig, axs = plt.subplots(1, 4, figsize=(18, 3))

sns.heatmap(edge_fc_land, annot=True, fmt=".2f", cmap="viridis", ax=axs[0], square=True,
            xticklabels=param2_vals, yticklabels=param1_vals, cbar_kws={"label": "Edge FC Correlation"}, annot_kws={"size": 6})
axs[0].set_title("Edge FC Correlation")
axs[0].set_xlabel(param2)
axs[0].set_ylabel(param1)

sns.heatmap(node_fc_land, annot=True, fmt=".2f", cmap="viridis", ax=axs[1], square=True,
            xticklabels=param2_vals, yticklabels=param1_vals, cbar_kws={"label": "Node FC Correlation"}, annot_kws={"size": 6})
axs[1].set_title("Node FC Correlation")
axs[1].set_xlabel(param2)
axs[1].set_ylabel(param1)

sns.heatmap(fcd_land, annot=True, fmt=".2f", cmap="viridis", ax=axs[2], square=True,
            xticklabels=param2_vals, yticklabels=param1_vals, cbar_kws={"label": "FCD KS"}, annot_kws={"size": 6})
axs[2].set_title("FCD KS")
axs[2].set_xlabel(param2)
axs[2].set_ylabel(param1)

sns.heatmap(combined_land, annot=True, fmt=".2f", cmap="viridis", ax=axs[3], square=True,
            xticklabels=param2_vals, yticklabels=param1_vals, cbar_kws={"label": "Combined Score"}, annot_kws={"size": 6})
axs[3].set_title("Combined Score")
axs[3].set_xlabel(param2)
axs[3].set_ylabel(param1)

plt.tight_layout()
plt.show()