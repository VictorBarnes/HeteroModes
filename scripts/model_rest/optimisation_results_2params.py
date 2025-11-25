"""
Visualize 2D parameter sweep results for neural field model optimization.

This script generates heatmaps showing model performance across two parameter
dimensions (e.g., alpha vs r), allowing visualization of the parameter landscape
and identification of optimal parameter regions.
"""

# %%
import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from heteromodes.utils import get_project_root

sns.set_theme(style="white")

# Configuration
PROJ_DIR = get_project_root()
species = "human"
id_num = 1
hmap_label = "myelinmap"
evaluation = "crossval"  # 'fit' or 'crossval'

# Parameters to sweep
param1 = "alpha"  # Note: alpha must be param1 if used
param2 = "r"

# Default parameter values (used for parameters not being swept)
default_params = {
    'alpha': 0.0,
    'r': 18.0,
    'beta': 1.0,
    'gamma': 0.116
}

results_dir = f"{PROJ_DIR}/results/{species}/model_rest/group/id-{id_num}/{evaluation}"


# %%
# Load parameter combinations and initialize result matrices
param_combs = pd.read_csv(f"{results_dir}/{hmap_label}/parameter_combinations.csv")
param1_vals = param_combs[param1].unique()
param2_vals = param_combs[param2].unique().round(3)

# Initialize performance matrices
edge_fc_land = np.full((len(param1_vals), len(param2_vals)), np.nan)
node_fc_land = np.full((len(param1_vals), len(param2_vals)), np.nan)
fcd_land = np.full((len(param1_vals), len(param2_vals)), np.nan)
combined_land = np.full((len(param1_vals), len(param2_vals)), np.nan)

# Load results for each parameter combination
for i, p1 in enumerate(param1_vals):
    for j, p2 in enumerate(param2_vals):
        # Create parameter dict with current sweep values
        params = default_params.copy()
        params[param1] = p1
        params[param2] = p2
        
        # Construct filename from parameters
        file_path = (
            f"{results_dir}/{hmap_label}/model_alpha-{params['alpha']}_"
            f"r-{params['r']}_gamma-{params['gamma']:.3f}_beta-{params['beta']}.h5"
        )
        
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist, skipping...")
            continue

        # Load results (average across CV splits if using cross-validation)
        with h5py.File(file_path, "r") as f:
            if evaluation == "crossval":
                edge_fc_land[i, j] = np.mean(f['results']['edge_fc_corr'][:])
                node_fc_land[i, j] = np.mean(f['results']['node_fc_corr'][:])
                fcd_land[i, j] = np.mean(f['results']['fcd_ks'][:])
            else:
                edge_fc_land[i, j] = f['results']['edge_fc_corr'][()]
                node_fc_land[i, j] = f['results']['node_fc_corr'][()]
                fcd_land[i, j] = f['results']['fcd_ks'][()]
            
            # Compute combined metric
            combined_land[i, j] = (
                edge_fc_land[i, j] + node_fc_land[i, j] + (1 - fcd_land[i, j])
            )

# Insert homogeneous model results at alpha=0 if alpha is being swept
if param1 == 'alpha':
    # Initialize homogeneous result arrays
    edge_fc_hom = np.full(len(param2_vals), np.nan)
    node_fc_hom = np.full(len(param2_vals), np.nan)
    fcd_hom = np.full(len(param2_vals), np.nan)
    combined_hom = np.full(len(param2_vals), np.nan)
    
    # Load homogeneous model results for each param2 value
    for j, p2 in enumerate(param2_vals):
        params = default_params.copy()
        params[param2] = p2
        
        file_path = (
            f"{results_dir}/None/model_alpha-{params['alpha']}_"
            f"r-{params['r']}_gamma-{params['gamma']:.3f}_beta-{params['beta']}.h5"
        )
        
        with h5py.File(file_path, 'r') as f:
            if evaluation == "crossval":
                edge_fc_hom[j] = np.mean(f['results']['edge_fc_corr'][:])
                node_fc_hom[j] = np.mean(f['results']['node_fc_corr'][:])
                fcd_hom[j] = np.mean(f['results']['fcd_ks'][:])
            else:
                edge_fc_hom[j] = f['results']['edge_fc_corr'][()]
                node_fc_hom[j] = f['results']['node_fc_corr'][()]
                fcd_hom[j] = f['results']['fcd_ks'][()]
            
            combined_hom[j] = edge_fc_hom[j] + node_fc_hom[j] + (1 - fcd_hom[j])

    # Insert homogeneous results at midpoint of alpha axis
    hom_ind = len(param1_vals) // 2
    edge_fc_land = np.insert(edge_fc_land, hom_ind, edge_fc_hom, axis=0)
    node_fc_land = np.insert(node_fc_land, hom_ind, node_fc_hom, axis=0)
    fcd_land = np.insert(fcd_land, hom_ind, fcd_hom, axis=0)
    combined_land = np.insert(combined_land, hom_ind, combined_hom, axis=0)
    param1_vals = np.insert(param1_vals, hom_ind, 0.0)


# %%
# Generate heatmaps for all performance metrics
fig, axs = plt.subplots(1, 4, figsize=(18, 3))

# Common heatmap settings
heatmap_kwargs = {
    'annot': True,
    'fmt': '.2f',
    'cmap': 'viridis',
    'square': True,
    'xticklabels': param2_vals,
    'yticklabels': param1_vals,
    'annot_kws': {'size': 6}
}

# Edge-level FC
sns.heatmap(
    edge_fc_land, 
    ax=axs[0], 
    cbar_kws={'label': "Pearson's r"},
    **heatmap_kwargs
)
axs[0].set_title("Edge-level FC")
axs[0].set_xlabel(param2)
axs[0].set_ylabel(param1)

# Node-level FC
sns.heatmap(
    node_fc_land, 
    ax=axs[1], 
    cbar_kws={'label': "Pearson's r"},
    **heatmap_kwargs
)
axs[1].set_title("Node-level FC")
axs[1].set_xlabel(param2)
axs[1].set_ylabel(param1)

# FCD KS statistic
sns.heatmap(
    fcd_land, 
    ax=axs[2], 
    cbar_kws={'label': 'KS Statistic'},
    **heatmap_kwargs
)
axs[2].set_title("FCD KS")
axs[2].set_xlabel(param2)
axs[2].set_ylabel(param1)

# Combined score
sns.heatmap(
    combined_land, 
    ax=axs[3], 
    cbar_kws={'label': 'Combined Score'},
    **heatmap_kwargs
)
axs[3].set_title("Combined Score")
axs[3].set_xlabel(param2)
axs[3].set_ylabel(param1)

plt.tight_layout()
plt.show()
