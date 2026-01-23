"""
Visualize and analyze neural field model optimization results.

This script loads model optimization results, compares different heterogeneity
maps, and generates visualization plots for model performance metrics and
optimal parameters.
"""

# %%
import os
import json
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from heteromodes.utils import get_project_root

sns.set_theme(style="white")
PROJ_DIR = get_project_root()

# %%
# Configuration
species = "human"
ids = {63: '63'}
evaluation = "crossval"

# Load heterogeneity map labels
with open(f"{PROJ_DIR}/data/heteromaps/{species}/heteromaps_config.json", "r") as f:
    config = json.load(f)
    hmap_labels = {key: val["label"] for key, val in config.items()}
hmap_labels["None"] = "Homogeneous"


# %%
# Load results from all heterogeneity maps
edge_fc_data, node_fc_data, fcd_data = [], [], []
alpha_data, r_data = [], []

for id_num in ids.keys():
    results_dir = f"{PROJ_DIR}/results/{species}/model_rest/group/id-{id_num}/{evaluation}"
    
    # Load evaluation metrics from config
    with open(f"{PROJ_DIR}/results/{species}/model_rest/group/id-{id_num}/config.json", "r") as f:
        metrics = json.load(f)["metrics"]
    
    for hmap_label in hmap_labels.keys():
        file = f"{results_dir}/{hmap_label}/best_model.h5"

        if not os.path.exists(file):
            print(f"File {file} does not exist, skipping...")
            continue

        with h5py.File(file, 'r') as f:
            # Extract metric values (handles both single values and cross-validation splits)
            edge_fc_values = np.array(f['results']['edge_fc_corr']).flatten()
            node_fc_values = np.array(f['results']['node_fc_corr']).flatten()
            fcd_values = np.array(f['results']['fcd_ks']).flatten()
            
            # Extract parameter values
            alpha_values = np.array(f['alpha']).flatten()
            r_values = np.array(f['r']).flatten()
            
            # Store data (one entry per cross-validation split)
            for i in range(len(edge_fc_values)):
                edge_fc_data.append({
                    'hmap_label': hmap_label, 
                    'id': id_num, 
                    'value': edge_fc_values[i]
                })
                node_fc_data.append({
                    'hmap_label': hmap_label, 
                    'id': id_num, 
                    'value': node_fc_values[i]
                })
                fcd_data.append({
                    'hmap_label': hmap_label, 
                    'id': id_num, 
                    'value': fcd_values[i]
                })
                alpha_data.append({
                    'hmap_label': hmap_label, 
                    'id': id_num, 
                    'value': alpha_values[i]
                })
                r_data.append({
                    'hmap_label': hmap_label, 
                    'id': id_num, 
                    'value': r_values[i]
                })

# Create DataFrames
edge_fc_df = pd.DataFrame(edge_fc_data)
node_fc_df = pd.DataFrame(node_fc_data)
fcd_df = pd.DataFrame(fcd_data)
alpha_df = pd.DataFrame(alpha_data)
r_df = pd.DataFrame(r_data)


# Create combined performance metric
combined_data = []
for _, row in edge_fc_df.iterrows():
    hmap_label, id_val = row['hmap_label'], row['id']
    
    # Get corresponding values from other metrics
    edge_fc_val = row['value'] if "edge_fc_corr" in metrics else 0
    node_fc_val = (node_fc_df[(node_fc_df['hmap_label'] == hmap_label) & 
                               (node_fc_df['id'] == id_val)]['value'].iloc[0] 
                   if "node_fc_corr" in metrics else 0)
    fcd_val = (fcd_df[(fcd_df['hmap_label'] == hmap_label) & 
                      (fcd_df['id'] == id_val)]['value'].iloc[0] 
               if "fcd_ks" in metrics else 0)
    
    combined_val = edge_fc_val + node_fc_val + (1 - fcd_val if "fcd_ks" in metrics else 0)
    combined_data.append({'hmap_label': hmap_label, 'id': id_val, 'value': combined_val})

combined_df = pd.DataFrame(combined_data)

# %%
# Print summary statistics for each heterogeneity map
for hmap_label in hmap_labels.keys():
    print(f"Heterogeneity map: {hmap_labels[hmap_label]}")
    
    if "edge_fc_corr" in metrics:
        edge_vals = edge_fc_df[edge_fc_df['hmap_label'] == hmap_label]['value']
        print(f"  Edge-level FC: Mean = {edge_vals.mean():.3f}, Std = {edge_vals.std():.3f}")
    
    if "node_fc_corr" in metrics:
        node_vals = node_fc_df[node_fc_df['hmap_label'] == hmap_label]['value']
        print(f"  Node-level FC: Mean = {node_vals.mean():.3f}, Std = {node_vals.std():.3f}")
    
    if "fcd_ks" in metrics:
        fcd_vals = fcd_df[fcd_df['hmap_label'] == hmap_label]['value']
        print(f"  FCD KS: Mean = {fcd_vals.mean():.3f}, Std = {fcd_vals.std():.3f}")
    
    combined_vals = combined_df[combined_df['hmap_label'] == hmap_label]['value']
    print(f"  Combined metric: Mean = {combined_vals.mean():.3f}, Std = {combined_vals.std():.3f}")
    print()


# %%
# Plotting configuration
fs_ax = 15
fs_title = 20
plt.rcParams.update({
    'xtick.major.size': 5,
    'xtick.major.width': 1.5,
    'xtick.bottom': True,
    'ytick.left': True,
    'figure.dpi': 300
})

# Sort heterogeneity maps by combined performance
sorted_hmap_labels = (combined_df.groupby('hmap_label')['value']
                      .mean()
                      .sort_values()
                      .index
                      .tolist())
sorted_plotting_labels = [hmap_labels[label] for label in sorted_hmap_labels]

# Plot performance metrics
fig, axs = plt.subplots(1, len(metrics) + 1, figsize=(len(metrics) * 9, 6))
if len(metrics) == 1:
    axs = [axs]

plot_idx = 0

# Plot edge-level FC
if "edge_fc_corr" in metrics:
    sns.barplot(
        data=edge_fc_df, x='hmap_label', y='value', hue="id", 
        order=sorted_hmap_labels, ax=axs[plot_idx], errorbar="sd", palette="deep"
    )
    axs[plot_idx].set_xticks(range(len(sorted_hmap_labels)))
    axs[plot_idx].set_xticklabels(sorted_plotting_labels, ha="right", rotation=45)
    
    # Update legend labels
    handles, labels = axs[plot_idx].get_legend_handles_labels()
    axs[plot_idx].legend(handles, [ids[int(label)] for label in labels])
    
    axs[plot_idx].tick_params(axis='y', labelsize=fs_ax)
    axs[plot_idx].set_title("Edge-level FC fit", fontsize=fs_title)
    axs[plot_idx].set_xlabel("Heterogeneity map", fontsize=fs_ax)
    axs[plot_idx].set_ylabel("Pearson's r", fontsize=fs_ax)
    axs[plot_idx].spines[['top', 'right']].set_visible(False)
    axs[plot_idx].set_ylim(0, 1)
    
    plot_idx += 1

# Plot node-level FC
if "node_fc_corr" in metrics:
    sns.barplot(
        data=node_fc_df, x='hmap_label', y='value', hue="id", 
        order=sorted_hmap_labels, ax=axs[plot_idx], errorbar="sd", palette="deep"
    )
    axs[plot_idx].set_xticks(range(len(sorted_hmap_labels)))
    axs[plot_idx].set_xticklabels(sorted_plotting_labels, ha="right", rotation=45)
    
    handles, labels = axs[plot_idx].get_legend_handles_labels()
    axs[plot_idx].legend(handles, [ids[int(label)] for label in labels])
    
    axs[plot_idx].tick_params(axis='y', labelsize=fs_ax)
    axs[plot_idx].set_title("Node-level FC fit", fontsize=fs_title)
    axs[plot_idx].set_xlabel("Heterogeneity map", fontsize=fs_ax)
    axs[plot_idx].set_ylabel("Pearson's r", fontsize=fs_ax)
    axs[plot_idx].spines[['top', 'right']].set_visible(False)
    axs[plot_idx].set_ylim(0, 1)
    
    plot_idx += 1

# Plot FCD (inverted for visualization)
if "fcd_ks" in metrics:
    fcd_plot_df = fcd_df.copy()
    fcd_plot_df['value'] = 1 - fcd_plot_df['value']
    
    sns.barplot(
        data=fcd_plot_df, x='hmap_label', y='value', hue="id", 
        order=sorted_hmap_labels, ax=axs[plot_idx], errorbar="sd", palette="deep"
    )
    axs[plot_idx].set_xticks(range(len(sorted_hmap_labels)))
    axs[plot_idx].set_xticklabels(sorted_plotting_labels, ha="right", rotation=45)
    
    handles, labels = axs[plot_idx].get_legend_handles_labels()
    axs[plot_idx].legend(handles, [ids[int(label)] for label in labels])
    
    axs[plot_idx].tick_params(axis='y', labelsize=fs_ax)
    axs[plot_idx].set_title("FCD fit", fontsize=fs_title)
    axs[plot_idx].set_xlabel("Heterogeneity map", fontsize=fs_ax)
    axs[plot_idx].set_ylabel("1 - KS statistic", fontsize=fs_ax)
    axs[plot_idx].spines[['top', 'right']].set_visible(False)
    axs[plot_idx].set_ylim(0, 1)
    
    plot_idx += 1

# Plot combined metric
sns.barplot(
    data=combined_df, x='hmap_label', y='value', hue="id", 
    order=sorted_hmap_labels, ax=axs[plot_idx], errorbar="sd", palette="deep"
)
axs[plot_idx].set_xticks(range(len(sorted_hmap_labels)))
axs[plot_idx].set_xticklabels(sorted_plotting_labels, ha="right", rotation=45)

handles, labels = axs[plot_idx].get_legend_handles_labels()
axs[plot_idx].legend(handles, [ids[int(label)] for label in labels])

axs[plot_idx].tick_params(axis='y', labelsize=fs_ax)
axs[plot_idx].set_title("Overall fit", fontsize=fs_title)
axs[plot_idx].set_xlabel("Heterogeneity map", fontsize=fs_ax)
axs[plot_idx].set_ylabel("Combined metric", fontsize=fs_ax)
axs[plot_idx].spines[['top', 'right']].set_visible(False)
axs[plot_idx].set_ylim(0, 2.25)

plt.tight_layout()
# plt.show()


# %%
# Plot optimal parameters (alpha and r)
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot alpha
sns.barplot(
    data=alpha_df, x='hmap_label', y='value', hue="id", 
    order=sorted_hmap_labels, ax=axs[0], palette="deep"
)
axs[0].set_xticks(range(len(sorted_hmap_labels)))
axs[0].set_xticklabels(sorted_plotting_labels, ha="right", rotation=45)

handles, labels = axs[0].get_legend_handles_labels()
axs[0].legend(handles, [ids[int(label)] for label in labels])

axs[0].tick_params(axis='y', labelsize=fs_ax)
axs[0].set_title(r"$\alpha$", fontsize=fs_title)
axs[0].set_xlabel("Heterogeneity map", fontsize=fs_ax)
axs[0].set_ylabel("Optimal value", fontsize=fs_ax)
axs[0].spines[['top', 'right']].set_visible(False)

# Plot r
sns.barplot(
    data=r_df, x='hmap_label', y='value', hue="id", 
    order=sorted_hmap_labels, ax=axs[1], palette="deep"
)
axs[1].set_xticks(range(len(sorted_hmap_labels)))
axs[1].set_xticklabels(sorted_plotting_labels, ha="right", rotation=45)

handles, labels = axs[1].get_legend_handles_labels()
axs[1].legend(handles, [ids[int(label)] for label in labels])

axs[1].tick_params(axis='y', labelsize=fs_ax)
axs[1].set_title(r"$r_s$", fontsize=fs_title)
axs[1].set_xlabel("Heterogeneity map", fontsize=fs_ax)
axs[1].set_ylabel("mm", fontsize=fs_ax)
axs[1].spines[['top', 'right']].set_visible(False)

plt.tight_layout()
# plt.show()

