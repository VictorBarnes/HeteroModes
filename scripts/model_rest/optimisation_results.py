# %%
import os
import json
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from neuromaps.datasets import fetch_atlas
from nsbtools.utils import unmask
from nsbtools.plotting import plot_surf, plot_heatmap
from heteromodes.utils import get_project_root

sns.set_theme(style="white")
PROJ_DIR = get_project_root()  # Project root directory

# %%
species = "macaque"
ids = {3: '3'}
evaluation = "fit"

with open(f"{PROJ_DIR}/data/heteromaps/{species}/heteromap_labels.json", "r") as f:
    hmap_labels = json.load(f)["heteromap_labels"]
hmap_labels["None"] = "Homogeneous"

# %%
# Initialize lists to store data for DataFrames
edge_fc_data, node_fc_data, fcd_data, alpha_data, r_data = [], [], [], [], []

for id in ids.keys():
    results_dir = f"{PROJ_DIR}/results/{species}/model_rest/group/id-{id}/{evaluation}"
    
    # Load metrics from config
    with open(f"{PROJ_DIR}/results/{species}/model_rest/group/id-{id}/config.json", "r") as f:
        metrics = json.load(f)["metrics"].split(" ")
    
    for hmap_label in hmap_labels.keys():
        file = f"{results_dir}/{hmap_label}/best_model.h5"

        if not os.path.exists(file):
            print(f"File {file} does not exist, skipping...")
            continue

        with h5py.File(file, 'r+') as f:
            # Extract metric values (flatten to handle cross-validation splits)
            edge_fc_values = np.array(f['results']['edge_fc_corr']).flatten()
            node_fc_values = np.array(f['results']['node_fc_corr']).flatten()
            fcd_values = np.array(f['results']['fcd_ks']).flatten()
            
            # Extract parameter values
            alpha_values = np.array(f['alpha']).flatten()
            r_values = np.array(f['r']).flatten()
            
            # Add data to lists (one row per cross-validation split)
            for i in range(len(edge_fc_values)):
                edge_fc_data.append({'hmap_label': hmap_label, 'id': id, 'value': edge_fc_values[i]})
                node_fc_data.append({'hmap_label': hmap_label, 'id': id, 'value': node_fc_values[i]})
                fcd_data.append({'hmap_label': hmap_label, 'id': id, 'value': fcd_values[i]})
                alpha_data.append({'hmap_label': hmap_label, 'id': id, 'value': alpha_values[i]})
                r_data.append({'hmap_label': hmap_label, 'id': id, 'value': r_values[i]})

# Create DataFrames
edge_fc_df = pd.DataFrame(edge_fc_data)
node_fc_df = pd.DataFrame(node_fc_data)
fcd_df = pd.DataFrame(fcd_data)
alpha_df = pd.DataFrame(alpha_data)
r_df = pd.DataFrame(r_data)

# Create combined DataFrame
combined_data = []
for _, row in edge_fc_df.iterrows():
    hmap_label, id_val = row['hmap_label'], row['id']
    
    # Get corresponding values from other metrics
    edge_fc_val = row['value'] if "edge_fc_corr" in metrics else 0
    node_fc_val = node_fc_df[(node_fc_df['hmap_label'] == hmap_label) & (node_fc_df['id'] == id_val)]['value'].iloc[0] if "node_fc_corr" in metrics else 0
    fcd_val = fcd_df[(fcd_df['hmap_label'] == hmap_label) & (fcd_df['id'] == id_val)]['value'].iloc[0] if "fcd_ks" in metrics else 0
    
    combined_val = edge_fc_val + node_fc_val + (1 - fcd_val if "fcd_ks" in metrics else 0)
    combined_data.append({'hmap_label': hmap_label, 'id': id_val, 'value': combined_val})

combined_df = pd.DataFrame(combined_data)

#%%
# Print mean and std for each metric and heterogeneity map
for hmap_label in hmap_labels.keys():
    print(f"Heterogeneity map: {hmap_labels[hmap_label]}")
    if "edge_fc_corr" in metrics:
        mean_edge_fc = edge_fc_df[edge_fc_df['hmap_label'] == hmap_label]['value'].mean()
        std_edge_fc = edge_fc_df[edge_fc_df['hmap_label'] == hmap_label]['value'].std()
        print(f"  Edge-level FC: Mean = {mean_edge_fc:.3f}, Std = {std_edge_fc:.3f}")
    if "node_fc_corr" in metrics:
        mean_node_fc = node_fc_df[node_fc_df['hmap_label'] == hmap_label]['value'].mean()
        std_node_fc = node_fc_df[node_fc_df['hmap_label'] == hmap_label]['value'].std()
        print(f"  Node-level FC: Mean = {mean_node_fc:.3f}, Std = {std_node_fc:.3f}")
    if "fcd_ks" in metrics:
        mean_fcd = fcd_df[fcd_df['hmap_label'] == hmap_label]['value'].mean()
        std_fcd = fcd_df[fcd_df['hmap_label'] == hmap_label]['value'].std()
        print(f"  FCD KS: Mean = {mean_fcd:.3f}, Std = {std_fcd:.3f}")
    mean_combined = combined_df[combined_df['hmap_label'] == hmap_label]['value'].mean()
    std_combined = combined_df[combined_df['hmap_label'] == hmap_label]['value'].std()
    print(f"  Combined metric: Mean = {mean_combined:.3f}, Std = {std_combined:.3f}")
    print()


# %%
# Set plotting defaults
fs_ax = 15
fs_title = 20
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['figure.dpi'] = 300

fig, axs = plt.subplots(1, len(metrics)+1, figsize=(len(metrics)*9, 6))
if len(metrics) == 1:
    axs = [axs]

# Sort the hmaps by mean combined score
combined_means = combined_df.groupby('hmap_label')['value'].mean()
sorted_hmap_labels = combined_means.sort_values().index.tolist()
sorteed_plotting_labels = [hmap_labels[label] for label in sorted_hmap_labels]

# Plot edge-level fc
i = 0
if "edge_fc_corr" in metrics:
    sns.barplot(data=edge_fc_df, x='hmap_label', y='value', hue="id", order=sorted_hmap_labels, ax=axs[i], errorbar="sd", palette="deep")
    axs[i].set_xticks(ticks=range(len(list(hmap_labels))), labels=[hmap_labels[label] for label in sorted_hmap_labels], ha="right")
    # Update legend labels
    handles, labels = axs[i].get_legend_handles_labels()
    axs[i].legend(handles, [ids[int(label)] for label in labels])
    axs[i].tick_params(axis='x', labelrotation=45)
    axs[i].tick_params(axis='y', labelsize=fs_ax)
    axs[i].set_title("Edge-level FC fit", fontsize=fs_title)
    axs[i].set_xlabel("Heterogeneity map", fontsize=fs_ax)
    axs[i].set_ylabel("Pearson's r", fontsize=fs_ax)
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].set_ylim(0, 1)

    i += 1

# Plot node-level fc
if "node_fc_corr" in metrics:
    sns.barplot(data=node_fc_df, x='hmap_label', y='value', hue="id", order=sorted_hmap_labels, ax=axs[i], errorbar="sd", palette="deep")
    axs[i].set_xticks(ticks=range(len(list(hmap_labels))), labels=[hmap_labels[label] for label in sorted_hmap_labels], ha="right")
    # Update legend labels
    handles, labels = axs[i].get_legend_handles_labels()
    axs[i].legend(handles, [ids[int(label)] for label in labels])
    axs[i].tick_params(axis='x', labelrotation=45)
    axs[i].tick_params(axis='y', labelsize=fs_ax)
    axs[i].set_title("Node-level FC fit", fontsize=fs_title)
    axs[i].set_xlabel("Heterogeneity map", fontsize=fs_ax)
    axs[i].set_ylabel("Pearson's r", fontsize=fs_ax)
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].set_ylim(0, 1)

    i += 1

# Plot FCD
if "fcd_ks" in metrics:
    # Create a copy with inverted FCD values for plotting
    fcd_plot_df = fcd_df.copy()
    fcd_plot_df['value'] = 1 - fcd_plot_df['value']
    
    sns.barplot(data=fcd_plot_df, x='hmap_label', y='value', hue="id", order=sorted_hmap_labels, ax=axs[i], errorbar="sd", palette="deep")
    axs[i].set_xticks(ticks=range(len(list(hmap_labels))), labels=[hmap_labels[label] for label in sorted_hmap_labels], ha="right")
    # Update legend labels
    handles, labels = axs[i].get_legend_handles_labels()
    axs[i].legend(handles, [ids[int(label)] for label in labels])
    axs[i].tick_params(axis='x', labelrotation=45)
    axs[i].tick_params(axis='y', labelsize=fs_ax)
    axs[i].set_title("FCD fit", fontsize=fs_title)
    axs[i].set_xlabel("Heterogeneity map", fontsize=fs_ax)
    axs[i].set_ylabel("1 - KS statistic", fontsize=fs_ax)
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].set_ylim(0, 1)

    i += 1

# Plot combined metric
sns.barplot(data=combined_df, x='hmap_label', y='value', hue="id", order=sorted_hmap_labels, ax=axs[i], errorbar="sd", palette="deep")
axs[i].set_xticks(ticks=range(len(list(hmap_labels))), labels=[hmap_labels[label] for label in sorted_hmap_labels], ha="right")
# Update legend labels
handles, labels = axs[i].get_legend_handles_labels()
axs[i].legend(handles, [ids[int(label)] for label in labels])
axs[i].tick_params(axis='x', labelrotation=45)
axs[i].tick_params(axis='y', labelsize=fs_ax)
axs[i].set_title("Overall fit", fontsize=fs_title)
axs[i].set_xlabel("Heterogeneity map", fontsize=fs_ax)
axs[i].set_ylabel("Combined metric", fontsize=fs_ax)
axs[i].spines['top'].set_visible(False)
axs[i].spines['right'].set_visible(False)
axs[i].set_ylim(0, 2.25)

plt.tight_layout()
# plt.show()

#%%
# Plot alpha and r parameters
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

sns.barplot(data=alpha_df, x='hmap_label', y='value', hue="id", order=sorted_hmap_labels, ax=axs[0], palette="deep")
axs[0].set_xticks(ticks=range(len(list(hmap_labels))), labels=[hmap_labels[label] for label in sorted_hmap_labels], ha="right")
# Update legend labels
handles, labels = axs[0].get_legend_handles_labels()
axs[0].legend(handles, [ids[int(label)] for label in labels])
axs[0].tick_params(axis='x', labelrotation=45)
axs[0].tick_params(axis='y', labelsize=fs_ax)
axs[0].set_title(r"$\alpha$", fontsize=fs_title)
axs[0].set_xlabel("Heterogeneity map", fontsize=fs_ax)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)

sns.barplot(data=r_df, x='hmap_label', y='value', hue="id", order=sorted_hmap_labels, ax=axs[1], palette="deep")
axs[1].set_xticks(ticks=range(len(list(hmap_labels))), labels=[hmap_labels[label] for label in sorted_hmap_labels], ha="right")
# Update legend labels
handles, labels = axs[1].get_legend_handles_labels()
axs[1].legend(handles, [ids[int(label)] for label in labels])
axs[1].tick_params(axis='x', labelrotation=45)
axs[1].tick_params(axis='y', labelsize=fs_ax)
axs[1].set_title(r"$r_s$", fontsize=fs_title)
axs[1].set_xlabel("Heterogeneity map", fontsize=fs_ax)
axs[1].set_ylabel("mm", fontsize=fs_ax)
# axs[1].set_ylim([15, 20])
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)

plt.tight_layout()
# plt.show()
