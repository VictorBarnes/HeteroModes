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
from nsbtools.plotting import plot_brain, plot_heatmap

sns.set_theme(style="white")
PROJ_DIR = "/fs04/kg98/vbarnes/HeteroModes"

# %%
species = "macaque"
id = 3
evaluation = "fit"

if species == "human":
    hmap_labels = {
        "None": "Homogeneous",
        "myelinmap": "T1w/T2w",
        "thickness": "Cortical thickness",
        "synapticden": "Synaptic density",
        "odi": "ODI",
        "ndi": "NDI",
        "genel4PC1": "Layer IV",
        "eiratio1.2": "E:I ratio",
        "megtimescale": "MEG timescale"
    }
elif species == "macaque":
    hmap_labels = {
        "None": "Homogeneous",
        "myelinmap": "T1w/T2w",
        "thickness": "Cortical thickness",
        "ampa": "AMPA",
        "cgp5": "GABA-B",
        "damp": "M3",
        "dpat": "5HT1A",
        "dpmg": "Adenosine 1",
        "exh": "average excitatory",
        "flum": "GABA-A/BZ",
        "inh": "average inhibitory",
        "kain": "kainate",
        "keta": "5HT2",
        "mk80": "NMDA",
        "mod": "average modulatory",
        "musc": "GABA-A",
        "oxot": "M2",
        "pire": "M1",
        "praz": "alpha1",
        "uk14": "alpha2"
    }
elif species == "marmoset":
    hmap_labels = {
        "None": "Homogeneous",
        "myelinmap": "T1w/T2w",
        "thickness": "Cortical thickness",
        "nissl": "Nissl"
    }

results_dir = f"{PROJ_DIR}/results/{species}/model_rest/group/id-{id}/{evaluation}"

# %%
edge_fc_xval, node_fc_xval, fcd_ks_xval, fc_matrices_xval, fcd_xval = [], [], [], [], []
alpha_best_xval, beta_best_xval, r_best_xval, gamma_best_xval = {}, {}, {}, {}
for hmap_label in hmap_labels.keys():
    file = f"{results_dir}/{hmap_label}/best_model.h5"
    if not os.path.exists(file):
        print(f"File {file} does not exist, skipping...")
        continue

    with h5py.File(file, 'r+') as f:
        edge_fc_xval.append(np.array(f['results']['edge_fc_corr']).flatten())
        node_fc_xval.append(np.array(f['results']['node_fc_corr']).flatten())
        fcd_ks_xval.append(np.array(f['results']['fcd_ks']).flatten())

        alpha_best_xval[hmap_label] = np.array(f['alpha'])
        beta_best_xval[hmap_label] = np.array(f['beta'])
        r_best_xval[hmap_label] = np.array(f['r'])
        gamma_best_xval[hmap_label] = np.array(f['gamma'])

        fc_matrices_xval.append(f['fc'][:])
        fcd_xval.append(f['fcd'][:])

with open(f"{PROJ_DIR}/results/{species}/model_rest/group/id-{id}/config.json", "r") as f:
    metrics = json.load(f)["metrics"].split(" ")
    # metrics = ["edge_fc_corr", "node_fc_corr"]

combined = np.zeros_like(edge_fc_xval)
if "edge_fc_corr" in metrics:
    combined += np.array(edge_fc_xval)
if "node_fc_corr" in metrics:
    combined += np.array(node_fc_xval)
if "fcd_ks" in metrics:
    combined += 1 - np.array(fcd_ks_xval)

print(f"alpha_best: {alpha_best_xval}")
print(f"r best: {r_best_xval}")
print(f"beta best: {beta_best_xval}")
print(f"gamma best: {gamma_best_xval}")
print(np.shape(fc_matrices_xval), np.shape(fcd_xval))

# %%
# Set plotting defaults
fs_ax = 15
fs_title = 20
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['figure.dpi'] = 300

fig, axs = plt.subplots(1, len(metrics)+1, figsize=(len(metrics)*9, 7))
if len(metrics) == 1:
    axs = [axs]

pnts = np.linspace(0, np.pi * 2, 24)
circ = np.c_[np.sin(pnts) / 2, -np.cos(pnts) / 2]
vert = np.r_[circ, circ[::-1] * .7]
open_circle = mpl.path.Path(vert)

# Sort the hmaps by lowest combined score
sorted_inds = np.argsort(combined.mean(axis=1))

# Plot edge-level fc
i = 0
if "edge_fc_corr" in metrics:
    sns.barplot(data=np.array(edge_fc_xval)[sorted_inds, :].T, ax=axs[i], errorbar="sd")
    # sns.stripplot(data=edge_fc_xval, ax=axs[i], marker=open_circle, size=4, alpha=0.5, zorder=1, linewidth=1)
    # sns.violinplot(data=np.array(edge_fc_xval)[sorted_inds, :].T, ax=axs[i], density_norm="count", fill=False, linewidth=3, inner="box", inner_kws={"box_width": 5, "whis_width": 1, "color": "black"})
    # sns.violinplot(data=np.array(edge_fc_xval)[sorted_inds, :].T, ax=axs[i], density_norm="count", inner="box")
    axs[i].set_xticks(ticks=range(len(list(hmap_labels.values()))))
    axs[i].set_xticklabels(labels=[list(hmap_labels.values())[i] for i in sorted_inds], ha='right', fontsize=15)
    axs[i].tick_params(axis='x', labelrotation=45)
    axs[i].tick_params(axis='y', labelsize=fs_ax)
    axs[i].set_title("Edge-level FC fit", fontsize=fs_title)
    axs[i].set_xlabel("Heterogeneity map", fontsize=fs_ax)
    axs[i].set_ylabel("Pearson's r", fontsize=fs_ax)
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].set_ylim(0, 1)
    # for i, violin in enumerate(axs[0].collections[:len(hmap_labels)]):  # ::2 to skip the body parts, focusing on the borders
    #     violin.set_edgecolor(sns.color_palette()[i])
    #     violin.set_linewidth(2)

    i += 1

# Plot node-level fc
if "node_fc_corr" in metrics:
    sns.barplot(data=np.array(node_fc_xval)[sorted_inds, :].T, ax=axs[i], errorbar="sd")
    # sns.stripplot(data=node_fc_xval, ax=axs[i], marker=open_circle, size=4, alpha=0.5, zorder=1, linewidth=1)
    # sns.violinplot(data=np.array(node_fc_xval)[sorted_inds, :].T, ax=axs[i], density_norm="count", fill=False, linewidth=3, inner="box", inner_kws={"box_width": 5, "whis_width": 1, "color": "black"})
    # sns.violinplot(data=np.array(node_fc_xval)[sorted_inds, :].T, ax=axs[i], density_norm="count", inner="box")
    axs[i].set_xticks(ticks=range(len(list(hmap_labels.values()))), labels=[list(hmap_labels.values())[i] for i in sorted_inds], ha='right', fontsize=15)
    axs[i].tick_params(axis='x', labelrotation=45)
    axs[i].tick_params(axis='y', labelsize=fs_ax)
    axs[i].set_title("Node-level FC fit", fontsize=fs_title)
    axs[i].set_xlabel("Heterogeneity map", fontsize=fs_ax)
    axs[i].set_ylabel("Pearson's r", fontsize=fs_ax)
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].set_ylim(0, 1)

    i += 1

# Plot phase
if "fcd_ks" in metrics:
    sns.barplot(data=1-np.array(fcd_ks_xval)[sorted_inds, :].T, ax=axs[i], errorbar="sd")
    # sns.stripplot(data=fcd_ks_xval, ax=axs[i], marker=open_circle, size=4, alpha=0.5, zorder=1, linewidth=1)
    # sns.violinplot(data=np.array(fcd_ks_xval)[sorted_inds, :].T, ax=axs[i], density_norm="count", fill=False, linewidth=3, inner="box", inner_kws={"box_width": 5, "whis_width": 1, "color": "black"})
    # sns.violinplot(data=np.array(fcd_ks_xval)[sorted_inds, :].T, ax=axs[i], density_norm="count", inner="box")
    axs[i].set_xticks(ticks=range(len(list(hmap_labels.values()))), labels=[list(hmap_labels.values())[i] for i in sorted_inds], ha='right', fontsize=15)
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
# combined = np.array(edge_fc_xval) + np.array(node_fc_xval) + np.array(phase_xval)
sns.barplot(data=combined[sorted_inds, :].T, ax=axs[i], errorbar="sd")
# sns.stripplot(data=combined.T, ax=axs[i], marker=open_circle, size=4, alpha=0.5, zorder=1, linewidth=1)
# sns.violinplot(data=combined[sorted_inds, :].T, ax=axs[i], density_norm="count", fill=False, linewidth=3, inner="box", inner_kws={"box_width": 5, "whis_width": 1, "color": "black"})
# sns.violinplot(data=combined[sorted_inds, :].T, ax=axs[i], density_norm="count", inner="box")
axs[i].set_xticks(ticks=range(len(list(hmap_labels.values()))), labels=[list(hmap_labels.values())[i] for i in sorted_inds], ha='right', fontsize=15)
axs[i].tick_params(axis='x', labelrotation=45)
axs[i].tick_params(axis='y', labelsize=fs_ax)
axs[i].set_title("Overall fit", fontsize=fs_title)
axs[i].set_xlabel("Heterogeneity map", fontsize=fs_ax)
axs[i].set_ylabel("Combined metric", fontsize=fs_ax)
axs[i].spines['top'].set_visible(False)
axs[i].spines['right'].set_visible(False)
axs[i].set_ylim(0, 2.25)

plt.suptitle(f"Model fit results for {species} (id={id})", fontsize=fs_title+5)
plt.tight_layout()
plt.show()