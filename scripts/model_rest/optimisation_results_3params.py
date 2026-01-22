"""
Visualize 2D parameter sweep results for neural field model optimization.

This script generates heatmaps showing model performance across two parameter
dimensions (e.g., alpha vs r), allowing visualization of the parameter landscape
and identification of optimal parameter regions.
"""

# %%
import os
import h5py
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from heteromodes.utils import get_project_root

sns.set_theme(style="white")

# Configuration
PROJ_DIR = get_project_root()
species = "human"
id_num = 102
hmap_label = "SAaxis"
evaluation = "fit"  # 'fit' or 'crossval'
plot_combined = False

config_file = f"{PROJ_DIR}/results/{species}/model_rest/group/id-{id_num}/config.json"
with open(config_file, 'r') as f:
    config = json.load(f)
metrics = config['metrics']
if plot_combined:
    metrics.append("combined")
print(metrics)

# Parameters to sweep
param1 = "beta"  # Note: alpha must be param1 if used
param2 = "r"
param3 = "gamma"

# Default parameter values (used for parameters not being swept)
default_params = {
    'alpha': 0.0,
    'beta': 5.0,
    'r': 30.0,
    'gamma': 0.1
}

results_dir = f"{PROJ_DIR}/results/{species}/model_rest/group/id-{id_num}/{evaluation}"


# %%
# Load parameter combinations and initialize result matrices
param_combs = pd.read_csv(f"{results_dir}/{hmap_label}/parameter_combinations.csv")
param1_vals = param_combs[param1].unique().round(2)
param2_vals = param_combs[param2].unique().round(2)
param3_vals = param_combs[param3].unique().round(2)

# Initialize performance matrices
edge_fc_land = np.full((len(param1_vals), len(param2_vals), len(param3_vals)), np.nan)
node_fc_land = np.full((len(param1_vals), len(param2_vals), len(param3_vals)), np.nan)
fcd_land = np.full((len(param1_vals), len(param2_vals), len(param3_vals)), np.nan)
cpc1_land = np.full((len(param1_vals), len(param2_vals), len(param3_vals)), np.nan)
combined_land = np.full((len(param1_vals), len(param2_vals), len(param3_vals)), 0.0)

# Load results for each parameter combination
for i, p1 in enumerate(param1_vals):
    for j, p2 in enumerate(param2_vals):
        for k, p3 in enumerate(param3_vals):
            # Create parameter dict with current sweep values
            params = default_params.copy()
            params[param1] = p1
            params[param2] = p2
            params[param3] = p3
            
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
                    if 'edge_fc_corr' in metrics:
                        edge_fc_land[i, j, k] = np.mean(f['results']['edge_fc_corr'][:])
                    if 'node_fc_corr' in metrics:
                        node_fc_land[i, j, k] = np.mean(f['results']['node_fc_corr'][:])
                    if 'fcd_ks' in metrics:
                        fcd_land[i, j, k] = np.mean(f['results']['fcd_ks'][:])
                    if 'cpc1_corr' in metrics:
                        cpc1_land[i, j, k] = np.mean(f['results']['cpc1_corr'][:])
                else:
                    if 'edge_fc_corr' in metrics:
                        edge_fc_land[i, j, k] = f['results']['edge_fc_corr'][()]
                    if 'node_fc_corr' in metrics:
                        node_fc_land[i, j, k] = f['results']['node_fc_corr'][()]
                    if 'fcd_ks' in metrics:
                        fcd_land[i, j, k] = f['results']['fcd_ks'][()]
                    if 'cpc1_corr' in metrics:
                        cpc1_land[i, j, k] = f['results']['cpc1_corr'][()]
            
            if 'combined' in metrics:
                # Compute combined metric
                if 'edge_fc_corr' in metrics:
                    combined_land[i, j, k] += edge_fc_land[i, j, k]
                if 'node_fc_corr' in metrics:
                    combined_land[i, j, k] += node_fc_land[i, j, k]
                if 'fcd_ks' in metrics:
                    combined_land[i, j, k] += (1 - fcd_land[i, j, k])
                if 'cpc1_corr' in metrics:
                    combined_land[i, j, k] += cpc1_land[i, j, k]

# Insert homogeneous model results at alpha=0 if alpha is being swept
# if param1 == 'alpha':
#     # Initialize homogeneous result arrays
#     edge_fc_hom = np.full(len(param2_vals), np.nan)
#     node_fc_hom = np.full(len(param2_vals), np.nan)
#     fcd_hom = np.full(len(param2_vals), np.nan)
#     combined_hom = np.full(len(param2_vals), 0)
    
#     # Load homogeneous model results for each param2 value
#     for j, p2 in enumerate(param2_vals):
#         params = default_params.copy()
#         params[param2] = p2
        
#         file_path = (
#             f"{results_dir}/None/model_alpha-{params['alpha']}_"
#             f"r-{params['r']}_gamma-{params['gamma']:.3f}_beta-{params['beta']}.h5"
#         )
        
#         with h5py.File(file_path, 'r') as f:
#             if evaluation == "crossval":
#                 edge_fc_hom[j] = np.mean(f['results']['edge_fc_corr'][:])
#                 node_fc_hom[j] = np.mean(f['results']['node_fc_corr'][:])
#                 fcd_hom[j] = np.mean(f['results']['fcd_ks'][:])
#             else:
#                 edge_fc_hom[j] = f['results']['edge_fc_corr'][()]
#                 node_fc_hom[j] = f['results']['node_fc_corr'][()]
#                 fcd_hom[j] = f['results']['fcd_ks'][()]
            
#             combined_hom[j] = edge_fc_hom[j] + node_fc_hom[j] + (1 - fcd_hom[j])

#     # Insert homogeneous results at midpoint of alpha axis
#     hom_ind = len(param1_vals) // 2
#     edge_fc_land = np.insert(edge_fc_land, hom_ind, edge_fc_hom, axis=0)
#     node_fc_land = np.insert(node_fc_land, hom_ind, node_fc_hom, axis=0)
#     fcd_land = np.insert(fcd_land, hom_ind, fcd_hom, axis=0)
#     combined_land = np.insert(combined_land, hom_ind, combined_hom, axis=0)
#     param1_vals = np.insert(param1_vals, hom_ind, 0.0)

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print(metrics)
fig = make_subplots(
    rows=1, cols=len(metrics), 
    subplot_titles=metrics,
    specs=[[{'type': 'scene'} for _ in range(len(metrics))]],
)

# Prepare coordinate arrays
x_coords = np.repeat(param1_vals, len(param2_vals)*len(param3_vals))
y_coords = np.tile(np.repeat(param2_vals, len(param3_vals)), len(param1_vals))
z_coords = np.tile(param3_vals, len(param1_vals)*len(param2_vals))

# Edge-level FC surface
col = 1
if 'edge_fc_corr' in metrics:
    fig.add_trace(
        go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=6,
                color=edge_fc_land.flatten(),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Pearson's r", x=0.28, len=0.5, thickness=10)
            )
        ),
        row=1, col=col
    )
    col += 1

if 'node_fc_corr' in metrics:
    fig.add_trace(
        go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=6,
                color=node_fc_land.flatten(),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Pearson's r", x=0.63, len=0.5, thickness=10)
            )
        ),
        row=1, col=col
    )
    col += 1

if 'fcd_ks' in metrics:
    fig.add_trace(
        go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=6,
                color=fcd_land.flatten(),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="1 - KS", x=0.89, len=0.5, thickness=10)
            )
        ),
        row=1, col=col
    )
    col += 1

if 'cpc1_corr' in metrics:
    fig.add_trace(
        go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=6,
                color=cpc1_land.flatten(),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Pearson's r", x=1.14, len=0.5, thickness=10)
            )
        ),
        row=1, col=col
    )
    col += 1

if 'combined' in metrics:
    fig.add_trace(
        go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=6,
                color=combined_land.flatten(),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="Combined", 
                    x=0.97 if 'fcd_ks' not in metrics else 1.22, 
                    len=0.5,
                    thickness=10
                )
            )
        ),
        row=1, col=col
    )

height = 400
width = (len(metrics)+1)*300
fig.update_layout(
    height=height, width=width,
    margin=dict(l=0, r=0, t=30, b=0),
    showlegend=False
)

# Update scene axes labels for each subplot
for i in range(1, len(metrics)+2):
    fig.update_scenes(
        xaxis_title=param1,
        yaxis_title=param2,
        zaxis_title=param3,
        camera=dict(
            eye=dict(x=2.5, y=2.5, z=2.5)
        ),
        row=1, col=i
    )
fig.show()

# export to html
html_file = f"{results_dir}/{hmap_label}/parameter_landscape.html"
fig.write_html(html_file)

# print html embed code (using repr to prevent link rendering)
embed_code = f'<iframe src="file://{html_file}" width="{width}" height="{height}"></iframe>'
print(embed_code)
