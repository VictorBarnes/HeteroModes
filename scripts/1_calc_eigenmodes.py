#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate heterogeneous eigenmodes of a surface by solving the heterogeneous Helmholtz equation.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
from lapy import Solver, TriaMesh
from brainspace.mesh import mesh_io, mesh_operations


# Global variables
DENSITIES = {"32k": 32492}
CMEAN = 3352.4  # 28.9
alpha_vals = np.arange(0.2, 2.2, 0.2)
beta_vals = np.sort(np.append(np.arange(-10.0, 11.0, 1.0), [-0.5, 0.5]))

# Load config file
config_file = "scripts/config.json"
with open(config_file, encoding="UTF-8") as f:
    config = json.load(f)

# Load map chosen to paramaterize heterogeneity
if config["hetero_label"] is None:
    # No heterogeneity is encoded by an array of ones
    hetero_map = np.ones(DENSITIES["32k"]).reshape(-1, 1)
    # Setting alpha and beta to 0 will mean that cs is CMEAN at every vertex
    alpha_vals = [0.0]
    beta_vals = [0.0]
else:
    # Load heterogeneity map
    hetero_file = config["hetero_maps"][config["hetero_label"]]
    hetero_map = nib.load(hetero_file).agg_data().reshape(-1, 1)

# Load surface template and medial wall mask
surf = mesh_io.read_surface(f"{config['surface_dir']}/atlas-{config['atlas']}_"
                            f"space-{config['space']}_den-{config['den']}_surf-{config['surf']}_"
                            f"hemi-{config['hemi']}_surface.vtk")
medial_mask = np.loadtxt(f"{config['surface_dir']}/atlas-{config['atlas']}_space-{config['space']}_"
                         f"den-{config['den']}_hemi-{config['hemi']}_medialMask.txt").astype(bool)
cortex_inds = np.array([i for i, med in enumerate(medial_mask) if med])

# Mask surface template and heterogeneous map
if config['mask_medial']:
    # Mask surface template
    surf_masked = mesh_operations.mask_points(surf, medial_mask)
    v = surf_masked.Points
    t = np.reshape(surf_masked.Polygons, [surf_masked.n_cells, 4])[:,1:4]
    mesh = TriaMesh(v, t)
    # Mask heterogeneous map
    hetero_map = hetero_map[medial_mask]
else:
    # Initialise mesh without masking medial wall
    mesh = TriaMesh.read_vtk(getattr(surf, config['hemi']))

alpha_beta_combs = []
# Loop through alpha and beta, calculate cs, and solve eigenmodes and eigenvalues
for i, alpha in enumerate(alpha_vals):
    for j, beta in enumerate(beta_vals):
        print(f"Space: {config['space']} | Density: {config['den']} | Surface: {config['surf']} | "
              f"Hetero: {config['hetero_label']} | alpha: {alpha} | beta: {beta} | "
              f"cmean: {CMEAN} | nmodes: {config['n_modes']}")
        # Scale propagation speed
        scaler = MinMaxScaler(feature_range=(0, 1))
        rho = scaler.fit_transform(hetero_map).flatten()
        cs = CMEAN * (1 + alpha*(rho - np.mean(rho)))**beta

        # Ensure C doesn't have negative values
        try:
            # If the assert passes, store the alpha and beta values
            assert np.min(cs) >= 1.0, f"Values of cs less than 1.0 can lead to infinity problems."
            alpha_beta_combs.append((alpha, beta))
        except AssertionError as e:
            # If the assert fails, skip to the next iteration
            print(e)
            continue

        # Each term in C needs to be squared (according to the NFT equation)
        cs **= 2

        # Calculate cs for each triangle by taking the average of the values at its vertices
        cs_tri = mesh.map_vfunc_to_tfunc(cs)
        # Initialise FEM solver and solve for eigenvalues and eigenmodes
        fem = Solver(mesh, aniso=(0, 0), hetero=cs_tri)
        evals, emodes = fem.eigs(k=config['n_modes'])

        if config['mask_medial']:
            # Reshape emodes to match vertices of original surface
            emodes_reshaped = np.zeros([surf.n_points, config['n_modes']])
            for mode in range(config['n_modes']):
                emodes_reshaped[cortex_inds, mode] = emodes[:, mode]
            emodes = emodes_reshaped

            # Reshape propagation speed map to match vertices of original surface
            cs_reshaped = np.zeros(surf.n_points)
            cs_reshaped[cortex_inds] = cs
            cs = cs_reshaped            

        if config['save_results']:
            print("Saving eigenmodes and eigenvalues...")
            # Set output file names and save
            desc = (f"hetero-{config['hetero_label']}_atlas-{config['atlas']}_"
                    f"space-{config['space']}_den-{config['den']}_surf-{config['surf']}_"
                    f"hemi-{config['hemi']}_n-{config['n_modes']}_alpha-{alpha}_beta-{beta}_"
                    f"maskMed-{config['mask_medial']}")

            evals_savefile = Path(config['emode_dir'], f"{desc}_evals.txt")
            emodes_savefile = Path(config['emode_dir'], f"{desc}_emodes.txt")
            cmap_savefile = Path(config['emode_dir'], "cmaps", f"{desc}_cmap.txt")
            np.savetxt(evals_savefile, evals)
            np.savetxt(emodes_savefile, emodes)
            np.savetxt(cmap_savefile, cs)

# Save valid alpha and beta combinations
if config['save_results']:
    alpha_beta_combs_df = pd.DataFrame(alpha_beta_combs, columns=['alpha', 'beta'])
    alpha_beta_combs_df.to_csv(Path(config['project_dir'], "data", "alpha_beta_combs.csv"), index=False)
