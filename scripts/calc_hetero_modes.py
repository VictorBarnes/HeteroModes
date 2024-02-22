#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate heterogeneous eigenmodes of a surface by solving the heterogeneous Helmholtz equation.
"""

import os
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
CMEAN = 3352.4  # mm/s

# TODO: consider changing hetero_label to the path to the hetero map
# TODO: change the name of this function and the filename to be more intuitive
def calc_modes(config_file, hetero_label, alpha=0.0, beta=0.0):
    with open(config_file, encoding="UTF-8") as f:
        config = json.load(f)

    # Load map chosen to paramaterize heterogeneity
    if hetero_label is None:
        # No heterogeneity is encoded by an array of ones
        hetero_map = np.ones(DENSITIES["32k"]).reshape(-1, 1)
        # Alpha and beta must be equal to 0 in the homogeneous case
        assert alpha == 0.0 and beta == 0.0
    else:
        # Load heterogeneity map
        hetero_file = config["hetero_maps"][hetero_label]
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

    print(f"Space: {config['space']} | Density: {config['den']} | Surface: {config['surf']} | "
        f"Hetero: {hetero_label} | alpha: {alpha} | beta: {beta} | "
        f"cmean: {CMEAN} | nmodes: {config['n_modes']}")
    # Scale propagation speed
    scaler = MinMaxScaler(feature_range=(0, 1))
    rho = scaler.fit_transform(hetero_map).flatten()
    cs = CMEAN * (1 + alpha*(rho - np.mean(rho)))**beta
    # Ensure cs is between 0.1 and 150 m/s (inclusive)
    if np.min(cs)/1e3 < 0.1 or np.max(cs)/1e3 > 150:
        print("cs values are not within a physiological range of 0.1 to 150 m/s")
        return
    print(f"cs range: {np.min(cs)/1e3:.1f} m/s to {np.max(cs)/1e3:.1f} m/s")
    # Each term in cs needs to be squared (according to the NFT equation)
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
        desc = (f"hetero-{hetero_label}_atlas-{config['atlas']}_"
                f"space-{config['space']}_den-{config['den']}_surf-{config['surf']}_"
                f"hemi-{config['hemi']}_n-{config['n_modes']}_alpha-{alpha}_beta-{beta}_"
                f"maskMed-{config['mask_medial']}")

        evals_savefile = Path(config['emode_dir'], f"{desc}_evals.txt")
        emodes_savefile = Path(config['emode_dir'], f"{desc}_emodes.txt")
        # TODO: remove 'n_modes' from cmap filename
        cmap_savefile = Path(config['emode_dir'], "cmaps", f"{desc}_cmap.txt")
        np.savetxt(evals_savefile, evals)
        np.savetxt(emodes_savefile, emodes)
        np.savetxt(cmap_savefile, cs)


def save_valid_combs(config_file, hetero_label):
    with open(config_file, encoding="UTF-8") as f:
        config = json.load(f)
    emode_dir = config["emode_dir"]
    project_dir = config["project_dir"]

    combs_valid = []
    combs_all = pd.read_csv(Path(project_dir, "data", "csParamCombs_all.csv"))
    for i in range(combs_all.shape[0]):
        alpha = combs_all.loc[i, "alpha"]
        beta = combs_all.loc[i, "beta"]
        filename = (f"hetero-{hetero_label}_atlas-{config['atlas']}_"
                    f"space-{config['space']}_den-{config['den']}_surf-{config['surf']}_"
                    f"hemi-{config['hemi']}_n-{config['n_modes']}_alpha-{alpha}_beta-{beta}_"
                    f"maskMed-{config['mask_medial']}_emodes.txt")
        filepath = os.path.join(emode_dir, filename)
        if os.path.isfile(filepath):
                combs_valid.append((alpha, beta))
    print(f"Number of valid combinations for {hetero_label} modes: {len(combs_valid)}")

    # Create a DataFrame with the pairs and save
    combs_valid_df = pd.DataFrame(combs_valid, columns=['alpha', 'beta'])
    combs_valid_df.to_csv(Path(project_dir, "data", f"hetero-{hetero_label}_csParamCombs_valid.csv"), 
                          index=None)