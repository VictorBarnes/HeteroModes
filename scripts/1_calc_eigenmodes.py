#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate heterogeneous eigenmodes of a surface by solving the heterogeneous Helmholtz equation.
"""

import json
import re
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from mpmath import mp
from lapy import Solver, TriaMesh
from brainspace.mesh import mesh_io, mesh_operations


# Global variables
DENSITIES = {"32k": 32492}
CMEAN = 3352.4  # 28.9

def calc_determinant(matrix):
    mp.dps = 25
    matrix_mp = [mp.mpf(float(elem)) for elem in matrix]
    det = mp.mpf(1)
    for elem in matrix_mp:
        det *= elem
    return det

def scale_norm(image):
    """Scale between 0 and 1"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(image.reshape(-1, 1)).flatten()

def scale_zscore(image, alpha=1.0, cmean=3352.4):
    """Z-score and shift data by c_mean value (from Pang2023)"""
    return cmean + alpha * stats.zscore(image)

def scale_cmean(image, alpha=1.0, cmean=3352.4):
    """Scale heterogeneity map to be a variation around the mean propagation speed (from Pang2023)
    while preserving the mean (i.e. cmean is still the mean of the new distribution)."""
    # Calculate normalized density (rho)
    scaler = MinMaxScaler(feature_range=(0, 1))
    rho = scaler.fit_transform(image.reshape(-1, 1)).flatten()
    return cmean + alpha * cmean * (rho - np.mean(rho))


if __name__ == '__main__':
    config_file = "scripts/config.json"
    with open(config_file, encoding="UTF-8") as f:
        config = json.load(f)
    save_results = config["save_results"]
    EMODE_DIR = config["emode_dir"]
    SURF_DIR = config["surface_dir"]
    RESULTS_DIR = config["results_dir"]
    atlas = config["atlas"]
    space = config["space"]
    den = config["den"]
    surf_type = config["surf"]
    hemi = config["hemi"]
    n_modes = config["n_modes"]
    mask_medial = config["mask_medial"]
    hetero_label = config["hetero_label"]
    scale = config["scale"]
    alpha_vals = config["alpha_vals"]
    beta_vals = config["beta_vals"]

    # Load map chosen to paramaterize heterogeneity
    if hetero_label is None:
        # No heterogeneity is encoded by an array of ones
        hetero_map = np.ones(DENSITIES["32k"])
    else:
        # Load heterogeneity map
        hetero_file = config["hetero_maps"][hetero_label]
        hetero_map = nib.load(hetero_file).agg_data()

        # TODO: try allowing alpha and vary to beta and set up a double for loop when calculating cs
        # Set alpha and beta values
        if len(alpha_vals) > 1 and len(beta_vals) > 1:
            raise ValueError("alpha_vals and beta_vals cannot both have a length larger than one")
        # Set up parameter that will be varied (can only vary one of these for now)
        if len(alpha_vals) > 1:
            param = "alpha"
            param_vals = alpha_vals
        elif len(beta_vals) > 1:
            param = "beta"
            param_vals = beta_vals     

    # Load surface template and medial wall mask
    surf = mesh_io.read_surface(f"{SURF_DIR}/atlas-{atlas}_space-{space}_den-{den}_"
                                f"surf-{surf_type}_hemi-{hemi}_surface.vtk")
    medial_mask = np.loadtxt(f"{SURF_DIR}/atlas-{atlas}_space-{space}_den-{den}_hemi-{hemi}_"
                        f"medialMask.txt").astype(bool)
    cortex_inds = np.array([i for i, med in enumerate(medial_mask) if med])

    # Mask surface template and heterogeneous map
    if mask_medial:
        # Mask surface template
        surf_masked = mesh_operations.mask_points(surf, medial_mask)
        v = surf_masked.Points
        t = np.reshape(surf_masked.Polygons, [surf_masked.n_cells, 4])[:,1:4]
        mesh = TriaMesh(v, t)
        # Mask heterogeneous map
        hetero_map = hetero_map[medial_mask]
    else:
        # Initialise mesh without masking medial wall
        mesh = TriaMesh.read_vtk(getattr(surf, hemi))

    # Set up C matrix (only one column since alpha is None)
    if hetero_label is None:
        # Calculate C matrix
        cs = hetero_map * CMEAN
        # Ensure C is doesn't have negative values
        assert np.min(cs) >= 0.0, f"Minimum value of cs is negative: {np.min(cs)}"
        # Each term in C needs to be squared (according to the NFT equation)
        cs = (cs ** 2).reshape(-1, 1)
    else:
        # Set up cs matrix (each col defines the heterogeneity in propagation speed across the cortex)
        cs = np.zeros((len(hetero_map), len(param_vals)))
        for i, val in enumerate(param_vals):
            alpha = val if param == "alpha" else alpha_vals[0]
            beta = val if param == "beta" else beta_vals[0]
            # Scale propagation speed
            scaler = MinMaxScaler(feature_range=(0, 1))
            rho = scaler.fit_transform(hetero_map.reshape(-1, 1)).flatten()
            cs[:, i] = CMEAN * (1 + alpha*(rho - np.mean(rho)))**beta

            # Ensure C doesn't have negative values
            assert np.min(cs[:, i]) >= 0.0, f"cs cannot have negative values"
            # Each term in C needs to be squared (according to the NFT equation)
            cs[:, i] **= 2

    # Solve heterogeneous Helmholtz equation (using each column of C as the heterogeneity)
    cs_reshaped = np.zeros((surf.n_points, cs.shape[1]))
    for i in range(cs.shape[1]):
        # Calculate the hetero value for each triangle by taking the average of the values at its vertices
        cs_tri = mesh.map_vfunc_to_tfunc(cs[:, i])
        # Initialise FEM solver and solve for eigenvalues and eigenmodes
        fem = Solver(mesh, aniso=(0, 0), hetero=cs_tri)
        evals, emodes = fem.eigs(k=n_modes)

        # Reshape emodes to match vertices of original surface
        if mask_medial:
            emodes_reshaped = np.zeros([surf.n_points, n_modes])
            for mode in range(n_modes):
                emodes_reshaped[cortex_inds, mode] = emodes[:, mode]
            # Reshape propagation speed map
            cs_reshaped[cortex_inds, i] = cs[:, i]

        if save_results:
            print("Saving eigenmodes and eigenvalues...")
            # Set output file names and save
            if hetero_label is None:
                print(f"Atlas: {atlas} | Space: {space} | Density: {den} | Surface: {surf_type} | "
                    f"Hetero: {hetero_label} | Scaling: {scale} | cmean: {CMEAN} | nmodes: {n_modes}")
                desc = f"hetero-{hetero_label}_atlas-{atlas}_space-{space}_den-{den}_surf-{surf_type}_" \
                    f"hemi-{hemi}_n-{n_modes}_scale-{scale}_maskMed-{mask_medial}"
            else:
                alpha = param_vals[i] if param == "alpha" else alpha_vals[0]
                beta = param_vals[i] if param == "beta" else beta_vals[0]
                print(f"Atlas: {atlas} | Space: {space} | Density: {den} | Surface: {surf_type} | "
                    f"Hetero: {hetero_label} | Scaling: {scale} | alpha: {alpha} | beta: {beta} | "
                    f"cmean: {CMEAN} | nmodes: {n_modes}")
                desc = f"hetero-{hetero_label}_atlas-{atlas}_space-{space}_den-{den}_surf-{surf_type}_"\
                    f"hemi-{hemi}_n-{n_modes}_scale-{scale}_alpha-{alpha}_beta-{beta}_maskMed-{mask_medial}"

            evals_savefile = Path(EMODE_DIR, f"{desc}_evals.txt")
            emodes_savefile = Path(EMODE_DIR, f"{desc}_emodes.txt")
            cmap_savefile = Path(EMODE_DIR, "cmaps", f"{desc}_cmap.txt")
            np.savetxt(evals_savefile, evals)
            np.savetxt(emodes_savefile, emodes_reshaped)
            np.savetxt(cmap_savefile, cs_reshaped[:, i])
