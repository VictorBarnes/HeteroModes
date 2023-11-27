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
CMEAN = 3352.4

def calc_determinant(matrix):
    mp.dps = 25
    matrix_mp = [mp.mpf(float(elem)) for elem in matrix]
    det = mp.mpf(1)
    for elem in matrix_mp:
        det *= elem

    return det

def scale_zscore(image, alpha=1.0, cmean=3352.4):
    """Z-score and shift data by c_mean value (from Pang2023)"""
    return cmean + alpha*stats.zscore(image)

def scale_cmean(image, alpha=1.0, cmean=3352.4):
    """Scale heterogeneity map to be a variation around the mean propagation speed (from Pang2023)
    while preserving the mean (i.e. cmean is still the mean of the new distribution)."""
    # Calculate normalized density (rho)
    scaler = MinMaxScaler(feature_range=(0, 1))
    rho = scaler.fit_transform(image.reshape(-1, 1)).flatten()

    return cmean + alpha * cmean * (rho - np.mean(rho))


if __name__ == '__main__':
    config_file = "/fs04/kg98/vbarnes/HeteroModes/scripts/config.json"
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
    alpha_vals = np.arange(0.1, 1.1, 0.1)

    # Load map chosen to paramaterize heterogeneity
    if hetero_label is not None:
        # Load heterogeneity map
        mode_match = re.search(r"mode(\d+)", hetero_label)
        if mode_match:
            mode_id = int(mode_match.group(1)) - 1
            hetero_modes = np.loadtxt(f"{EMODE_DIR}/hetero-None_atlas-{atlas}_space-{space}_"
                                        f"den-{den}_surf-{surf_type}_hemi-{hemi}_n-{n_modes}_"
                                        f"maskMed-{mask_medial}_emodes.txt")
            hetero_map = hetero_modes[:, mode_id]
        elif hetero_label in config["hetero_maps"]:
            # Load heterogeneity map
            hetero_file = config["hetero_maps"][hetero_label]
            hetero_map = nib.load(hetero_file).agg_data()
        else:
            hetero_map = np.loadtxt(f"./data/hetero-{hetero_label}_surf-{surf_type}.txt")
    else:
        # No heterogeneity is encoded by an array of ones
        hetero_map = np.ones(DENSITIES["32k"])
        alpha_vals = [None]
        scale = None

    # Load surface template and medial wall mask
    surf = mesh_io.read_surface(f"{SURF_DIR}/atlas-{atlas}_space-{space}_den-{den}_"
                                f"surf-{surf_type}_hemi-{hemi}_surface.vtk")
    medial = np.loadtxt(f"{SURF_DIR}/atlas-{atlas}_space-{space}_den-{den}_hemi-{hemi}_"
                        f"medialMask.txt").astype(bool)
    medial_inds = np.array([i for i, med in enumerate(medial) if med])

    # Mask surface template and heterogeneous map
    if mask_medial:
        # Mask surface template
        surf_masked = mesh_operations.mask_points(surf, medial)
        v = surf_masked.Points
        t = np.reshape(surf_masked.Polygons, [surf_masked.n_cells, 4])[:,1:4]
        mesh = TriaMesh(v, t)
        # Mask heterogeneous map
        hetero_map = hetero_map[medial]
    else:
        # Initialise mesh without masking medial wall
        mesh = TriaMesh.read_vtk(getattr(surf, hemi))

    # Set up C matrix (only one column since alpha is None)
    if hetero_label is None:
        # Calculate C matrix
        C = hetero_map * CMEAN
        # Ensure C is doesn't have negative values
        assert np.min(C) >= 0.0, f"Minimum value of C is negative: {np.min(C)}"
        # Each term in C needs to be squared (according to the NFT equation)
        C = (C ** 2).reshape(-1, 1)
    else:
        # Set up C matrix (each col defines the heterogeneity in propagation speed across the cortex)
        C = np.zeros((len(hetero_map), len(alpha_vals)))
        for i, alpha in enumerate(alpha_vals):
            # Scale the heterogeneity map to vary around CMEAN (while still preserving CMEAN)
            C[:, i] = scale_cmean(hetero_map, alpha=alpha, cmean=CMEAN)
            # Ensure C doesn't have negative values
            assert np.min(C[:, i]) >= 0.0, f"Minimum value of C is negative: {np.min(C[:, i])}"
            # Each term in C needs to be squared (according to the NFT equation)
            C[:, i] **= 2

    # Solve heterogeneous Helmholtz equation (using each column of C as the heterogeneity)
    C_reshaped = np.zeros((surf.n_points, C.shape[1]))
    for i in range(C.shape[1]):
        alpha = f"{alpha_vals[i]:.1f}" if isinstance(alpha_vals[i], float) else alpha_vals[i]
        print(f"Atlas: {atlas} | Space: {space} | Density: {den} | Surface: {surf_type} | "
            f"Hetero: {hetero_label} | Scaling: {scale} | alpha: {alpha} | cmean: {CMEAN}")

        # Calculate the hetero value for each triangle by taking the average of the values at its vertices
        C_tfunc = mesh.map_vfunc_to_tfunc(C[:, i])
        # Initialise FEM solver and solve for eigenvalues and eigenmodes
        fem = Solver(mesh, aniso=(0, 0), hetero=C_tfunc)
        evals, emodes = fem.eigs(k=n_modes)

        # Reshape emodes to match vertices of original surface
        if mask_medial:
            emodes_reshaped = np.zeros([surf.n_points, n_modes])
            for mode in range(n_modes):
                emodes_reshaped[medial_inds, mode] = emodes[:, mode]
            # Reshape propagation speed map
            C_reshaped[medial_inds, i] = C[:, i]

        if save_results:
            print("Saving eigenmodes and eigenvalues...")
            # Set output file names and save
            if hetero_label is None:
                desc = f"hetero-{hetero_label}_atlas-{atlas}_space-{space}_den-{den}_surf-{surf_type}_" \
                    f"hemi-{hemi}_n-{n_modes}_maskMed-{mask_medial}"
            else:
                desc = f"hetero-{hetero_label}_atlas-{atlas}_space-{space}_den-{den}_surf-{surf_type}_"\
                    f"hemi-{hemi}_n-{n_modes}_scale-{scale}_alpha-{alpha}_maskMed-{mask_medial}"

            evals_savefile = Path(EMODE_DIR, f"{desc}_evals.txt")
            emodes_savefile = Path(EMODE_DIR, f"{desc}_emodes.txt")
            cmap_savefile = Path(EMODE_DIR, "cmaps", f"{desc}_cmap.txt")
            np.savetxt(evals_savefile, evals)
            np.savetxt(emodes_savefile, emodes_reshaped)
            np.savetxt(cmap_savefile, C_reshaped[:, i])
