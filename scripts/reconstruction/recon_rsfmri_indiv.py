import os
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from heteromodes.eigentools import calc_eigenreconstruction
from heteromodes import EigenSolver
from heteromodes.utils import load_hmap
from neuromaps.datasets import fetch_fslr
from joblib import Parallel, delayed
from dotenv import load_dotenv
from memory_profiler import profile
import argparse


load_dotenv()
PROJ_DIR = os.getenv("PROJ_DIR")

def recon_subject(bold_subj, method="orthogonal"):
    # Load individual data
    ...

    # Calculate modes
    solver = EigenSolver(surf, hmap, medmask, alpha=args.alpha)
    _, emodes = solver.solve(n_modes=args.n_modes, fix_mode1=True, standardise=True)

    # remove nan timepoints
    bold_subj = bold_subj[:, ~np.isnan(bold_subj).any(axis=0)]

    scaler = StandardScaler()
    bold_z = scaler.fit_transform(bold_subj.T).T

    _, _, fc_corr = calc_eigenreconstruction(
        bold_z, 
        emodes,
        method=method,
        modesq=None,
        mass=solver.mass,
        data_type="timeseries"
    )

    return fc_corr

# @profile
def main():
    # Parse args
    parser = argparse.ArgumentParser(description='Reconstruct task fMRI maps')
    parser.add_argument('--hmap_label', type=str, default=None, help='The label of the heterogeneity map')
    parser.add_argument('--alpha', type=float, default=0, help='The alpha value for scaling the heterogeneity map.')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of jobs to run in parallel')
    parser.add_argument('--n_modes', type=int, default=100, help='Number of modes')
    parser.add_argument('--method', type=str, default="orthogonal", help="Method for eigen-reconstruction")
    args = parser.parse_args()

    print("Loading data...")
    with h5py.File(f"{PROJ_DIR}/data/empirical/HCP_nsubj-255_bold_parc-None_fsLR4k_hemi-L.hdf5", "r") as f:
        medmask = np.array(f["medmask"][:], dtype=bool)
        bold = np.array(f["bold"][medmask, :, :])  

    print("Reconstructing data...")
    fc_corr = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(recon_subject)(
            bold_subj=bold[:, :, i],
            method=args.method,
        ) for i in range(bold.shape[2])
    )
    fc_corr = np.array(fc_corr)
    print(fc_corr.shape)

    print("Saving results...")
    out_dir = f"{PROJ_DIR}/results/reconstruction/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with h5py.File(f"{out_dir}/rsFC_recon_hmap-{str(args.hmap_label)}_alpha-{args.alpha:.2f}_method-{args.method}_nmodes-{args.n_modes}.h5", "w") as f:
        f.attrs["hmap_label"] = str(args.hmap_label)
        f.attrs["alpha"] = np.float32(args.alpha)
        f.attrs["n_modes"] = int(args.n_modes)

        f.create_dataset("fc_recon_corr", data=fc_corr)


if __name__ == "__main__":
    main()
