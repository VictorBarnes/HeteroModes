import os
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from heteromodes.eigentools import calc_eigenreconstruction
from neuromaps.datasets import fetch_fslr
from joblib import Parallel, delayed
from memory_profiler import profile
import argparse
from heteromodes import EigenSolver
from heteromodes.utils import load_hmap, load_project_env
from heteromodes.restingstate import filter_bold

load_project_env()
PROJ_DIR = os.getenv("PROJ_DIR")

def recon_subject(bold_subj, emodes, mass, method="orthogonal", metric="pearsonr"):
    # remove nan timepoints
    bold_subj = bold_subj[:, ~np.isnan(bold_subj).any(axis=0)]

    scaler = StandardScaler()
    bold_z = scaler.fit_transform(bold_subj.T).T

    # Bandpass filter the data
    bold_zf = filter_bold(bold_z, tr=0.72, lowcut=0.01, highcut=0.1)

    _, _, fc_score = calc_eigenreconstruction(
        bold_zf, 
        emodes,
        method=method,
        modesq=None,
        mass=mass,
        data_type="timeseries",
        metric=metric
    )

    return fc_score

# @profile
def main():
    # Parse args
    parser = argparse.ArgumentParser(description='Reconstruct task fMRI maps')
    parser.add_argument('--id', type=str, default="recon_rsFC", help='The ID of the reconstruction')
    parser.add_argument('--hmap_label', type=lambda x: None if x.lower() == "none" else x, default=None, help="The label of the heterogeneity map. Defaults to None (indicating homogeneity)")
    parser.add_argument('--alpha', type=float, default=0, help='The alpha value for scaling the heterogeneity map.')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of jobs to run in parallel')
    parser.add_argument('--n_modes', type=int, default=200, help='Number of modes')
    parser.add_argument('--method', type=str, default="orthogonal", help="Method for eigen-reconstruction")
    parser.add_argument('--metric', type=str, default="pearsonr", help="Metric for calculating reconstruction accuracy")
    parser.add_argument('--scaling', type=str, default="exponential", help="Scaling function to apply to the heterogeneity map")
    parser.add_argument('--q_norm', type=str, default="normal", help="Type of distribution to match to when doing the quantile normalisation")
    args = parser.parse_args()

    print("Loading data...")
    with h5py.File(f"{PROJ_DIR}/data/empirical/HCP_nsubj-255_bold_parc-None_fsLR4k_hemi-L.h5", "r") as f:
        medmask = np.array(f["medmask"][:], dtype=bool)
        bold = np.array(f["bold"][medmask, :, :100])

    # Calculate modes
    surf = fetch_fslr("4k")["midthickness"][0]

    print("Calculating eigenmodes...")
    if args.hmap_label is not None:
        hmap = load_hmap(args.hmap_label, "4k")
    else:
        hmap = None
    solver = EigenSolver(surf, medmask=medmask, hetero=hmap, alpha=args.alpha, scaling=args.scaling,
                         q_norm=args.q_norm)
    evals, emodes = solver.solve(n_modes=args.n_modes, fix_mode1=True)

    print("Reconstructing data...")
    fc_score = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(recon_subject)(
            bold_subj=bold[:, :, i],
            emodes=emodes,
            method=args.method,
            mass=solver.mass,
            metric=args.metric
        ) for i in range(bold.shape[2])
    )
    fc_score = np.array(fc_score)
    print(fc_score.shape)

    print("Saving results...")
    out_dir = f"{PROJ_DIR}/results/reconstruction/id-{args.id}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with h5py.File(f"{out_dir}/rsFC_recon_hmap-{str(args.hmap_label)}_alpha-{args.alpha:.2f}.h5", "w") as f:
        f.attrs["hmap_label"] = str(args.hmap_label)
        f.attrs["alpha"] = np.float32(args.alpha)
        f.attrs["n_modes"] = int(args.n_modes)

        f.create_dataset("fc_recon_score", data=fc_score)
        # Save evals in case we want to reconstruct by number of modes up to a given eigenvalue
        f.create_dataset("evals", data=evals)


if __name__ == "__main__":
    main()
