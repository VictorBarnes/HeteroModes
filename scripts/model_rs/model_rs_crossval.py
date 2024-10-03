import os
import h5py
import argparse
import itertools
import numpy as np
import nibabel as nib
from joblib import Parallel, delayed
from dotenv import load_dotenv
from sklearn.model_selection import KFold
from neuromaps.datasets import fetch_atlas
from brainspace.utils.parcellation import reduce_by_labels
from heteromodes import HeteroSolver
from heteromodes.utils import load_hmap, pad_sequences
from heteromodes.restingstate import simulate_bold, calc_fc_fcd, evaluate_model

load_dotenv()
PROJ_DIR = os.getenv("PROJ_DIR")
SURF_LH = os.getenv("SURF_LH")
GLASSER360_LH = os.getenv("GLASSER360_LH")

def run_model(run, evals, emodes, parc, medmask, args, params, emp_results, B=None, return_all=False):
    # Load external input, run model and parcellate
    ext_input = np.load(f"{PROJ_DIR}/data/resting_state/extInput_den-{args.den}_randseed-{run}.npy")
    bold_model = simulate_bold(evals, emodes, ext_input, solver_method='Fourier', 
                               eig_method='orthonormal', r=params[1], gamma=params[2], B=B)
    bold_model = reduce_by_labels(bold_model, parc[medmask], axis=1)
    
    # Compute model FC and FCD
    fc_model, fcd_model = calc_fc_fcd(bold_model, tr=0.72, filter=False)
    model_results = {"fc": fc_model, "fcd": fcd_model}

    # Evaluate model
    edge_fc, node_fc, fcd, fc, fcd_dist = evaluate_model(emp_results, model_results)

    # Allow option to only return edge_fc, node_fc, fcd for memory efficiency
    if return_all:
        return edge_fc, node_fc, fcd, fc, fcd_dist
    else:
        return edge_fc, node_fc, fcd

# TODO: combine params and args into a single object
def training_job(surf, hmap, parc, medmask, params, args, emp_results):
    # Calculate modes
    solver = HeteroSolver(
        surf=surf,
        hmap=hmap,
        medmask=medmask,
        alpha=params[0],
    )
    evals, emodes = solver.solve(n_modes=args.n_modes, fix_mode1=True, standardise=True)

    # Parallelize the iterations
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(run_model)(run, evals, emodes, parc, medmask, args, params, emp_results, 
                           B=solver.mass, return_all=False)
        for run in range(args.n_runs)
    )
    edge_fcs, node_fcs, fcds = zip(*results)

    return np.array(edge_fcs), np.array(node_fcs), np.array(fcds)

def main():
    parser = argparse.ArgumentParser(description="Model resting-state fMRI BOLD data and evaluate against empirical data.")
    parser.add_argument("--hmap_label", type=str, default=None, help="The label of the heterogeneity map. Defaults to None (indicating homogeneity)")
    parser.add_argument("--id", type=int, help="The id of the run for saving outputs.")
    parser.add_argument("--scale_method", type=str, default="zscore", help="The scaling method for the heterogeneity map. Defaults to `zscore`.")
    parser.add_argument("--aniso_method", type=str, default="hetero", help="The method to calculate the modes. Defaults to `aniso`.")
    parser.add_argument("--n_runs", type=int, default=5, help="The number of runs to simulate. Defaults to 50.")
    parser.add_argument("--n_modes", type=int, default=500, help="The number of modes to calculate. Defaults to 500.")
    parser.add_argument("--n_splits", type=int, default=5, help="The number of splits for cross-validation. Defaults to 5.")
    parser.add_argument("--n_subjs", type=int, default=384, help="The number of subjects in the empirical data. Defaults to 384.")
    parser.add_argument("--n_jobs", type=int, default=-1, help="The number of CPUs for parallelization. Defaults to -1")
    parser.add_argument("--alpha_step", type=float, default=0.2, help="The step size for alpha values. Defaults to 0.5.")
    parser.add_argument("--den", type=str, default="32k", help="The density of the surface. Defaults to `32k`.")
    parser.add_argument("--parc", type=str, default="hcpmmp1", help="The parcellation to use to downsample the BOLD data.")
    args = parser.parse_args()

    # Get surface, medial mask and parcellation files
    fslr = fetch_atlas(atlas='fsLR', density=args.den)
    surf = str(fslr['midthickness'][0])

    out_dir = f'{PROJ_DIR}/results/model_rs/crossval/id-{args.id}'

    # Get hmap and alpha values
    if args.hmap_label is None:
        hmap = None
        param_combs = [(0, 28.9, 0.116)]
    else:
        # If hmap_label is null, load null map
        if args.hmap_label[:4] == "null":
            null_id = int(args.hmap_label.split('-')[1])
            hmap = np.load(f"{PROJ_DIR}/data/nulls/data-myelinmap_space-fsLR_den-{args.den}_hemi-L_nmodes-500_nnulls-5000_nulls_resample-True.npy")
            hmap = hmap[null_id, :]

            out_dir = out_dir + "/nulls"
        # Otherwise assume hmap_label is a valid label
        else:
            hmap = load_hmap(args.hmap_label, den=args.den)

        alpha_vals = np.arange(-2, 2 + args.alpha_step, args.alpha_step)
        alpha_vals = alpha_vals[alpha_vals != 0] # Remove 0 since that is the homogeneous case
        r_vals = [28.9] #np.arange(10, 60, 10)
        gamma_vals = [0.116]

        param_combs = list(itertools.product(alpha_vals, r_vals, gamma_vals))

    print(f"Number of parameter combinations: {len(param_combs)}")

    parc_file = f"{PROJ_DIR}/data/parcellations/Q1-Q6_RelatedParcellation210.L.CorticalAreas_dil_Colors.{args.den}_fs_LR.label.gii"
    parc = nib.load(parc_file).darrays[0].data.astype(int)
    medmask = np.where(parc != 0, True, False)

    # Load empirical BOLD data
    # TODO: make this generalizable to different parcellations (parc_name = os.path.basename(args.parc_lh).split('_')[0])
    # bold_data = h5py.File(f"{PROJ_DIR}/data/empirical/HCP_unrelated-445_rfMRI_hemi-L_nsubj-{args.n_subjs}_parc-glasser360_BOLD.hdf5", 'r')
    # bold_emp = bold_data['bold']
    # subj_ids = bold_data['subj_ids']
    # _, _, nsubjs = np.shape(bold_emp)

    # # Parallelize the calculation of FC and FCD
    # print("Calculating empirical FC and FCD...")
    # results = Parallel(n_jobs=args.n_jobs, verbose=1)(
    #     delayed(calc_fc_fcd)(bold=bold_emp[:, :, subj], tr=0.72, filter=False)
    #     for subj in range(nsubjs)
    # )
    # fc_emp_all, fcd_emp_all = zip(*results)
    # fc_emp_all = np.dstack(fc_emp_all)
    # fcd_emp_all = np.array(fcd_emp_all)

    # Initialise output arrays
    best_combs = []

    edge_fc_train = np.empty((args.n_splits, len(param_combs), args.n_runs))
    node_fc_train = np.empty((args.n_splits, len(param_combs), args.n_runs))
    fcd_train = np.empty((args.n_splits, len(param_combs), args.n_runs))
    combined_metric_train = np.empty((args.n_splits, len(param_combs), args.n_runs))
    train_subjs_split = []

    edge_fc_test = np.empty((args.n_splits, args.n_runs))
    node_fc_test = np.empty((args.n_splits, args.n_runs))
    fcd_test = np.empty((args.n_splits, args.n_runs))
    fc_test = []
    fcd_dist_test = []
    combined_metric_test = np.empty((args.n_splits, args.n_runs))
    test_subjs_split = []
    
    kf = KFold(n_splits=args.n_splits, shuffle=False)
    for i, (train_index, test_index) in enumerate(kf.split(np.arange(args.n_subjs))):
        print(f"\n==========\nSplit {i+1}/{args.n_splits}\n==========")

        # Select train and test bold data
        emp_file = f"{PROJ_DIR}/data/empirical/HCP_unrelated-445_rfMRI_hemi-L_nsubj-{args.n_subjs}_parc-glasser360_FC_FCD.hdf5"
        with h5py.File(emp_file, "r") as f:
            # Load empirical FC and FCD
            fc_emp_train = np.mean(f["fc_matrices"][:, :, train_index], axis=2)
            fc_emp_test = np.mean(f["fc_matrices"][:, :, test_index], axis=2)
            fcd_emp_train = f["fcd_distributions"][train_index, :]
            fcd_emp_test = f["fcd_distributions"][test_index, :]

            # Get subject ids
            train_subjs_split.append(f["subj_ids"][train_index])
            test_subjs_split.append(f["subj_ids"][test_index])

        if args.hmap_label is not None:
            print(f"\nTraining...\n=====================")
            
            results_train = Parallel(n_jobs=args.n_jobs, verbose=1)(
                delayed(training_job)(
                    surf=surf,
                    hmap=hmap,
                    parc=parc,
                    medmask=medmask,
                    params=params,
                    args=args,
                    emp_results={"fc": fc_emp_train, "fcd": fcd_emp_train},
                )
                for params in param_combs
            )
            edge_fc_train[i, :, :], node_fc_train[i, :, :], fcd_train[i, :, :] = zip(*results_train)
            combined_metric_train[i, :, :] = (
                np.array(edge_fc_train[i, :, :]) + 
                np.array(node_fc_train[i, :, :]) + 
                (1 - np.array(fcd_train[i, :, :]))
            )

            # Get best training results (average across runs)
            best_combs_ind = np.argmax(np.mean(combined_metric_train[i, :, :], axis=1))
            best_alpha, best_r, best_gamma = param_combs[best_combs_ind]

            print("\nTraining results:\n-----------------")
            print(f"    Best alpha: {best_alpha:.3g}")
            print(f"    Best r: {best_r:.3g}")
            print(f"    Best gamma: {best_gamma:.3g}")
            print(f"    Best combined metric: {np.max(np.mean(combined_metric_train[i, :, :], axis=1)):.4g}")
            print(f"    Best edge-level FC: {np.mean(edge_fc_train[i, best_combs_ind, :]):.3g}")
            print(f"    Best node-level FC: {np.mean(node_fc_train[i, best_combs_ind, :]):.3g}")
            print(f"    Best FCD: {np.mean(fcd_train[i, best_combs_ind, :]):.3g}")
        else:
            best_alpha = 0
            best_r = 28.9
            best_gamma = 0.116
        best_combs.append((best_alpha, best_r, best_gamma))
        
        # Evaluate on test set
        print(f"\nTesting (using alpha = {best_alpha:.1f}, r = {best_r:.1f})...\n=====================")
        # Calculate modes
        solver = HeteroSolver(
            surf=surf,
            medmask=medmask,
            hmap=hmap,
            alpha=best_alpha,
        )
        evals, emodes = solver.solve(n_modes=args.n_modes, fix_mode1=True, standardise=True)
        # Run model
        results_test = Parallel(n_jobs=args.n_runs)(
            delayed(run_model)(
                run=run, 
                evals=evals,
                emodes=emodes,
                parc=parc, 
                medmask=medmask, 
                args=args,
                params=(best_alpha, best_r, best_gamma),
                emp_results={"fc": fc_emp_test, "fcd": fcd_emp_test},
                B=solver.mass,
                return_all=True)
            for run in range(args.n_runs)
        )
        edge_fc_test[i, :], node_fc_test[i, :], fcd_test[i, :], fcs, fcd_dists = zip(*results_test)
        fc_test.append(fcs)
        fcd_dist_test.append(fcd_dists)
        combined_metric_test[i, :] = (
            np.array(edge_fc_test[i, :]) + 
            np.array(node_fc_test[i, :]) + 
            (1 - np.array(fcd_test[i, :]))
        )

        print(f"\nTest results:\n--------------")
        print(f"    Combined metric: {np.mean(combined_metric_test[i, :]):.4g}")
        print(f"    Edge-level FC: {np.mean(edge_fc_test[i, :]):.3g}")
        print(f"    Node-level FC: {np.mean(node_fc_test[i, :]):.3g}")
        print(f"    FCD: {np.mean(fcd_test[i, :]):.3g}")

    print("\n==========\nFinal results\n==========")
    print(f"Best alpha: {np.mean(best_combs, axis=0)[0]}")
    print(f"Best r: {np.mean(best_combs, axis=0)[1]}")
    print(f"Best combined metric: {np.max(np.mean(combined_metric_test, axis=1)):.4g}") 
    print(f"Best edge-level FC: {np.mean(edge_fc_test):.3g}")
    print(f"Best node-level FC: {np.mean(node_fc_test):.3g}")
    print(f"Best FCD: {np.mean(fcd_test):.3g}")

    # Create output directory if it doesn't exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_file = f"{args.hmap_label}_results.hdf5"
    # If file exists: append number to folder name to avoid overwriting
    if os.path.exists(f"{out_dir}/{out_file}"):
        i = 1
        while os.path.exists(f"{out_dir}/{out_file}"):
            out_file = f"{args.hmap_label}_results_{i}.hdf5"
            i += 1
    out_path = f"{out_dir}/{out_file}"
    print(f"Output path: {out_path}")

    # Save results
    with h5py.File(out_path, 'w') as f:
        # Write metadata to file
        for key, value in vars(args).items():
            if value is None:
                f.attrs[key] = "None"
            else:
                f.attrs[key] = value

        # Write outputs to file
        f.create_dataset('edge_fc_train', data=edge_fc_train)
        f.create_dataset('node_fc_train', data=node_fc_train)
        f.create_dataset('fcd_train', data=fcd_train)
        f.create_dataset('combined_metric_train', data=combined_metric_train)

        f.create_dataset('edge_fc_test', data=edge_fc_test)
        f.create_dataset('node_fc_test', data=node_fc_test)
        f.create_dataset('fcd_test', data=fcd_test)
        f.create_dataset('combined_metric_test', data=combined_metric_test)
        f.create_dataset('fc_test', data=np.dstack(fc_test))
        f.create_dataset('fcd_dist_test', data=np.vstack(fcd_dist_test).T)

        f.create_dataset('combs', data=param_combs)
        f.create_dataset('best_combs', data=best_combs)

        # Pad train_subjs_split and test_subjs_split with -1 to have the same length
        f.create_dataset('train_subjs_split', data=pad_sequences(train_subjs_split))
        f.create_dataset('test_subjs_split', data=pad_sequences(test_subjs_split))

if __name__ == "__main__":
    main()
    