import os
import h5py
import argparse
import itertools
import numpy as np
import nibabel as nib
from joblib import Parallel, delayed
from dotenv import load_dotenv
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from neuromaps.datasets import fetch_atlas
from brainspace.utils.parcellation import reduce_by_labels
from heteromodes import HeteroSolver
from heteromodes.utils import load_hmap, pad_sequences
from heteromodes.restingstate import simulate_bold, calc_phase, calc_edge_and_node, calc_phase_delay_combined

load_dotenv()
PROJ_DIR = os.getenv("PROJ_DIR")


def calc_fc_and_phase(bold, tr, band_freq, parc=None):
    # Z-score bold data
    scaler = StandardScaler()
    bold_z = scaler.fit_transform(bold.T).T

    # Compute combined phase delay on vertex data
    phase = calc_phase(bold_z, tr=tr, lowcut=band_freq[0], highcut=band_freq[1])

    # Compute FC on parcellated data
    if parc is not None:
        bold_z = reduce_by_labels(bold_z, parc, axis=1)
    fc = np.corrcoef(bold_z)

    return fc, phase

def run_model(run, evals, emodes, parc, args, params, B=None):
    # Load external input, run model and parcellate
    ext_input = np.load(f"{PROJ_DIR}/data/resting_state/extInput_parc-{args.parc}_den-{args.den}_hemi-L_randseed-{run}.npy")
    bold_model = simulate_bold(evals, emodes, ext_input, solver_method='Fourier', 
                               eig_method='orthonormal', r=params[1], gamma=params[2], B=B)

    # Z-score bold data
    scaler = StandardScaler()
    bold_model = scaler.fit_transform(bold_model.T).T
    
    # Compute model FC and combined phase delay
    fc_model, phase_model = calc_fc_and_phase(
        bold=bold_model, 
        tr=0.72, 
        band_freq=(args.band_freq[0], args.band_freq[1]),
        parc=parc
    )

    return fc_model, phase_model

def run_and_evaluate(surf, medmask, hmap, parc, params, args, emp_results, return_all=False):
    # Calculate modes
    solver = HeteroSolver(
        surf=surf,
        hmap=hmap,
        medmask=medmask,
        alpha=params[0],
    )
    evals, emodes = solver.solve(n_modes=args.n_modes, fix_mode1=True, standardise=True)

    # Run model but only return evaluation metrics
    if parc is not None:
        n_regions = len(np.unique(parc))
    else:
        n_regions = np.sum(medmask)
    fc_model = np.empty((n_regions, n_regions, args.n_runs))
    n_verts = np.sum(medmask)
    phase_model = np.empty((n_verts, 1200, args.n_runs))    # 1200 time points in HCP data
    for run in range(args.n_runs):
        fc_model[:, :, run], phase_model[:, :, run] = run_model(
            run=run, 
            evals=evals, 
            emodes=emodes, 
            parc=parc,
            args=args, 
            params=params, 
            B=solver.mass,
        )
    # Concatenate subjects/runs
    phase_model = phase_model.reshape(n_verts, -1)
    model_results = {"fc": fc_model, "phase": phase_model}

    # Calculate edge and node FC metric
    edge_fc_corr, node_fc_corr = calc_edge_and_node(
        empirical=emp_results, 
        model=model_results, 
    )

    # Calculate phase metric
    phase_emp_combined = calc_phase_delay_combined(emp_results["phase"], n_components=4)
    phase_model_combined = calc_phase_delay_combined(phase_model, n_components=4)
    if parc is not None:
        phase_emp_combined = reduce_by_labels(phase_emp_combined, parc, axis=0)
        phase_model_combined = reduce_by_labels(phase_model_combined, parc, axis=0)

    phase_corr = np.corrcoef(phase_emp_combined, phase_model_combined)[0, 1]

    if return_all:
        return edge_fc_corr, node_fc_corr, phase_corr, np.mean(fc_model, axis=2), phase_model_combined
    else:
        return edge_fc_corr, node_fc_corr, phase_corr


def main():
    parser = argparse.ArgumentParser(description="Model resting-state fMRI BOLD data and evaluate against empirical data.")
    parser.add_argument("--id", type=int, help="The id of the run for saving outputs.")
    parser.add_argument("--hmap_label", type=str, default=None, help="The label of the heterogeneity map. Defaults to None (indicating homogeneity)")
    parser.add_argument("--n_runs", type=int, default=5, help="The number of runs to simulate. Defaults to 50.")
    parser.add_argument("--n_modes", type=int, default=500, help="The number of modes to calculate. Defaults to 500.")
    parser.add_argument("--n_splits", type=int, default=5, help="The number of splits for cross-validation. Defaults to 5.")
    parser.add_argument("--n_subjs", type=int, default=384, help="The number of subjects in the empirical data. Defaults to 384.")
    parser.add_argument("--n_jobs", type=int, default=-1, help="The number of CPUs for parallelization. Defaults to -1")
    parser.add_argument('--alpha', type=float, nargs=3, metavar=('alpha_min', 'alpha_max', 'alpha_step'), help='The alpha_min, alpha_max, and alpha_step values for scaling the heterogeneity map.')
    parser.add_argument("--den", type=str, default="32k", help="The density of the surface. Defaults to `32k`.")
    parser.add_argument("--parc", type=str, default=None, help="The parcellation to use to downsample the BOLD data.")
    parser.add_argument("--band_freq", type=float, nargs=2, default=[0.01, 0.1], metavar=('low', 'high'), help="The low and high bandpass frequencies for filtering the BOLD data. Defaults to [0.01, 0.1].")
    parser.add_argument("--metrics", type=str, nargs='+', default=["edge_fc", "node_fc", "phase"], help="The metrics to use for evaluation. Defaults to ['edge_fc', 'node_fc', 'phase']")
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
        # If hmap_label is null, load null mapg
        if args.hmap_label[:4] == "null":
            null_id = int(args.hmap_label.split('-')[1])
            hmap = np.load(f"{PROJ_DIR}/data/nulls/data-myelinmap_space-fsLR_den-{args.den}_hemi-L_nmodes-500_nnulls-5000_nulls_resample-True.npy")
            hmap = hmap[null_id, :]

            out_dir = out_dir + "/nulls"
        # Otherwise assume hmap_label is a valid label
        else:
            hmap = load_hmap(args.hmap_label, den=args.den)

        alpha_min, alpha_max, alpha_step = args.alpha
        alpha_vals = np.arange(alpha_min, alpha_max + alpha_step, alpha_step)
        alpha_vals = alpha_vals[alpha_vals != 0] # Remove 0 since that is the homogeneous case
        r_vals = [28.9] #np.arange(10, 60, 10)
        gamma_vals = [0.116]

        param_combs = list(itertools.product(alpha_vals, r_vals, gamma_vals))

    print(f"Number of parameter combinations: {len(param_combs)}")

    # Load empirical BOLD data
    bold_data = h5py.File(f"{PROJ_DIR}/data/empirical/HCP_unrelated-445_rfMRI_hemi-L_nsubj-{args.n_subjs}_parc-None_fsLR{args.den}_hemi-L_BOLD.hdf5", 'r')
    bold_emp = bold_data['bold'][:]
    subj_ids = bold_data['subj_ids']
    
    # Load parcellation
    if args.parc is not None:
        try:
            parc_file = f"{PROJ_DIR}/data/parcellations/parc-{args.parc}_space-fsLR_den-{args.den}_hemi-L.label.gii"
            parc = nib.load(parc_file).darrays[0].data.astype(int)
        except:
            raise ValueError(f"Parcellation '{args.parc}' with den '{args.den}' not found.")
        medmask = np.where(parc != 0, True, False)
        parc = parc[medmask]
    else:
        parc = None
        medmask = np.where(bold_emp[:, 0, 0] != 0, True, False)

    bold_emp = bold_emp[medmask, :, :]
    nverts, _, nsubjs = np.shape(bold_emp)

    # Parallelize the calculation of FC and phase_delay
    print("Calculating empirical FC and Phase Delay...")
    results = Parallel(n_jobs=args.n_jobs, verbose=1)(
        delayed(calc_fc_and_phase)(
            bold=bold_emp[:, :, subj], 
            tr=0.72, 
            band_freq=(args.band_freq[0], args.band_freq[1]),
            parc=parc
        )
        for subj in range(nsubjs)
    )
    fc_emp_all, phase_delay_emp_all = zip(*results)
    fc_emp_all = np.dstack(fc_emp_all)
    phase_delay_emp_all = np.dstack(phase_delay_emp_all)

    # Initialise output arrays
    best_combs = []

    edge_fc_train = np.empty((args.n_splits, len(param_combs)))
    node_fc_train = np.empty((args.n_splits, len(param_combs)))
    phase_train = np.empty((args.n_splits, len(param_combs)))
    combined_train = np.zeros((args.n_splits, len(param_combs)))

    edge_fc_test = np.empty((args.n_splits))
    node_fc_test = np.empty((args.n_splits))
    phase_test = np.empty((args.n_splits))
    combined_test = np.zeros((args.n_splits))

    fc_matrices, phase_maps = [], []
    train_subjs_split, test_subjs_split = [], []
    
    kf = KFold(n_splits=args.n_splits, shuffle=False)
    for i, (train_index, test_index) in enumerate(kf.split(np.arange(args.n_subjs))):
        print(f"\n==========\nSplit {i+1}/{args.n_splits}\n==========")

        # Select train and test bold data
        fc_emp_train = np.mean(fc_emp_all[:, :, train_index], axis=2)
        fc_emp_test = np.mean(fc_emp_all[:, :, test_index], axis=2)
        phase_emp_train = phase_delay_emp_all[:, :, train_index].reshape(nverts, -1)
        phase_emp_test = phase_delay_emp_all[:, :, test_index].reshape(nverts, -1)

        # Get subject ids
        train_subjs_split.append(subj_ids[train_index])
        test_subjs_split.append(subj_ids[test_index])

        if args.hmap_label is not None:
            print(f"\nTraining...\n===========")
            
            results_train = Parallel(n_jobs=args.n_jobs, verbose=1)(
                delayed(run_and_evaluate)(
                    surf=surf,
                    medmask=medmask,
                    hmap=hmap,
                    parc=parc,
                    params=params,
                    args=args,
                    emp_results={"fc": fc_emp_train, "phase": phase_emp_train},
                    return_all=False,
                )
                for params in param_combs
            )
            edge_fc_train[i, :], node_fc_train[i, :], phase_train[i, :] = zip(*results_train)

            # Calculate combined metric
            if "edge_fc" in args.metrics:
                combined_train[i, :] += np.array(edge_fc_train[i, :])
            if "node_fc" in args.metrics:
                combined_train[i, :] += np.array(node_fc_train[i, :])
            if "phase" in args.metrics:
                combined_train[i, :] += np.array(phase_train[i, :])

            # Get best training results (average across runs)
            best_combs_ind = np.argmax(combined_train[i, :])
            best_alpha, best_r, best_gamma = param_combs[best_combs_ind]

            print("\nTraining results:\n-----------------")
            print(f"    Best alpha: {best_alpha:.3g}")
            print(f"    Best r: {best_r:.3g}")
            print(f"    Best gamma: {best_gamma:.3g}")
            print(f"    Best combined metric: {combined_train[i, best_combs_ind]:.4g}")
            print(f"    Best edge-level FC: {edge_fc_train[i, best_combs_ind]:.3g}")
            print(f"    Best node-level FC: {node_fc_train[i, best_combs_ind]:.3g}")
            print(f"    Best phase delay: {phase_train[i, best_combs_ind]:.3g}")
        else:
            best_alpha = 0
            best_r = 28.9
            best_gamma = 0.116
        best_combs.append((best_alpha, best_r, best_gamma))
        
        # Evaluate on test set
        print(f"\nTesting (alpha = {best_alpha:.1f}, r = {best_r:.1f}, gamma = {best_gamma:.3f})...\n=====================")
        # Run model and evaluate
        edge_fc_test[i], node_fc_test[i], phase_test[i], fc, phase_map = run_and_evaluate(
            surf=surf,
            medmask=medmask,
            hmap=hmap,
            parc=parc,
            params=(best_alpha, best_r, best_gamma),
            args=args,
            emp_results={"fc": fc_emp_test, "phase": phase_emp_test},
            return_all=True
        )
        fc_matrices.append(fc)
        phase_maps.append(phase_map)

        # Calculate combined metric
        if "edge_fc" in args.metrics:
            combined_test[i] += np.array(edge_fc_test[i])
        if "node_fc" in args.metrics:
            combined_test[i] += np.array(node_fc_test[i])
        if "phase" in args.metrics:
            combined_test[i] += np.array(phase_test[i])

        print(f"\nTest results:\n-------------")
        print(f"    Combined metric: {combined_test[i]:.4g}")
        print(f"    Edge-level FC: {edge_fc_test[i]:.3g}")
        print(f"    Node-level FC: {node_fc_test[i]:.3g}")
        print(f"    Phase Delay: {phase_test[i]:.3g}")

    print("\n==========\nFinal results\n==========")
    print(f"Best alpha: {np.mean(best_combs, axis=0)[0]}")
    print(f"Best r: {np.mean(best_combs, axis=0)[1]}")
    print(f"Best gamma: {np.mean(best_combs, axis=0)[2]}")
    print(f"Best combined metric: {np.mean(combined_test):.4g}") 
    print(f"Best edge-level FC: {np.mean(edge_fc_test):.3g}")
    print(f"Best node-level FC: {np.mean(node_fc_test):.3g}")
    print(f"Best Phase Delay: {np.mean(phase_test):.3g}")

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
        f.create_dataset('phase_delay_train', data=phase_train)
        f.create_dataset('combined_train', data=combined_train)

        f.create_dataset('edge_fc_test', data=edge_fc_test)
        f.create_dataset('node_fc_test', data=node_fc_test)
        f.create_dataset('phase_delay_test', data=phase_test)
        f.create_dataset('combined_test', data=combined_test)

        f.create_dataset('combs', data=param_combs)
        f.create_dataset('best_combs', data=best_combs)
        f.create_dataset('best_alpha', data=np.mean(best_combs, axis=0)[0])
        f.create_dataset('best_r', data=np.mean(best_combs, axis=0)[1])
        f.create_dataset('best_gamma', data=np.mean(best_combs, axis=0)[2])

        f.create_dataset('fc_matrices', data=np.dstack(fc_matrices))
        f.create_dataset('phase_maps', data=np.vstack(phase_maps).T)

        # Pad train_subjs_split and test_subjs_split with -1 to have the same length
        f.create_dataset('train_subjs_split', data=pad_sequences(train_subjs_split))
        f.create_dataset('test_subjs_split', data=pad_sequences(test_subjs_split))

if __name__ == "__main__":
    main()
    