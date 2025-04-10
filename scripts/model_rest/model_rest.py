import os
import h5py
import time
import argparse
import itertools
import numpy as np
from joblib import Parallel, delayed, Memory
from neuromaps.datasets import fetch_atlas
from heteromodes.utils import load_hmap, load_project_env
from heteromodes.restingstate import run_simulation, evaluate_model

load_project_env()
PROJ_DIR = os.getenv("PROJ_DIR")
memory = Memory("/fs03/kg98/vbarnes/cache/", verbose=0)

def print_heading(text):
    print(f"\n{text}\n{'=' * len(text)}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Model resting-state fMRI BOLD data and evaluate against empirical data.")
    parser.add_argument("--id", type=str, default=0, help="The id of the run for saving outputs.")
    parser.add_argument("--hmap_label", type=lambda x: None if x.lower() == "none" else x, default=None, 
                        help="The label of the heterogeneity map. Defaults to None (indicating homogeneity)")
    parser.add_argument("--n_runs", type=int, default=5, 
                        help="The number of runs to simulate. Defaults to 50.")
    parser.add_argument("--n_modes", type=int, default=500, 
                        help="The number of modes to calculate. Defaults to 500.")
    parser.add_argument("--n_splits", type=int, default=5, 
                        help="The number of splits for cross-validation. Defaults to 5.")
    parser.add_argument("--n_subjs", type=int, default=255, 
                        help="The number of subjects in the empirical data. Defaults to 255.")
    parser.add_argument("--n_jobs", type=int, default=-1, 
                        help="The number of CPUs for parallelization. Defaults to -1")
    parser.add_argument('--alpha', type=float, nargs=3, default=[-2, 2, 0.2], metavar=('alpha_min', 'alpha_max', 'alpha_step'), 
                        help='The alpha_min, alpha_max, and alpha_step values for scaling the heterogeneity map.')
    parser.add_argument("--r", type=float, nargs="+", default=[28.9], metavar="r_values", 
                        help="The spatial length scale of the wave model. Defaults to 28.9 s^-1. Provide either a single r value or three values (r_min, r_max, r_step).")
    parser.add_argument("--gamma", type=float, nargs="+", default=[0.116], metavar="gamma values",
                        help="The dampening rate of the wave model. Defaults to 0.116 s^-1. Provide either a single gamma value or three values (gamma_min, gamma_max, gamma_step).")
    parser.add_argument("--den", type=str, default="4k", 
                        help="The density of the surface. Defaults to `4k`.")
    parser.add_argument("--band_freq", type=float, nargs=2, default=[0.01, 0.1], metavar=('low', 'high'), 
                        help="The low and high bandpass frequencies for filtering the BOLD data. Defaults to [0.01, 0.1].")
    parser.add_argument("--metrics", type=str, nargs='+', default=["edge_fc", "node_fc", "phase"], 
                        help="The metrics to use for evaluation. Defaults to ['edge_fc', 'node_fc', 'phase']")
    parser.add_argument("--crossval", type=lambda x: x.lower() == "true", default=False, 
                        help="Whether to perform cross-validation. Defaults to False.")
    parser.add_argument("--scaling", type=str, default="exponential",
                        help="The scaling to apply to the heterogeneity map. Defaults to 'exponential'.")
    parser.add_argument('--q_norm', type=lambda x: None if x.lower() == "none" else x, default=None, 
                        help="Type of distribution to match to when doing the quantile normalisation")
    parser.add_argument("--phase_type", type=str, default="cpc1",
                        help="The type of phase to calculate: 'cpc1' or 'combined'. Defaults to 'cpc1'")
    parser.add_argument("--n_comps", type=int, default=3,
                        help="The number of components to calculate for the phase map if phase_type == 'combined'. Defaults to 3.")
    
    return parser.parse_args()


# @profile
def run_split(model_fcs, model_phase_maps, emp_train_fc, emp_test_fc, emp_train_phase_map, emp_test_phase_map, param_combs, args):
    # Evaluate each model on train data
    train_edge_fc_corr = np.empty(len(model_fcs))
    train_node_fc_corr = np.empty(len(model_fcs))
    train_phase_corr = np.empty(len(model_fcs))
    for param_i in range(len(model_fcs)):
        if "phase" in args.metrics:
            model_phase_map_param = model_phase_maps[param_i]
        else:
            model_phase_map_param = None
        train_edge_fc_corr[param_i], train_node_fc_corr[param_i], train_phase_corr[param_i] = evaluate_model(
            model_fcs[param_i],
            model_phase_map_param,
            emp_train_fc,
            emp_train_phase_map,
            args.metrics
        )

        print(f"alpha: {param_combs[param_i][0]:.2f}, r: {param_combs[param_i][1]:.1f}, gamma: {param_combs[param_i][2]:.3f} | Combined metric: {train_edge_fc_corr[param_i] + train_node_fc_corr[param_i] + train_phase_corr[param_i]:.3g}")

    # Calculate combined metric
    train_combined = train_edge_fc_corr + train_node_fc_corr + train_phase_corr
    best_ind = np.argmax(train_combined)
    best_combs = param_combs[best_ind]
    
    # Get best FC and phase map from best model
    best_fc = model_fcs[best_ind]
    if "phase" in args.metrics:
        best_phase_map = model_phase_maps[best_ind]
    else:
        best_phase_map = None
        emp_test_phase_map = None
    # Evaluate best model on test data
    test_edge_fc_corr, test_node_fc_corr, test_phase_corr = evaluate_model(
        best_fc,
        best_phase_map,
        emp_test_fc,
        emp_test_phase_map,
        args.metrics
    )

    # Calculate combined metric
    test_combined = test_edge_fc_corr + test_node_fc_corr + test_phase_corr

    return (
        train_edge_fc_corr, train_node_fc_corr, train_phase_corr, train_combined,
        test_edge_fc_corr, test_node_fc_corr, test_phase_corr, test_combined,
        best_fc, best_phase_map, best_combs
    )


if __name__ == "__main__":
    t1 = time.time()

    # Define constants
    dt_emp = 720        # empirical time step (ms)
    nt_emp = 1200       # empirical number of time points
    dt = dt_emp / 8     # model time step (ms)
    tsteady = 50 * 1e3  # burn time to remove transient effects (ms)

    args = parse_arguments()
    if args.hmap_label == "None":
        args.hmap_label = None
    if args.q_norm == "None":
        args.q_norm = None

    out_dir = f'{PROJ_DIR}/results/model_rest/group/id-{args.id}'

    # Get surface
    fslr = fetch_atlas(atlas='fsLR', density=args.den)
    surf = str(fslr['midthickness'][0])
    # Load medmask
    with h5py.File(f"{PROJ_DIR}/data/empirical/HCP_nsubj-{args.n_subjs}_bold_parc-None_fsLR{args.den}_hemi-L.h5", 'r') as f:
        medmask = f['medmask'][:]

    # Get hmap and alpha values
    if args.hmap_label is None:
        hmap = None
        alpha_vals = [0]
    else:
        # If hmap_label is null, load null map
        if args.hmap_label[:4] == "null":
            null_id = int(args.hmap_label.split('-')[1])
            hmap = np.load(f"{PROJ_DIR}/data/nulls/data-myelinmap_space-fsLR_den-{args.den}_hemi-L_nmodes-500_nnulls-5000_nulls_resample-True.npy")
            hmap = hmap[null_id, :]

            out_dir = out_dir + "/nulls"
        # Otherwise assume hmap_label is a valid label
        else:
            hmap = load_hmap(args.hmap_label, trg_den=args.den)
            num_nonmed_zeros = np.sum(np.where(hmap[medmask] == 0, True, False))
            if num_nonmed_zeros > 0 and np.min(hmap[medmask]) == 0:
                print(f"Warning: {num_nonmed_zeros} vertices on the heterogeneity maps have a "
                        f"value of 0.")

        alpha_min, alpha_max, alpha_step = args.alpha
        alpha_num = int(abs(alpha_max - alpha_min) / alpha_step) + 1
        alpha_vals = np.linspace(alpha_min, alpha_max, alpha_num)
        if alpha_min < 0 < alpha_max:
            alpha_vals = alpha_vals[alpha_vals != 0] # Remove 0 since that is the homogeneous case
        print(alpha_vals)
    r_vals = args.r if len(args.r) == 1 else np.arange(args.r[0], args.r[1], args.r[2])
    gamma_vals = args.gamma if len(args.gamma) == 1 else np.arange(args.gamma[0], args.gamma[1], args.gamma[2])
    param_combs = list(itertools.product(alpha_vals, r_vals, gamma_vals))

    # Run all models (i.e. for all parameter combinations) and store outputs in cache
    print_heading(f"Running {args.hmap_label} model for {len(param_combs)} parameter combinations...")
    results = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(run_simulation)(
            surf=surf, 
            medmask=medmask, 
            hetero=hmap, 
            alpha=params[0],
            r=params[1], 
            gamma=params[2],
            scaling=args.scaling,
            q_norm=args.q_norm,
            nmodes=args.n_modes,
            nruns=args.n_runs,
            dt_emp=dt_emp,
            nt_emp=nt_emp,
            dt=dt,
            tsteady=tsteady,
            phase_type=args.phase_type if "phase" in args.metrics else None,
            band_freq=args.band_freq,
            n_comps=args.n_comps
        ) for params in param_combs
    )
    fcs_model, phase_maps_model = zip(*results)
    # Find invalid combinations and remove those 
    nan_indices = [i for i, x in enumerate(fcs_model) if x is None]
    print(f"Only {len(param_combs) - len(nan_indices)} out of {len(param_combs)} parameter combinations were valid.")
    
    param_combs = [param_combs[i] for i in range(len(param_combs)) if i not in nan_indices]
    fcs_model = [fcs_model[i] for i in range(len(fcs_model)) if i not in nan_indices]
    phase_maps_model = [phase_maps_model[i] for i in range(len(phase_maps_model)) if i not in nan_indices]
    if "phase" not in args.metrics:
        phase_maps_model = None

    # Evaluated against empirical data
    if not args.crossval:
        print_heading("Fitting models to empirical data...")
        with h5py.File(f"{PROJ_DIR}/data/empirical/HCP_nsubj-255_fc_parc-None_fsLR4k_hemi-L.h5", "r") as f:
            emp_fc = f['fc_group'][:]

        with h5py.File(f"{PROJ_DIR}/data/empirical/HCP_nsubj-255_complexPhase_parc-None_fsLR4k_hemi-L_freql-0.01_freqh-0.1.h5", "r") as f:
            phase_cpcs_group = f['phase_cpcs_group'][:]
            if args.phase_type == "cpc1":
                emp_phase_map = phase_cpcs_group[:, 0]
            elif args.phase_type == "combined":
                svals_emp = f['svals_group'][:args.n_comps]
                emp_phase_map = np.sum(phase_cpcs_group[:, :args.n_comps] * svals_emp, axis=1) / np.sum(svals_emp)

        edge_fc_corr = np.empty(len(param_combs))
        node_fc_corr = np.empty(len(param_combs))
        phase_corr = np.empty(len(param_combs))
        for param_i in range(len(param_combs)):
            if "phase" in args.metrics:
                model_phase_map_param = phase_maps_model[param_i]
            else:
                model_phase_map_param = None
            edge_fc_corr[param_i], node_fc_corr[param_i], phase_corr[param_i] = evaluate_model(
                fcs_model[param_i],
                model_phase_map_param,
                emp_fc,
                emp_phase_map,
                args.metrics
            )

        # Calculate combined metric
        combined = edge_fc_corr + node_fc_corr + phase_corr
        best_ind = np.argmax(combined)
        best_combs = param_combs[best_ind]
        model_fc_best = fcs_model[best_ind]
        if "phase" in args.metrics:
            model_phase_map_best = phase_maps_model[best_ind]

        # Print results
        print_heading("Model fit results")
        print(f"alpha: {best_combs[0]:.2f}")
        if len(r_vals) > 1:
            print(f"r: {best_combs[1]:.2f}")
        if len(gamma_vals) > 1:
            print(f"gamma: {best_combs[2]:.2f}")
        if "edge_fc" in args.metrics:
            print(f"Edge-level FC corr: {edge_fc_corr[best_ind]:.3g}")
        if "node_fc" in args.metrics:
            print(f"Node-level FC corr: {node_fc_corr[best_ind]:.3g}")
        if "phase" in args.metrics:
            print(f"Phase corr: {phase_corr[best_ind]:.3g}")
        print(f"Combined metric: {combined[best_ind]:.3g}")

    else:
        print_heading("Fitting models to empirical data using cross validation...")
        with h5py.File(f"{PROJ_DIR}/data/empirical/HCP_nsubj-255_fc-kfold5_parc-None_fsLR{args.den}_hemi-L.h5", "r") as f:
            fcs_train = f['fc_train'][:]
            fcs_test = f['fc_test'][:]

        with h5py.File(f"{PROJ_DIR}/data/empirical/HCP_nsubj-255_phasecpcs-kfold5_parc-None_fsLR{args.den}_hemi-L_freql-{args.band_freq[0]}_freqh-{args.band_freq[1]}.h5", "r") as f:
            phase_cpcs_train = f['phase_cpcs_train'][:]
            phase_cpcs_test = f['phase_cpcs_test'][:]
            if args.phase_type == "cpc1":
                phase_maps_train = phase_cpcs_train[:, 0, :]
                phase_maps_test = phase_cpcs_test[:, 0, :]
            elif args.phase_type == "combined":
                svals_train = f['svals_train'][:args.n_comps]
                svals_test = f['svals_test'][:args.n_comps]
                phase_maps_train = np.empty((np.sum(medmask), args.n_splits))
                phase_maps_test = np.empty((np.sum(medmask), args.n_splits))
                for i in range(args.n_splits):
                    phase_maps_train[:, i] = np.sum(phase_cpcs_train[:, :args.n_comps, i] * svals_train[:args.n_comps, i], axis=1) / np.sum(svals_train[:args.n_comps, i])
                    phase_maps_test[:, i] = np.sum(phase_cpcs_test[:, :args.n_comps, i] * svals_test[:args.n_comps, i], axis=1) / np.sum(svals_test[:args.n_comps, i])

        split_results = Parallel(n_jobs=args.n_jobs, backend="loky")(
            delayed(run_split)(
            fcs_model,
            phase_maps_model,
            fcs_train[:, :, i], 
            fcs_test[:, :, i], 
            phase_maps_train[:, i], 
            phase_maps_test[:, i],
            param_combs, 
            args
            )
            for i in range(args.n_splits)
        )
        train_edge_fc_corr = np.vstack([split[0] for split in split_results])
        train_node_fc_corr = np.vstack([split[1] for split in split_results])
        train_phase_corr = np.vstack([split[2] for split in split_results])
        train_combined = np.vstack([split[3] for split in split_results])

        test_edge_fc_corr = np.array([split[4] for split in split_results])
        test_node_fc_corr = np.array([split[5] for split in split_results])
        test_phase_corr = np.array([split[6] for split in split_results])
        test_combined = np.array([split[7] for split in split_results])

        best_fc_matrices = np.dstack([split[8] for split in split_results])
        best_phase_maps = np.vstack([split[9] for split in split_results]).T if "phase" in args.metrics else None
        best_combs = np.vstack([split[10] for split in split_results])

        # Print test results
        for i in range(args.n_splits):
            print(f"\nSplit {i+1}: Test results")
            print(f"alpha: {best_combs[i, 0]:.2f}")
            if len(r_vals) > 1:
                print(f"r: {best_combs[i, 1]:.2f}")
            if len(gamma_vals) > 1:
                print(f"gamma: {best_combs[i, 2]:.2f}")
            if "edge_fc" in args.metrics:
                print(f"Edge-level FC corr: {np.mean(test_edge_fc_corr[i]):.3g}")
            if "node_fc" in args.metrics:
                print(f"Node-level FC corr: {np.mean(test_node_fc_corr[i]):.3g}")
            if "phase" in args.metrics:
                print(f"Phase corr: {np.mean(test_phase_corr[i]):.3g}")
            print(f"Combined metric: {np.mean(test_combined[i]):.3}") 

        # Print final results
        print_heading("Final Results")
        print(f"alpha: {np.mean(best_combs, axis=0)[0]:.2f}")
        if len(r_vals) > 1:
            print(f"r: {np.mean(best_combs, axis=0)[1]:.2f}")
        if len(gamma_vals) > 1:
            print(f"gamma: {np.mean(best_combs, axis=0)[2]:.2f}")
        if "edge_fc" in args.metrics:
            print(f"Edge-level FC corr: {np.mean(test_edge_fc_corr):.3g}")
        if "node_fc" in args.metrics:
            print(f"Node-level FC corr: {np.mean(test_node_fc_corr):.3g}")
        if "phase" in args.metrics:
            print(f"Phase corr: {np.mean(test_phase_corr):.3g}")
        print(f"Combined metric: {np.mean(test_combined):.3g}") 

    # Save results
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_path = f"{out_dir}/{str(args.hmap_label)}_results_crossval-{args.crossval}.h5"
    print(f"Output path: {out_path}")

    with h5py.File(out_path, 'w') as f:
        # Write metadata to file
        for key, value in vars(args).items():
            if value is None:
                f.attrs[key] = "None"
            else:
                f.attrs[key] = value
        # Write outputs to file
        if not args.crossval:
            f.create_dataset('edge_fc_corr', data=edge_fc_corr)
            f.create_dataset('node_fc_corr', data=node_fc_corr)
            f.create_dataset('phase_corr', data=phase_corr)
            f.create_dataset('combined', data=combined)
            f.create_dataset('best_fc', data=model_fc_best)
            if "phase" in args.metrics:
                f.create_dataset('best_phase_map', data=model_phase_map_best)

        else:
            if args.hmap_label is not None:
                f.create_dataset('edge_fc_train', data=train_edge_fc_corr)
                f.create_dataset('node_fc_train', data=train_node_fc_corr)
                f.create_dataset('phase_train', data=train_phase_corr)
                f.create_dataset('combined_train', data=train_combined)

            f.create_dataset('edge_fc_test', data=test_edge_fc_corr)
            f.create_dataset('node_fc_test', data=test_node_fc_corr)
            f.create_dataset('phase_test', data=test_phase_corr)
            f.create_dataset('combined_test', data=test_combined)

            f.create_dataset('best_fcs', data=best_fc_matrices)
            if "phase" in args.metrics:
                f.create_dataset('best_phase_maps', data=best_phase_maps)

        f.create_dataset('best_combs', data=best_combs)
        f.create_dataset('combs', data=np.array(param_combs))
        f.create_dataset('medmask', data=medmask)

    t2 = time.time()
    print(f"Total time: {(t2 - t1)/60:.1f} mins")
