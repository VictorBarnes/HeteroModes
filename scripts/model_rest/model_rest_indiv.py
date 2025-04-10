import os
import h5py
import time
import argparse
import itertools
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from brainspace.utils.parcellation import reduce_by_labels
from heteromodes import EigenSolver
from heteromodes.restingstate import simulate_bold, calc_gen_phase, calc_edge_and_node_fc, calc_phase_map
from heteromodes.solver import scale_hmap
from heteromodes.utils import load_project_env

load_project_env()
PROJ_DIR = os.getenv("PROJ_DIR")

def calc_fc_and_phase(bold, tr, band_freq, parc=None):
    # Z-score bold data
    scaler = StandardScaler()
    bold_z = scaler.fit_transform(bold.T).T

    # Compute phase on vertex data
    phase = calc_gen_phase(bold_z, tr=tr, lowcut=band_freq[0], highcut=band_freq[1])

    # Compute FC on parcellated data
    if parc is not None:
        bold_z = reduce_by_labels(bold_z, parc, axis=1)
    fc = np.corrcoef(bold_z)

    return fc, phase

def run_model(run, evals, emodes, parc, args, params, B=None):
    # Load external input, run model and parcellate
    ext_input = np.load(f"{PROJ_DIR}/data/resting_state/extInput_parc-{args.parc}_den-{args.den}_nT-1200_randseed-{run}.npy")
    bold_model = simulate_bold(evals, emodes, ext_input, solver_method='Fourier', 
                               eig_method='orthogonal', r=params[1], gamma=params[2], mass=B)

    # Z-score bold data
    scaler = StandardScaler()
    bold_model = scaler.fit_transform(bold_model.T).T
    
    # Compute model FC and phase
    fc_model, phase_model = calc_fc_and_phase(
        bold=bold_model, 
        tr=0.72, 
        band_freq=(args.band_freq[0], args.band_freq[1]),
        parc=parc
    )

    return fc_model, phase_model

def run_and_evaluate(surf, medmask, hmap, parc, params, args, emp_results, return_all=False):
    t1 = time.time()

    # Calculate modes
    solver = EigenSolver(
        surf=surf,
        hetero=hmap,
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

    # Calculate edge and node FC
    edge_fc_corr, node_fc_corr = calc_edge_and_node_fc(
        emp_results=emp_results, 
        model_results=model_results, 
    )

    # Calculate phase corr
    phase_map_emp = calc_phase_map(emp_results["phase"], n_components=4)
    phase_map_model = calc_phase_map(phase_model, n_components=4)
    if parc is not None:
        phase_map_emp = reduce_by_labels(phase_map_emp, parc, axis=0)
        phase_map_model = reduce_by_labels(phase_map_model, parc, axis=0)
    phase_corr = np.corrcoef(phase_map_emp, phase_map_model)[0, 1]

    t2 = time.time()
    print(f"alpha = {params[0]:.1f} | {(t2 - t1)/60:.1f} mins | Edge FC: {edge_fc_corr:.3f} | Node FC: {node_fc_corr:.3f} | Phase: {phase_corr:.3f}")

    if return_all:
        return edge_fc_corr, node_fc_corr, phase_corr, np.mean(fc_model, axis=2), phase_map_model
    else:
        return edge_fc_corr, node_fc_corr, phase_corr

def main():
    parser = argparse.ArgumentParser(description="Model resting-state fMRI BOLD data and evaluate against empirical data.")
    parser.add_argument("--id", type=int, help="The id of the run for saving outputs.")
    parser.add_argument("--hmap_label", type=str, default=None, help="The label of the heterogeneity map to use. If None, the null map is used.")
    parser.add_argument("--metrics", type=str, nargs='+', default=["edge_fc", "node_fc", "phase"], help="The metrics to use for evaluation. Defaults to ['edge_fc', 'node_fc', 'phase']")
    parser.add_argument("--n_runs", type=int, default=5, help="The number of runs to simulate. Defaults to 50.")
    parser.add_argument("--n_modes", type=int, default=500, help="The number of modes to calculate. Defaults to 500.")
    parser.add_argument("--n_jobs", type=int, default=-1, help="The number of CPUs for parallelization. Defaults to -1")
    parser.add_argument('--alpha', type=float, nargs=3, metavar=('alpha_min', 'alpha_max', 'alpha_step'), help='The alpha_min, alpha_max, and alpha_step values for scaling the heterogeneity map.')
    parser.add_argument("--den", type=str, default="32k", help="The density of the surface. Defaults to `32k`.")
    parser.add_argument("--parc", type=str, default=None, help="The parcellation to use to downsample the BOLD data.")
    parser.add_argument("--band_freq", type=float, nargs=2, default=[0.01, 0.1], metavar=('low', 'high'), help="The low and high bandpass frequencies for filtering the BOLD data. Defaults to [0.01, 0.1].")
    parser.add_argument("--subj_id", type=int, help="The subject ID to run the model on")
    args = parser.parse_args()

    out_dir = f'{PROJ_DIR}/results/model_rest/individual/id-{args.id}/subj-{args.subj_id}'

    # Get surface, medial mask and parcellation files
    hcp_dir = f"/fs03/kg98/vbarnes/HCP/{args.subj_id}/MNINonLinear/Results"
    surf_file = f"{args.subj_id}.L.midthickness_MSMAll.4k_fs_LR.surf.gii"
    surf = f"{hcp_dir}/{surf_file}"
    
    # Load parcellation and medial wall mask
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
        medmask = np.where(bold_emp_train[:, 0] != 0, True, False)

    # Load empirical BOLD data and z-score
    scaler = StandardScaler()
    bold_emp_train_path_lr = Path(
        hcp_dir, 
        "rfMRI_REST1_LR",
        "resampled",
        "rfMRI_REST1_LR_Atlas_hp2000_clean_4k.L.func.gii"
    )
    bold_emp_train_path_rl = Path(
        hcp_dir, 
        "rfMRI_REST1_RL",
        "resampled",
        "rfMRI_REST1_RL_Atlas_hp2000_clean_4k.L.func.gii"
    )
    bold_emp_train_lr = scaler.fit_transform(nib.load(bold_emp_train_path_lr).agg_data()[:, medmask]).T
    bold_emp_train_rl = scaler.fit_transform(nib.load(bold_emp_train_path_rl).agg_data()[:, medmask]).T
    bold_emp_train = np.concatenate((bold_emp_train_lr, bold_emp_train_rl), axis=1)

    bold_emp_test_path_lr = Path(
        hcp_dir, 
        "rfMRI_REST2_LR",
        "resampled",
        "rfMRI_REST2_LR_Atlas_hp2000_clean_4k.L.func.gii"
    )
    bold_emp_test_path_rl = Path(
        hcp_dir, 
        "rfMRI_REST2_RL",
        "resampled",
        "rfMRI_REST2_RL_Atlas_hp2000_clean_4k.L.func.gii"
    )
    bold_emp_test_lr = scaler.fit_transform(nib.load(bold_emp_test_path_lr).agg_data()[:, medmask]).T
    bold_emp_test_rl = scaler.fit_transform(nib.load(bold_emp_test_path_rl).agg_data()[:, medmask]).T
    bold_emp_test = np.concatenate((bold_emp_test_lr, bold_emp_test_rl), axis=1)

    # Get hmap and alpha values
    if args.hmap_label is None:
        hmap = None
        param_combs = [(0, 28.9, 0.116)]
    elif args.hmap_label == "myelinmap":
        hmap = nib.load(Path(hcp_dir, f"{args.subj_id}.L.SmoothedMyelinMap_BC.4k_fs_LR.func.gii")).darrays[0].data
        num_nonmed_zeros = np.sum(np.where(hmap[medmask] == 0, True, False))
        if num_nonmed_zeros > 0 and np.min(hmap[medmask]) == 0:
            print(f"Warning: {num_nonmed_zeros} vertices on the heterogeneity maps have a "
                f"value of 0.")

        alpha_min, alpha_max, alpha_step = args.alpha
        alpha_num = int(abs(alpha_max - alpha_min) / alpha_step) + 1
        alpha_vals = np.linspace(alpha_min, alpha_max, alpha_num)
        alpha_vals = alpha_vals[alpha_vals != 0] # Remove 0 since that is the homogeneous case
        # Only keep valid alpha values (i.e. max wave speed <= 150 m/s)
        valid_alpha = []
        for i, alpha in enumerate(alpha_vals):
            if np.max(3.3524*np.sqrt(scale_hmap(hmap[medmask], alpha=alpha))) <= 150:
                valid_alpha.append(alpha)

        r_vals = [28.9]
        gamma_vals = [0.116]

        param_combs = list(itertools.product(valid_alpha, r_vals, gamma_vals))
    else:
        raise ValueError(f"Invalid hmap label: {args.hmap_label}")

    print(f"Number of parameter combinations: {len(param_combs)}")

    # Parallelize the calculation of FC and phase
    print("Calculating empirical FC and Phase...")
    fc_emp_train, phase_emp_train = calc_fc_and_phase(
        bold=bold_emp_train, 
        tr=0.72, 
        band_freq=(args.band_freq[0], args.band_freq[1]), 
        parc=parc
    )
    fc_emp_test, phase_emp_test = calc_fc_and_phase(
        bold=bold_emp_test, 
        tr=0.72, 
        band_freq=(args.band_freq[0], args.band_freq[1]), 
        parc=parc
    )

    # Train model
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
        edge_fc_train, node_fc_train, phase_train = zip(*results_train)

        combined_train = np.zeros(len(param_combs))
        # Calculate combined metric
        if "edge_fc" in args.metrics:
            combined_train += np.array(edge_fc_train)
        if "node_fc" in args.metrics:
            combined_train += np.array(node_fc_train)
        if "phase" in args.metrics:
            combined_train += np.array(phase_train)

        # Get best training results (average across runs)
        best_combs_ind = np.argmax(combined_train)
        best_alpha, best_r, best_gamma = param_combs[best_combs_ind]

        print("\nTraining results:\n-----------------")
        print(f"    Best alpha: {best_alpha:.3g}")
        print(f"    Best r: {best_r:.3g}")
        print(f"    Best gamma: {best_gamma:.3g}")
        print(f"    Best combined metric: {combined_train[best_combs_ind]:.4g}")
        print(f"    Best edge-level FC: {edge_fc_train[best_combs_ind]:.3g}")
        print(f"    Best node-level FC: {node_fc_train[best_combs_ind]:.3g}")
        print(f"    Best phase corr: {phase_train[best_combs_ind]:.3g}")
    else:
        best_alpha = 0
        best_r = 28.9
        best_gamma = 0.116
    best_combs = (best_alpha, best_r, best_gamma)
    
    # Evaluate on test set
    print(f"\nTesting (alpha = {best_alpha:.1f}, r = {best_r:.1f}, gamma = {best_gamma:.3f})..."
        f"\n==================================================")
    # Run model and evaluate
    edge_fc_test, node_fc_test, phase_test, fc, phase_map = run_and_evaluate(
        surf=surf,
        medmask=medmask,
        hmap=hmap,
        parc=parc,
        params=(best_alpha, best_r, best_gamma),
        args=args,
        emp_results={"fc": fc_emp_test, "phase": phase_emp_test},
        return_all=True
    )

    combined_test = 0
    # Calculate combined metric
    if "edge_fc" in args.metrics:
        combined_test += edge_fc_test
    if "node_fc" in args.metrics:
        combined_test += node_fc_test
    if "phase" in args.metrics:
        combined_test += phase_test

    print(f"\nTest results:\n-------------")
    print(f"    Combined metric: {combined_test:.4g}")
    print(f"    Edge-level FC: {edge_fc_test:.3g}")
    print(f"    Node-level FC: {node_fc_test:.3g}")
    print(f"    Phase corr: {phase_test:.3g}")

    print("\n==========\nFinal results\n==========")
    print(f"Best alpha: {best_combs[0]}")
    print(f"Best r: {best_combs[1]:.3g}")
    print(f"Best gamma: {best_combs[2]:.3g}")
    print(f"Best combined metric: {combined_test:.4g}") 
    print(f"Best edge-level FC: {edge_fc_test:.3g}")
    print(f"Best node-level FC: {node_fc_test:.3g}")
    print(f"Best phase corr: {phase_test:.3g}")

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
        if args.hmap_label is not None:
            f.create_dataset('edge_fc_train', data=edge_fc_train)
            f.create_dataset('node_fc_train', data=node_fc_train)
            f.create_dataset('phase_train', data=phase_train)
            f.create_dataset('combined_train', data=combined_train)
            
            f.create_dataset('combs', data=param_combs)

        f.create_dataset('best_combs', data=best_combs)

        f.create_dataset('edge_fc_test', data=edge_fc_test)
        f.create_dataset('node_fc_test', data=node_fc_test)
        f.create_dataset('phase_test', data=phase_test)
        f.create_dataset('combined_test', data=combined_test)

        f.create_dataset('fc', data=np.dstack(fc))
        f.create_dataset('phase_map', data=np.vstack(phase_map).T)

        f.create_dataset('medmask', data=medmask)

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print(f"Total time: {(t2 - t1)/60:.1f} mins")
    