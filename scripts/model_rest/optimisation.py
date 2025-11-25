"""
Optimize neural field model parameters against empirical resting-state fMRI data.

This script performs model parameter optimization using either simple fitting or
k-fold cross-validation to identify optimal heterogeneity scaling and model parameters.
"""

import os
import h5py
import time
import argparse
import itertools
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from joblib import Parallel, delayed
from neuromaps.datasets import fetch_atlas
from heteromodes.utils import load_hmap, get_project_root
from heteromodes.restingstate import run_model, analyze_bold, evaluate_model


PROJ_DIR = get_project_root()


def print_heading(text):
    """Print a formatted section heading."""
    print(f"\n{text}\n{'=' * len(text)}")


def parse_arguments():
    """
    Parse command-line arguments for model optimization.
    
    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Model resting-state fMRI BOLD data and evaluate against empirical data."
    )
    parser.add_argument(
        "--species", type=str, default="human",
        help="Species: 'human', 'macaque', or 'marmoset' (default: human)."
    )
    parser.add_argument(
        "--id", type=str, default=0, 
        help="Run ID for saving outputs."
    )
    parser.add_argument(
        "--hmap_label", 
        type=lambda x: None if x.lower() == "none" else x, 
        default=None, 
        help="Heterogeneity map label (None for homogeneous model)."
    )
    parser.add_argument(
        "--n_runs", type=int, default=10, 
        help="Number of simulation runs (default: 10)."
    )
    parser.add_argument(
        "--n_modes", type=int, default=500, 
        help="Number of eigenmodes (default: 500)."
    )
    parser.add_argument(
        "--n_splits", type=int, default=5, 
        help="Number of cross-validation splits (default: 5)."
    )
    parser.add_argument(
        "--n_subjs", type=int, default=255, 
        help="Number of subjects in empirical data (default: 255)."
    )
    parser.add_argument(
        "--n_jobs", type=int, default=-1, 
        help="Number of parallel jobs, -1 for all CPUs (default: -1)."
    )
    parser.add_argument(
        '--alpha', type=float, nargs=3, default=[-5, 5, 0.1], 
        metavar=('alpha_min', 'alpha_max', 'alpha_step'), 
        help='Alpha range: min, max, step for heterogeneity scaling.'
    )
    parser.add_argument(
        "--beta", type=float, default=[1.0], nargs="+", 
        metavar="beta_values",
        help="Beta value(s) for scaling (single value or min/max/step)."
    )
    parser.add_argument(
        "--r", type=float, nargs="+", default=[28.9], 
        metavar="r_values", 
        help="Spatial length scale in mm (single value or min/max/step, default: 28.9)."
    )
    parser.add_argument(
        "--gamma", type=float, nargs="+", default=[0.116], 
        metavar="gamma_values",
        help="Damping rate in s^-1 (single value or min/max/step, default: 0.116)."
    )
    parser.add_argument(
        "--den", type=str, default="4k", 
        help="Surface density (default: 4k)."
    )
    parser.add_argument(
        "--band_freq", type=float, nargs=2, default=[0.04, 0.07], 
        metavar=('low', 'high'), 
        help="Bandpass filter frequencies in Hz (default: [0.04, 0.07])."
    )
    parser.add_argument(
        "--metrics", type=str, nargs='+', 
        default=["edge_fc_corr", "node_fc_corr"], 
        help="Evaluation metrics (default: edge_fc_corr, node_fc_corr, fcd_ks). Note that "
             "optimizing using fcd_kw is very computationally intensive. I recommend "
             "using a high-performance computing cluster if including this metric."
    )
    parser.add_argument(
        "--evaluation", type=str, choices=["fit", "crossval"], 
        default="crossval",
        help="Evaluation type: 'fit' (all data) or 'crossval' (k-fold, default)."
    )
    parser.add_argument(
        "--scaling", type=str, default="sigmoid",
        help="Heterogeneity map scaling function (default: sigmoid)."
    )
    parser.add_argument(
        "--nt_emp", type=int, default=600,
        help="Number of empirical timepoints. Human: 600, macaque: 500, marmoset: 510 (default: 600)."
    )
    parser.add_argument(
        "--parc", type=str, default=None,
        help="Parcellation name (None for vertex-wise, default: None)."
    )
    return parser.parse_args()


def model_job(params, surf, medmask, hmap, args, dt_emp, dt, tsteady, parc=None, out_dir=None):
    """
    Run single model simulation with given parameter combination.
    
    Checks if results already exist before running simulation. If found, loads
    cached results; otherwise runs new simulation and analyzes BOLD data.
    
    Parameters
    ----------
    params : tuple
        Parameter combination (alpha, r, gamma, beta).
    surf : str or Surface
        Path to cortical surface or Surface object.
    medmask : np.ndarray
        Binary mask for medial wall.
    hmap : np.ndarray or None
        Heterogeneity map data.
    args : argparse.Namespace
        Command-line arguments.
    dt_emp : float
        Empirical sampling interval in milliseconds.
    dt : float
        Model simulation time step in milliseconds.
    tsteady : int
        Number of initial timepoints to discard as steady-state burn-in.
    parc : np.ndarray, optional
        Parcellation labels.
    out_dir : str, optional
        Output directory for saving results.
    
    Returns
    -------
    outputs : dict or None
        Dictionary containing 'fc', 'fcd', etc., or None if simulation failed.
    """
    # Check for cached results
    out_file = (
        f"{out_dir}/model_alpha-{params[0]:.1f}_r-{params[1]:.1f}_"
        f"gamma-{params[2]:.3f}_beta-{params[3]:.1f}.h5"
    )
    if os.path.exists(out_file):
        print(f"Loading cached model outputs from {out_file}")
        outputs = {}
        with h5py.File(out_file, "r") as f:
            if "edge_fc_corr" in args.metrics or "node_fc_corr" in args.metrics:
                outputs['fc'] = f['fc'][:]
            if "fcd_ks" in args.metrics:
                outputs['fcd'] = f['fcd'][:]
        return outputs
    
    # Run new simulation
    try:
        bold_data = run_model(
            surf=surf, 
            medmask=medmask, 
            hetero=hmap, 
            alpha=params[0],
            r=params[1],
            gamma=params[2],
            beta=params[3],
            scaling=args.scaling,
            n_modes=args.n_modes,
            n_runs=args.n_runs,
            dt_emp=dt_emp,
            nt_emp=args.nt_emp,
            dt_model=dt,
            tsteady=tsteady,
            parc=parc,
        )
        
        if bold_data is None:
            print(f"Simulation failed for params {params}.")
            return None
        
        # Analyze BOLD data
        outputs = analyze_bold(
            bold_data, 
            dt_emp=dt_emp, 
            band_freq=args.band_freq,
            metrics=args.metrics
        )
        
        return outputs
        
    except Exception as e:
        print(f"Error in model_job with params {params}: {e}")
        return None


def run_split(model_outputs, emp_outputs_train, emp_outputs_test, args):
    """
    Run one cross-validation split: train on training set, test on held-out set.
    
    Parameters
    ----------
    model_outputs : list of dict
        List of model outputs for all parameter combinations.
    emp_outputs_train : dict
        Empirical training data metrics.
    emp_outputs_test : dict
        Empirical test data metrics.
    args : argparse.Namespace
        Command-line arguments.
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'train_results': List of training metrics for all models
        - 'test_results': Test metrics for best model
        - 'best_ind': Index of best model
    """
    # Evaluate each model on training data
    train_results = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(evaluate_model)(
            model_outputs[i],
            emp_outputs_train,
            args.metrics
        ) for i in range(len(model_outputs))
    )

    # Select best model based on combined metric
    combined = [
        results.get('edge_fc_corr', 0) + 
        results.get('node_fc_corr', 0) + 
        (1 - results.get('fcd_ks', 0) if 'fcd_ks' in args.metrics else 0)
        for results in train_results
    ]
    best_ind = np.argmax(combined)

    # Evaluate best model on test set
    test_results = evaluate_model(
        model_outputs[best_ind],
        emp_outputs_test,
        args.metrics
    )
    
    return {
        'train_results': train_results,
        'test_results': test_results,
        'best_ind': best_ind
    }

if __name__ == "__main__":
    t1 = time.time()

    # Parse and normalize arguments
    args = parse_arguments()
    if args.hmap_label == "None":
        args.hmap_label = None
    if args.parc == "None":
        args.parc = None

    # Species-specific parameters
    TR_SPECIES = {"human": 720, "macaque": 2600, "marmoset": 2000}  # ms
    DT_SPECIES = {"human": 90, "macaque": 100, "marmoset": 100}  # ms
    DATA_DESC_SPECIES = {
        "human": f"hcp-s1200_nsubj-{args.n_subjs}",
        "macaque": f"macaque-awake_nsubj-{args.n_subjs}",
        "marmoset": f"mbm-v4_nsubj-{args.n_subjs}"
    }

    dt_emp = TR_SPECIES[args.species]
    dt = DT_SPECIES[args.species]
    data_desc = DATA_DESC_SPECIES[args.species]
    tsteady = 550  # Number of timepoints to discard as burn-in (~45 seconds at 90ms dt)

    out_dir = (
        f'{PROJ_DIR}/results/{args.species}/model_rest/group/id-{args.id}/'
        f'{args.evaluation}/{args.hmap_label}'
    )

    # Setup parcellation and surface
    parc = None
    if args.parc is not None:
        if args.species != "human":
            raise ValueError("Parcellation is only valid for human species.")
        
        parc = nib.load(
            f"{PROJ_DIR}/data/parcellations/parc-{args.parc}_space-fsLR_"
            f"den-{args.den}_hemi-L.label.gii"
        ).darrays[0].data.astype(int)

        # Parcellation uses 32k downsampled data
        space_desc = f"space-fsLR_den-32k_parc-{args.parc}"
        medmask = parc != 0
    else:
        space_desc = f"space-fsLR_den-{args.den}"

    # Load cortical surface
    if args.species == "human":
        fslr = fetch_atlas(atlas='fsLR', density=args.den)
        surf = str(fslr['midthickness'][0])
    else:
        surf = (
            f'{PROJ_DIR}/data/empirical/{args.species}/space-fsLR_den-{args.den}_'
            f'hemi-L_desc-midthickness.surf.gii'
        )
    
    # Load medial wall mask if not using parcellation
    if args.parc is None:
        medmask = nib.load(
            f"{PROJ_DIR}/data/empirical/{args.species}/space-fsLR_den-{args.den}_"
            f"hemi-L_desc-nomedialwall.func.gii"
        ).darrays[0].data.astype(bool)

    # Load heterogeneity map and setup alpha values
    is_null = False
    if args.hmap_label is None:
        # Homogeneous model
        hmap = None
        alpha_vals = [0]
    else:
        # Check if this is a null map
        if args.hmap_label.startswith("null"):
            is_null = True
            split = args.hmap_label.split('-')  # Format: null-{hmap_label}-{null_id}
            args.hmap_label = split[1]
            null_id = int(split[2])
            
            hmap = np.load(
                f"{PROJ_DIR}/data/nulls/{args.species}/data-{args.hmap_label}_{space_desc}_hemi-L_"
                f"nmodes-500_nnulls-1000_nulls_resample-True.npy"
            )[null_id, :]

            out_dir = (
                f'{PROJ_DIR}/results/{args.species}/model_rest/group/id-{args.id}/'
                f'{args.evaluation}/{args.hmap_label}/nulls/null-{null_id}'
            )
        else:
            # Load actual heterogeneity map
            hmap = load_hmap(args.hmap_label, species=args.species)

        # Clip extreme values for numerical stability
        p_lower, p_upper = np.percentile(hmap[medmask], [2, 98])
        hmap = np.clip(hmap, p_lower, p_upper)
        
        # Warn about zero values
        num_zeros = np.sum(hmap[medmask] == 0)
        if num_zeros > 0 and np.min(hmap[medmask]) == 0:
            print(f"Warning: {num_zeros} cortical vertices have heterogeneity value of 0.")

        # Setup alpha parameter range
        alpha_min, alpha_max, alpha_step = args.alpha
        alpha_num = int(abs(alpha_max - alpha_min) / alpha_step) + 1
        alpha_vals = np.linspace(alpha_min, alpha_max, alpha_num).round(
            len(str(alpha_step).split('.')[-1])
        )
        # Remove alpha=0 (handled by homogeneous model)
        if alpha_min < 0 < alpha_max:
            alpha_vals = alpha_vals[alpha_vals != 0]


    # Setup parameter combinations
    r_vals = args.r if len(args.r) == 1 else np.arange(args.r[0], args.r[1] + args.r[2], args.r[2])
    gamma_vals = (args.gamma if len(args.gamma) == 1 else 
                  np.arange(args.gamma[0], args.gamma[1] + args.gamma[2], args.gamma[2]))
    beta_vals = (args.beta if len(args.beta) == 1 else 
                 np.arange(args.beta[0], args.beta[1] + args.beta[2], args.beta[2]))
    
    param_combs = list(itertools.product(alpha_vals, r_vals, gamma_vals, beta_vals))
    
    if len(param_combs) == 1:
        raise ValueError("At least two parameter combinations required for optimization.")

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # Run all parameter combinations
    print_heading(
        f"Running {args.hmap_label} model for {len(param_combs)} parameter combinations..."
    )
    print(f"alpha: {alpha_vals}\nr: {r_vals}\ngamma: {gamma_vals}\nbeta: {beta_vals}\n")
    
    model_outputs = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(model_job)(
            params=params,
            surf=surf, 
            medmask=medmask, 
            hmap=hmap,
            args=args,
            dt_emp=dt_emp,
            dt=dt,
            tsteady=tsteady,
            parc=parc,
            out_dir=out_dir
        ) for params in tqdm(param_combs, desc="Running models")
    )

    # Filter out failed simulations
    valid_indices = [i for i, x in enumerate(model_outputs) if x is not None]
    print(f"{len(valid_indices)} of {len(param_combs)} parameter combinations were valid.")

    param_combs = [param_combs[i] for i in valid_indices]
    model_outputs = [model_outputs[i] for i in valid_indices]

    # Evaluate against empirical data
    if args.evaluation == "fit":
        print_heading("Fitting models to empirical data...")
        
        # Load empirical FC data
        emp_outputs = {}
        emp_fc_file = (
            f"{PROJ_DIR}/data/empirical/{args.species}/{data_desc}_desc-fc_"
            f"{space_desc}_hemi-L_nt-{args.nt_emp}.h5"
        )
        with h5py.File(emp_fc_file, "r") as f:
            emp_outputs['fc'] = f['fc_group'][:]

        # Load empirical FCD data if needed
        if "fcd_ks" in args.metrics:
            emp_fcd_file = (
                f"{PROJ_DIR}/data/empirical/{args.species}/{data_desc}_desc-fcd_"
                f"{space_desc}_hemi-L_freql-{args.band_freq[0]}_freqh-{args.band_freq[1]}_"
                f"nt-{args.nt_emp}.h5"
            )
            with h5py.File(emp_fcd_file, "r") as f:
                emp_outputs['fcd'] = f['fcd_group'][:]
        
        # Evaluate all models
        results = Parallel(n_jobs=args.n_jobs, backend="loky")(
            delayed(evaluate_model)(model_outputs[i], emp_outputs, args.metrics)
            for i in tqdm(range(len(model_outputs)), desc="Evaluating models")
        )

        # Save individual model results (skip for null models)
        if not is_null:
            for i in range(len(param_combs)):
                out_file = (
                    f"{out_dir}/model_alpha-{param_combs[i][0]:.1f}_"
                    f"r-{param_combs[i][1]:.1f}_gamma-{param_combs[i][2]:.3f}_"
                    f"beta-{param_combs[i][3]:.1f}.h5"
                )
                with h5py.File(out_file, "w") as f:
                    f.create_dataset('fc', data=model_outputs[i].get('fc', np.nan))
                    f.create_dataset('fcd', data=model_outputs[i].get('fcd', np.nan))

                    results_group = f.create_group('results')
                    for metric in args.metrics:
                        results_group.create_dataset(metric, data=results[i].get(metric, np.nan))

        # Find best model based on combined metric
        combined = [
            results[i].get('edge_fc_corr', 0) + 
            results[i].get('node_fc_corr', 0) + 
            (1 - results[i].get('fcd_ks', 0) if 'fcd_ks' in args.metrics else 0)
            for i in range(len(results))
        ]
        best_ind = np.argmax(combined)

        # Save best model
        with h5py.File(f"{out_dir}/best_model.h5", "w") as f:
            f.create_dataset('alpha', data=param_combs[best_ind][0])
            f.create_dataset('r', data=param_combs[best_ind][1])
            f.create_dataset('gamma', data=param_combs[best_ind][2])
            f.create_dataset('beta', data=param_combs[best_ind][3])
            f.create_dataset('fc', data=model_outputs[best_ind].get('fc', np.nan))
            f.create_dataset('fcd', data=model_outputs[best_ind].get('fcd', np.nan))

            results_group = f.create_group('results')
            for metric in args.metrics:
                results_group.create_dataset(metric, data=results[best_ind].get(metric, np.nan))

        # Print results
        print_heading("Final Results")
        results_str = ", ".join([
            f"{metric}: {results[best_ind].get(metric, np.nan):.3f}" 
            for metric in args.metrics
        ])
        print(
            f"Best Model | alpha: {param_combs[best_ind][0]:.1f}, "
            f"r: {param_combs[best_ind][1]:.1f}, "
            f"gamma: {param_combs[best_ind][2]:.3f}, "
            f"beta: {param_combs[best_ind][3]:.1f} | {results_str}"
        )

    elif args.evaluation == "crossval":
        print_heading("Fitting models using cross-validation...")

        # Load cross-validation data
        emp_outputs_train = []
        emp_outputs_test = []
        
        cv_fc_file = (
            f"{PROJ_DIR}/data/empirical/{args.species}/{data_desc}_desc-fc-kfold5_"
            f"{space_desc}_hemi-L_nt-{args.nt_emp}.h5"
        )
        with h5py.File(cv_fc_file, "r") as f:
            fcs_train = f['fc_train'][:]
            fcs_test = f['fc_test'][:]
        
        for i in range(args.n_splits):
            emp_outputs_train.append({'fc': fcs_train[:, :, i]})
            emp_outputs_test.append({'fc': fcs_test[:, :, i]})

        # Load FCD data if needed
        if "fcd_ks" in args.metrics:
            cv_fcd_file = (
                f"{PROJ_DIR}/data/empirical/{args.species}/{data_desc}_desc-fcd-kfold5_"
                f"{space_desc}_hemi-L_freql-{args.band_freq[0]}_freqh-{args.band_freq[1]}_"
                f"nt-{args.nt_emp}.h5"
            )
            with h5py.File(cv_fcd_file, "r") as f:
                fcds_train = f['fcd_train'][:]
                fcds_test = f['fcd_test'][:]
                
                for i in range(args.n_splits):
                    emp_outputs_train[i]['fcd'] = fcds_train[:, :, i]
                    emp_outputs_test[i]['fcd'] = fcds_test[:, :, i]

        # Run cross-validation
        split_results = []
        for i in range(args.n_splits):
            split_result = run_split(
                model_outputs,
                emp_outputs_train[i], 
                emp_outputs_test[i], 
                args
            )
            best_ind = split_result['best_ind']

            results_str = ", ".join([
                f"{metric}: {split_result['test_results'][metric]:.3f}" 
                for metric in args.metrics
            ])
            print(
                f"Split {i} | Best: alpha={param_combs[best_ind][0]:.2f}, "
                f"r={param_combs[best_ind][1]:.1f}, gamma={param_combs[best_ind][2]:.3f}, "
                f"beta={param_combs[best_ind][3]:.1f} | {results_str}"
            )
            
            split_results.append(split_result)

        # Save individual model results averaged across splits (skip for null models)
        if not is_null:
            train_results = [split['train_results'] for split in split_results]
            
            for i in range(len(param_combs)):
                out_file = (
                    f"{out_dir}/model_alpha-{param_combs[i][0]:.1f}_"
                    f"r-{param_combs[i][1]:.1f}_gamma-{param_combs[i][2]:.3f}_"
                    f"beta-{param_combs[i][3]:.1f}.h5"
                )
                with h5py.File(out_file, "w") as f:
                    f.create_dataset('fc', data=model_outputs[i].get('fc', np.nan))
                    f.create_dataset('fcd', data=model_outputs[i].get('fcd', np.nan))

                    results_group = f.create_group('results')
                    for metric in args.metrics:
                        metric_vals = np.array([
                            train_results[j][i][metric] for j in range(args.n_splits)
                        ])
                        results_group.create_dataset(metric, data=metric_vals)

        # Aggregate best model results across splits
        best_indices = [split['best_ind'] for split in split_results]
        best_combs = np.array([param_combs[idx] for idx in best_indices])
        
        best_results = {
            metric: np.array([
                split['test_results'][metric] for split in split_results
            ]) for metric in args.metrics
        }
        
        # Save best model
        with h5py.File(f"{out_dir}/best_model.h5", "w") as f:
            f.create_dataset('alpha', data=best_combs[:, 0])
            f.create_dataset('r', data=best_combs[:, 1])
            f.create_dataset('gamma', data=best_combs[:, 2])
            f.create_dataset('beta', data=best_combs[:, 3])
            f.create_dataset('fc', data=np.dstack([
                model_outputs[best_indices[i]].get('fc', np.nan) 
                for i in range(args.n_splits)
            ]))
            f.create_dataset('fcd', data=np.dstack([
                model_outputs[best_indices[i]].get('fcd', np.nan) 
                for i in range(args.n_splits)
            ]))
            
            results_group = f.create_group('results')
            for metric, values in best_results.items():
                results_group.create_dataset(metric, data=values)

        # Print average results across splits
        print_heading("Final Results (Averaged Across Splits)")
        results_str = ", ".join([
            f"{metric}: {np.mean(best_results[metric]):.3f}" 
            for metric in args.metrics
        ])
        print(
            f"Best Model | alpha: {np.mean(best_combs[:, 0]):.1f}, "
            f"r: {np.mean(best_combs[:, 1]):.1f}, "
            f"gamma: {np.mean(best_combs[:, 2]):.3f}, "
            f"beta: {np.mean(best_combs[:, 3]):.1f} | {results_str}"
        )

    # Save parameter combinations
    pd.DataFrame(param_combs, columns=['alpha', 'r', 'gamma', 'beta']).to_csv(
        f"{out_dir}/parameter_combinations.csv", index=False
    )

    t2 = time.time()
    print(f"\nTotal time: {(t2 - t1) / 3600:.2f} hours")
