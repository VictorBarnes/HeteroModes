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
from heteromodes.utils import load_hmap
from nsbtools.utils import load_project_env
from heteromodes.restingstate import run_model, analyze_bold, evaluate_model

load_project_env()
PROJ_DIR = os.getenv("PROJ_DIR")
os.environ["NUMEXPR_MAX_THREADS"] = "1" # this is for calc_fcd_efficient

def print_heading(text):
    print(f"\n{text}\n{'=' * len(text)}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Model resting-state fMRI BOLD data and evaluate against empirical data.")
    parser.add_argument("--id", type=str, default=0, help="The id of the run for saving outputs.")
    parser.add_argument("--hmap_label", type=lambda x: None if x.lower() == "none" else x, default=None, 
                        help="The label of the heterogeneity map. Defaults to None (indicating homogeneity)")
    parser.add_argument("--n_runs", type=int, default=10, 
                        help="The number of runs to simulate. Defaults to 50.")
    parser.add_argument("--n_modes", type=int, default=500, 
                        help="The number of modes to calculate. Defaults to 500.")
    parser.add_argument("--n_splits", type=int, default=5, 
                        help="The number of splits for cross-validation. Defaults to 5.")
    parser.add_argument("--n_subjs", type=int, default=255, 
                        help="The number of subjects in the empirical data. Defaults to 255.")
    parser.add_argument("--n_jobs", type=int, default=-1, 
                        help="The number of CPUs for parallelization. Defaults to -1")
    parser.add_argument('--alpha', type=float, nargs=3, default=[-5, 5, 0.1], metavar=('alpha_min', 'alpha_max', 'alpha_step'), 
                        help='The alpha_min, alpha_max, and alpha_step values for scaling the heterogeneity map.')
    parser.add_argument("--beta", type=float, default=[1.0], nargs="+", metavar="beta values",
                        help="The beta value for scaling the heterogeneity map. Defaults to 1.0. Provide either a single beta value or three values (beta_min, beta_max, beta_step).")
    parser.add_argument("--r", type=float, nargs="+", default=[28.9], metavar="r_values", 
                        help="The spatial length scale of the wave model. Defaults to 28.9 s^-1. Provide either a single r value or three values (r_min, r_max, r_step).")
    parser.add_argument("--gamma", type=float, nargs="+", default=[0.116], metavar="gamma values",
                        help="The dampening rate of the wave model. Defaults to 0.116 s^-1. Provide either a single gamma value or three values (gamma_min, gamma_max, gamma_step).")
    parser.add_argument("--den", type=str, default="4k", 
                        help="The density of the surface. Defaults to `4k`.")
    parser.add_argument("--band_freq", type=float, nargs=2, default=[0.04, 0.07], metavar=('low', 'high'), 
                        help="The low and high bandpass frequencies for filtering the BOLD data. Defaults to [0.01, 0.1].")
    parser.add_argument("--metrics", type=str, nargs='+', default=["edge_fc_corr", "node_fc_corr", "fcd_ks"], 
                        help="The metrics to use for evaluation. Defaults to ['edge_fc', 'node_fc', 'fcd']")
    parser.add_argument("--crossval", type=lambda x: x.lower() == "true", default=False, 
                        help="Whether to perform cross-validation. Defaults to False.")
    parser.add_argument("--scaling", type=str, default="sigmoid",
                        help="The scaling to apply to the heterogeneity map. Defaults to 'sigmoid'.")
    parser.add_argument('--q_norm', type=lambda x: None if x.lower() == "none" else x, default=None, 
                        help="Type of distribution to match to when doing the quantile normalisation")
    parser.add_argument("--phase_type", type=str, default="cpc1",
                        help="The type of phase to calculate: 'cpc1' or 'combined'. Defaults to 'cpc1'")
    parser.add_argument("--n_comps", type=int, default=3,
                        help="The number of components to calculate for the phase map if phase_type == 'combined'. Defaults to 3.")
    parser.add_argument("--nt_emp", type=int, default=600,
                        help="The number of time points in the empirical data. Defaults to 1200.")
    parser.add_argument("--species", type=str, default="human",
                        help="The species of the data. Defaults to 'human'.")
    
    return parser.parse_args()

def model_job(params, surf, medmask, hmap, args, dt_emp, dt, tsteady):
    """
    Run model simulation and analyze BOLD data in one step.
    
    Parameters:
    -----------
    params : tuple
        (alpha, r, gamma) parameter combination
    ... other parameters for run_model ...
    
    Returns:
    --------
    results : dict or None
        Dictionary containing fc, phase_map, etc. or None if simulation failed
    """
    
    # Run simulation
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
            q_norm=args.q_norm,
            n_modes=args.n_modes,
            n_runs=args.n_runs,
            dt_emp=dt_emp,
            nt_emp=args.nt_emp,
            dt_model=dt,
            tsteady=tsteady
        )
        
        # Check if simulation was successful
        if bold_data is None:
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
    model_outputs: list of dictionaries, each containing model results
    emp_outputs_train: dictionary containing empirical training data
    emp_outputs_test: dictionary containing empirical test data
    """
    
    # Evaluate each model on train data    
    train_results = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(evaluate_model)(
            model_outputs[i],  # Dictionary with fc, phase_map, etc.
            emp_outputs_train,       # Dictionary with fc, phase_map, etc.
            args.metrics
        ) for i in range(len(model_outputs))
    )

    # Check if this is the best model so far
    combined = [
        results.get('edge_fc_corr', 0) + results.get('node_fc_corr', 0) + 1 - results.get('fcd_ks', 0) 
        for results in train_results
    ]
    best_ind = np.argmax(combined)   

    # Evaluate best model on test data
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

    args = parse_arguments()
    if args.hmap_label == "None":
        args.hmap_label = None
    if args.q_norm == "None":
        args.q_norm = None

    TR_SPECIES = {"human": 720, "macaque": 750, "marmoset": 2000} # in ms
    DT_SPECIES = {"human": 90, "macaque": 50, "marmoset": 50} # in ms
    DATA_DIR_SPECIES = {
        "human": "/fs03/kg98/vbarnes/HCP",
        "macaque": "/fs03/kg98/vbarnes/nhp_nnp/macaque",
        "marmoset": "/fs03/kg98/vbarnes/nhp_nnp/marmoset"
    }
    DATA_DESC_SPECIES = {
        "human": "hcp-s1200_nsubj-255",
        "macaque": "_nsubj-10", # TODO:
        "marmoset": "mbm-v4_nsubj-39"
    }

    dt_emp = TR_SPECIES[args.species]
    dt = DT_SPECIES[args.species]  # model time step (ms)
    data_dir = DATA_DIR_SPECIES[args.species]
    data_desc = DATA_DESC_SPECIES[args.species]
    
    # Define constants
    tsteady = 5 * 1e4  # burn time to remove transient effects (ms)

    out_dir = f'{PROJ_DIR}/results/{args.species}/model_rest/group/id-{args.id}/{args.hmap_label}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Get surface
    fslr = fetch_atlas(atlas='fsLR', density=args.den)
    surf = str(fslr['midthickness'][0])
    # Load medmask
    medmask = nib.load(f"{PROJ_DIR}/data/empirical/{args.species}/space-fsLR_den-{args.den}_hemi-L_desc-nomedialwall.func.gii").darrays[0].data.astype(bool)

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
        alpha_vals = np.linspace(alpha_min, alpha_max, alpha_num).round(len(str(alpha_step).split('.')[-1]))
        if alpha_min < 0 < alpha_max:
            alpha_vals = alpha_vals[alpha_vals != 0] # Remove 0 since that is the homogeneous case
        print(alpha_vals)
    r_vals = args.r if len(args.r) == 1 else np.arange(args.r[0], args.r[1], args.r[2])
    gamma_vals = args.gamma if len(args.gamma) == 1 else np.arange(args.gamma[0], args.gamma[1], args.gamma[2])
    beta_vals = args.beta if len(args.beta) == 1 else np.arange(args.beta[0], args.beta[1], args.beta[2])
    param_combs = list(itertools.product(alpha_vals, r_vals, gamma_vals, beta_vals))

    # Run all models (i.e. for all parameter combinations) and store outputs in cache
    print_heading(f"Running {args.hmap_label} model for {len(param_combs)} parameter combinations...")
    model_outputs = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(model_job)(
            params=params,
            surf=surf, 
            medmask=medmask, 
            hmap=hmap,
            args=args,
            dt_emp=dt_emp,
            dt=dt,
            tsteady=tsteady
        ) for params in tqdm(param_combs, desc="Running models")
    )

    # Find invalid combinations and remove those 
    nan_indices = [i for i, x in enumerate(model_outputs) if x is None]
    print(f"{len(param_combs) - len(nan_indices)} out of {len(param_combs)} parameter combinations were valid.")

    param_combs = [param_combs[i] for i in range(len(param_combs)) if i not in nan_indices]
    model_outputs = [model_outputs[i] for i in range(len(model_outputs)) if i not in nan_indices]

    # Evaluate against empirical data
    if not args.crossval:
        print_heading("Fitting models to empirical data...")
        
        # Load empirical data and create emp_outputs dictionary
        emp_outputs = {}
        with h5py.File(f"{PROJ_DIR}/data/empirical/{args.species}/{data_desc}_desc-fc_space-fsLR_den-{args.den}_hemi-L.h5", "r") as f:
            emp_outputs['fc'] = f['fc_group'][:]

        if "fcd_ks" in args.metrics:
            with h5py.File(f"{PROJ_DIR}/data/empirical/{args.species}/{data_desc}_desc-fcd_space-fsLR_den-{args.den}_hemi-L_freql-{args.band_freq[1]}_freqh-{args.band_freq[1]}.h5", "r") as f:
                emp_outputs['fcd_ks'] = f['fcd_group'][:]
        
        # Evaluate all models
        all_results = []
        best_score = -np.inf
        best_ind = 0
        
        for param_i in range(len(param_combs)):
            results = evaluate_model(
                model_outputs[param_i],
                emp_outputs,
                args.metrics
            )
            all_results.append(results)
            
            # Check if this is the best model
            combined_score = sum(results.values())
            if combined_score > best_score:
                best_score = combined_score
                best_ind = param_i

        best_combs = param_combs[best_ind]
        best_model_outputs = model_outputs[best_ind]

    else:
        print_heading("Fitting models to empirical data using cross validation...")

        # Prepare empirical outputs for each split
        emp_outputs_train = []
        emp_outputs_test = []
        
        # Load empirical cross-validation data
        with h5py.File(f"{PROJ_DIR}/data/empirical/{args.species}/{data_desc}_desc-fc-kfold5_space-fsLR_den-{args.den}_hemi-L_nt-{args.nt_emp}.h5", "r") as f:
            fcs_train = f['fc_train'][:]
            fcs_test = f['fc_test'][:]
        
        for i in range(args.n_splits):
            emp_train = {'fc': fcs_train[:, :, i]}
            emp_test = {'fc': fcs_test[:, :, i]}
            emp_outputs_train.append(emp_train)
            emp_outputs_test.append(emp_test)

        if "fcd_ks" in args.metrics:
            with h5py.File(f"{PROJ_DIR}/data/empirical/{args.species}/{data_desc}_desc-fcd-kfold5_space-fsLR_den-{args.den}_hemi-L_freql-{args.band_freq[0]}_freqh-{args.band_freq[1]}_nt-{args.nt_emp}.h5", "r") as f:
                fcds_train = f['fcd_train'][:]
                fcds_test = f['fcd_test'][:]
                for i in range(args.n_splits):
                    emp_outputs_train[i]['fcd'] = fcds_train[:, :, i]
                    emp_outputs_test[i]['fcd'] = fcds_test[:, :, i]

        # Run cross-validation splits
        split_results = []
        for i in range(args.n_splits):

            split_results_i = run_split(
                model_outputs,
                emp_outputs_train[i], 
                emp_outputs_test[i], 
                args
            )

            best_ind_i = split_results_i['best_ind']

            resuls_str = ", ".join([
                f"{metric}: {split_results_i['test_results'][metric]:.3f}" for metric in args.metrics
            ])
            print(f"Split {i} Best Model | alpha: {param_combs[best_ind_i][0]:.2f}, r: {param_combs[best_ind_i][1]:.1f}, "
                    f"gamma: {param_combs[best_ind_i][2]:.3f}, beta: {param_combs[best_ind_i][3]} | {resuls_str}")
            
            split_results.append(split_results_i)

        # Save results for each model by averaging across splits
        train_results = [split['train_results'] for split in split_results]
        for i in range(len(param_combs)):
            results_i = {}
            for metric in args.metrics:
                results_i[metric] = np.array([train_results[j][i][metric] for j in range(args.n_splits)])

            out_file = f"{out_dir}/results_alpha-{param_combs[i][0]:.1f}_r-{param_combs[i][1]:.1f}_gamma-{param_combs[i][2]:.3f}_beta-{param_combs[i][3]:.1f}.h5"
            with h5py.File(out_file, "w") as f:
                f.create_dataset('fc', data=model_outputs[i].get('fc', np.nan))
                f.create_dataset('fcd', data=model_outputs[i].get('fcd', np.nan))

                group = f.create_group('results')
                for key, value in results_i.items():
                    group.create_dataset(key, data=value)

        # Save best model outputs
        split_results_i = [split['test_results'] for split in split_results]
        best_ind = [split['best_ind'] for split in split_results]

        best_combs = np.array([param_combs[best_ind[i]] for i in range(args.n_splits)])
        best_results = {metric: np.array([
            split_results_i[i][metric] for i in range(args.n_splits)
        ]) for metric in args.metrics}
        with h5py.File(f"{out_dir}/best_model_results.h5", "w") as f:
            f.create_dataset('alpha', data=best_combs[:, 0])
            f.create_dataset('r', data=best_combs[:, 1])
            f.create_dataset('gamma', data=best_combs[:, 2])
            f.create_dataset('beta', data=best_combs[:, 3])
            f.create_dataset('fc', data=np.dstack([model_outputs[best_ind[i]].get('fc', np.nan) for i in range(args.n_splits)]))
            f.create_dataset('fcd', data=np.dstack([model_outputs[best_ind[i]].get('fcd', np.nan) for i in range(args.n_splits)]))
            
            group = f.create_group('results')
            for key, value in best_results.items():
                group.create_dataset(key, data=value)

        # Print final results (averages across splits)
        print_heading("Final Results")
        results_str = ", ".join([
            f"{metric}: {np.mean([split_results_i[i][metric] for i in range(args.n_splits)]):.3f}"
            for metric in args.metrics
        ])
        print(
            f"Best Model | alpha: {np.mean(best_combs, axis=0)[0]:.1f} "
            f"r: {np.mean(best_combs, axis=0)[1]:.1f} "
            f"gamma: {np.mean(best_combs, axis=0)[2]:.3f} "
            f"beta: {np.mean(best_combs, axis=0)[3]:.1f} | {results_str}"
        )

    # Save parameter combinations to csv file
    param_combs_df = pd.DataFrame(param_combs, columns=['alpha', 'r', 'gamma', 'beta'])
    param_combs_df.to_csv(f"{out_dir}/parameter_combinations.csv", index=False)

    t2 = time.time()
    print(f"\nTotal time: {(t2 - t1)/3600:.1f} hours")
