import os
import numpy as np
import h5py
from dotenv import load_dotenv
import optuna
import argparse
from joblib import Parallel, delayed
from neuromaps.datasets import fetch_atlas
import nibabel as nib
from sklearn.preprocessing import StandardScaler
from brainspace.utils.parcellation import reduce_by_labels
from heteromodes.utils import calc_phase_fcd, load_hmap
from heteromodes.restingstate import ModelBOLD, evaluate_model


load_dotenv()
PROJ_DIR = os.getenv("PROJ_DIR")
GLASSER360_LH = os.getenv("GLASSER360_LH")
SURF_LH = os.getenv("SURF_LH")

def calc_fc_fcd(bold_emp_subj):
    # Ensure data is standardised
    if not np.isclose(np.mean(bold_emp_subj, axis=1), 0).all() or not np.isclose(np.std(bold_emp_subj, axis=1), 1.0).all():
        scaler = StandardScaler()
        bold_emp_subj = scaler.fit_transform(bold_emp_subj.T).T
    # Caculate FC and FCD
    fc_emp = np.corrcoef(bold_emp_subj)
    fcd_emp = calc_phase_fcd(bold_emp_subj, tr=0.72)

    return fc_emp, fcd_emp

def run_model(surf, hmap, parc, medmask, run, alpha, r, gamma, args, fc_emp, fcd_emp):
    # Initialise model
    model_rs = ModelBOLD(surf_file=surf, medmask=medmask, hmap=hmap,
                         alpha=alpha, r=r, gamma=gamma, scale_method=args.scale_method)
    model_rs.calc_modes(args.n_modes, method=args.aniso_method)
    # Load external input, run model and parcellate
    ext_input = np.load(f"{PROJ_DIR}/data/resting_state/extInput_den-{args.den}_randseed-{run}.npy")
    bold_model = model_rs.run_rest(ext_input=ext_input)
    # parc = nib.load(parc).darrays[0].data.astype(int)#[medmask]
    bold_model = reduce_by_labels(bold_model, parc[medmask], axis=1)

    # Evaluate model
    results = evaluate_model(fc_emp, fcd_emp, bold_model, TR=0.72)

    return results[0], results[1], results[2], results[3], results[4]

class Objective(object):
    def __init__(self, surf, hmap, parc, medmask, args, fc_emp, fcd_emp):
        self.surf = surf
        self.hmap = hmap
        self.parc = parc
        self.medmask = medmask
        self.n_modes = args.n_modes
        self.args = args
        self.n_runs = args.n_runs
        self.n_jobs = args.n_jobs
        self.fc_emp = fc_emp
        self.fcd_emp = fcd_emp

    def __call__(self, trial):
        if self.hmap is not None:
            alpha = trial.suggest_float("alpha", -2, 2, step=0.1)
        else:
            alpha = 0
        r = trial.suggest_float("r", 10, 100, step=0.1)
        gamma = trial.suggest_float("gamma", 0.1, 1, step=0.001)

        # Run model
        results = Parallel(n_jobs=self.n_jobs, verbose=1)(
            delayed(run_model)(
                surf=self.surf,
                hmap=self.hmap,
                parc=self.parc,
                medmask=self.medmask,
                run=run,
                alpha=alpha,
                r=r,
                gamma=gamma,
                args=self.args,
                fc_emp=self.fc_emp,
                fcd_emp=self.fcd_emp,
            )
            for run in range(self.n_runs)
        )
        edge_fc, node_fc, fcd, _, _ = zip(*results)
        print(f"Edge FC: {np.mean(edge_fc):.3f} | Node FC: {np.mean(node_fc):.3f} | FCD: {np.mean(fcd):.3f}")

        combined_metric = np.mean(edge_fc) + np.mean(node_fc)# + 1 - np.mean(fcd)

        return combined_metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model resting-state fMRI BOLD data and evaluate against empirical data.")
    parser.add_argument("--hmap_label", type=str, default=None, help="The label of the heterogeneity map. Defaults to None (indicating homogeneity)")
    parser.add_argument("--study_name", type=str, help="The study name.")
    parser.add_argument("--scale_method", type=str, default="zscore", help="The scaling method for the heterogeneity map. Defaults to `zscore`.")
    parser.add_argument("--aniso_method", type=str, default="hetero", help="The method to calculate the modes. Defaults to `aniso`.")
    parser.add_argument("--n_runs", type=int, default=10, help="The number of runs to simulate. Defaults to 50.")
    parser.add_argument("--n_modes", type=int, default=500, help="The number of modes to calculate. Defaults to 500.")
    parser.add_argument("--n_splits", type=int, default=5, help="The number of splits for cross-validation. Defaults to 5.")
    parser.add_argument("--n_subjs", type=int, default=384, help="The number of subjects in the empirical data. Defaults to 384.")
    parser.add_argument("--n_jobs", type=int, default=-1, help="The number of CPUs for parallelization. Defaults to -1")
    parser.add_argument("--n_trials", type=int, default=100, help="The number of trials for the optimisation. Defaults to 100.")
    parser.add_argument("--den", type=str, default="32k", help="The density of the surface. Defaults to `32k`.")
    args = parser.parse_args()

    # Get surface, medial mask and parcellation files
    fslr = fetch_atlas(atlas='fsLR', density=args.den)
    surf = str(fslr['midthickness'][0])

    # Get hmap and alpha values
    if args.hmap_label is None:
        hmap = None
    else:
        hmap = load_hmap(args.hmap_label, den=args.den)

    parc_file = f"{PROJ_DIR}/data/parcellations/Q1-Q6_RelatedParcellation210.L.CorticalAreas_dil_Colors.{args.den}_fs_LR.label.gii"
    parc = nib.load(parc_file).darrays[0].data.astype(int)
    medmask = np.where(parc != 0, True, False)

    # Load empirical BOLD data
    # TODO: make this generalizable to different parcellations (parc_name = os.path.basename(args.parc_lh).split('_')[0])
    bold_data = h5py.File(f"{PROJ_DIR}/data/empirical/HCP_unrelated-445_rfMRI_hemi-L_nsubj-{args.n_subjs}_parc-glasser360_BOLD.hdf5", 'r')
    bold_emp = bold_data['bold']
    subj_ids = bold_data['subj_ids']
    _, _, nsubjs = np.shape(bold_emp)

    # Parallelize the calculation of FC and FCD
    print("Calculating empirical FC and FCD...")
    results = Parallel(n_jobs=args.n_jobs, verbose=1)(
        delayed(calc_fc_fcd)(bold_emp[:, :, subj])
        for subj in range(nsubjs)
    )
    fc_emp_all, fcd_emp_all = zip(*results)
    fc_emp_all = np.dstack(fc_emp_all)
    fcd_emp_all = np.array(fcd_emp_all)

    # Set output folder and description
    out_dir = f'{os.getenv("PROJ_DIR")}/results/model_rs/optuna/{args.hmap_label}'
    opt_storage = f"sqlite:///{out_dir}/optuna.db"

    # Run optimiser
    study = optuna.create_study(direction="maximize", storage=opt_storage, 
                                study_name=args.study_name, load_if_exists=True)
    objective = Objective(surf, hmap, parc, medmask, args, np.mean(fc_emp_all, axis=2), fcd_emp_all)
    study.optimize(objective, n_trials=args.n_trials)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
