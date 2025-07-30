import os
import h5py
import fbpca
import argparse
import numpy as np

import nibabel as nib
from scipy.stats import zscore
from dotenv import load_dotenv

from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from brainspace.utils.parcellation import reduce_by_labels
from heteromodes.restingstate import calc_fc, calc_hilbert, calc_fcd_efficient
from heteromodes.utils import pad_sequences


load_dotenv()
PROJ_DIR = os.getenv("PROJ_DIR")
DENSITIES = {"4k": 4002, "8k": 7842, "32k": 32492}

def calc_outputs(bold, fnq, band_freq):
    # Compute FC
    fc = calc_fc(bold).astype(np.float32)
    
    complex_data = calc_hilbert(bold, fnq=fnq, band_freq=band_freq).astype(np.complex64)

    fcd = calc_fcd_efficient(bold, fnq=fnq, band_freq=band_freq).astype(np.float32)

    return fc, complex_data, fcd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare empirical data for analysis.")
    parser.add_argument("--species", type=str, default="human", help="Species of the data (e.g., 'human', 'macaque', 'marmoset').")
    parser.add_argument("--den", type=str, default="4k", help="Density of the data (e.g., '4k', '8k', '32k').")
    parser.add_argument("--crossval", action="store_true", default=False, help="Whether to perform cross-validation.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of splits for KFold cross-validation.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs.")
    parser.add_argument("--band_freq", type=float, nargs=2, default=(0.04, 0.07), help="Frequency band (low, high) for filtering.")
    parser.add_argument("--parc", type=str, default=None, help="Parcellation to use (e.g., 'schaefer300', 'hcpmmp1').")
    parser.add_argument("--nt_emp", type=int, default=600, help="Number of time points in empirical data.")
    args = parser.parse_args()

    # Fixed parameters
    ncpcs = 10
    TR_SPECIES = {"human": 0.72, "macaque": 0.75, "marmoset": 2.0}
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

    tr = TR_SPECIES[args.species]
    data_dir = DATA_DIR_SPECIES[args.species]
    data_desc = DATA_DESC_SPECIES[args.species]
    nsubj_expected = int(data_desc.split("nsubj-")[-1])

    fnq = 0.5 * 1/tr
    
    # Load subject IDs
    subj_ids = np.loadtxt(f"{PROJ_DIR}/data/empirical/{args.species}/{data_desc}_desc-subjects.txt", dtype=str)

    bold_all = []
    subj_ids_valid = []
    for i, subj in enumerate(subj_ids):
        print(f"Processing subject {subj} ({i+1}/{len(subj_ids)})")

        ### Load BOLD data ###
        if args.species == "human":
            if args.den == "32k":
                bold_folder = f"{data_dir}/{subj}/MNINonLinear/Results/rfMRI_REST1_LR"
                bold_file = f"{bold_folder}/rfMRI_REST1_LR_Atlas_hp2000_clean.L.func.gii"
                medmask_dir = f"{data_dir}"
            else:
                bold_folder = f"{data_dir}/{subj}/MNINonLinear/Results/rfMRI_REST1_LR/resampled"
                bold_file = f"{bold_folder}/rfMRI_REST1_LR_Atlas_hp2000_clean_{args.den}.L.func.gii"
                medmask_dir = f"{data_dir}/resampled"
        elif args.species == "marmoset":
            bold_folder = f"{data_dir}/fmri"
            bold_file = f"{bold_folder}/{subj}_space-fsLR_den-{args.den}_hemi-L_desc-fmri_smooth-2.func.gii"
        elif args.species == "macaque":
            bold_folder = f"{data_dir}/fmri"
            bold_file = f"{bold_folder}/{subj}_space-fsLR_den-{args.den}_hemi-L_desc-fmri_clean.func.gii"
        else:
            raise ValueError(f"Unknown species: {args.species}")
        
        # Check if the file exists and load it
        if os.path.exists(f"{bold_file}"):
            bold = np.array(nib.load(bold_file).agg_data(), dtype=np.float32).T
            bold = bold[:, :args.nt_emp]  # Ensure we only take the first nt_emp time points
        else:
            print(f"File not found ({bold_file}). Skipping... ")
            continue
        
        # Check that shape is valid
        if np.shape(bold) != (DENSITIES[args.den], args.nt_emp):
            print(f"Skipping subject {subj} due to invalid shape")
            continue

        # Mask medial wall and Z-score
        medmask = nib.load(f"{medmask_dir}/space-fsLR_den-{args.den}_hemi-L_desc-nomedialwall.func.gii").darrays[0].data.astype(bool)
        bold_z = zscore(bold[medmask, :], axis=1)

        # Check for NaNs
        if np.any(np.isnan(bold_z)):
            print(f"Warning: NaNs found in subject {subj}.")
        # Check for infs
        if np.any(np.isinf(bold_z)):
            print(f"Warning: Infs found in subject {subj}.")
        # Check for constant time series
        if np.any(np.std(bold_z, axis=1) == 0):
            print(f"Warning: Constant time series found in subject {subj}.")
        
        bold_all.append(bold_z)
        subj_ids_valid.append(subj)

    # convert to numpy array
    subj_ids_valid = np.array(subj_ids_valid)
    n_subjs_valid = len(subj_ids_valid)
    if n_subjs_valid != nsubj_expected:
        raise ValueError(f"Warning: Number of subjects ({n_subjs_valid}) does not match expected ({nsubj_expected}).")
    
    print(f"Number of subjects with valid data: {n_subjs_valid}")
    bold_all = np.dstack(bold_all)
    nverts = np.shape(bold_all)[0]
    nt_emp = np.shape(bold_all)[1]

    # Parallelize the calculation of FC and phase
    print("Calculating empirical outputs from BOLD-fMRI..")
    results = Parallel(n_jobs=args.n_jobs, verbose=1, backend="loky")(
        delayed(calc_outputs)(
            bold=bold_all[:, :, subj],
            fnq=fnq,
            band_freq=args.band_freq,
        )
        for subj in range(n_subjs_valid)
    )
    fc_indiv, complex_indiv, fcd_indiv = zip(*results)

    # Use memmaps to save memory
    fc_indiv_mm = np.memmap(f"{PROJ_DIR}/data/temp/fc_memmap.dat", mode='w+', dtype=np.float32, shape=(nverts, nverts, n_subjs_valid))
    fc_indiv_mm[:] = np.dstack(fc_indiv)
    fc_group = np.mean(fc_indiv_mm, axis=2, dtype=np.float32)
    complex_indiv_mm = np.memmap(f"{PROJ_DIR}/data/temp/phase_memmap.dat", mode='w+', dtype=np.complex64, shape=(nverts, nt_emp, n_subjs_valid))
    complex_indiv_mm[:] = np.dstack(complex_indiv)
    fcd_indiv_mm = np.memmap(f"{PROJ_DIR}/data/temp/fcd_memmap.dat", mode='w+', dtype=np.float32, shape=(len(fcd_indiv[0]), n_subjs_valid))
    fcd_indiv_mm[:] = np.array(fcd_indiv).T
    fcd_group = fcd_indiv_mm

    print("Calculating group phase CPCs...")
    l = 10 + ncpcs
    _, svals_group, V_group = fbpca.pca(
        np.hstack([complex_indiv_mm[:, :, i] for i in range(n_subjs_valid)], dtype=np.complex64).T, 
        k=ncpcs, 
        n_iter=20, 
        l=l
    )
    cpcs_group = V_group.T

    if args.crossval:
        print("Preparing KFold cross-validation data...")
        fc_train = np.empty((nverts, nverts, args.n_splits))
        fc_test = np.empty((nverts, nverts, args.n_splits))

        fcd_train, fcd_test = [], []
        cpcs_train = np.empty((nverts, ncpcs, args.n_splits))
        cpcs_test = np.empty((nverts, ncpcs, args.n_splits))
        svals_train, svals_test = np.empty((ncpcs, args.n_splits)), np.empty((ncpcs, args.n_splits))
        phase_combined_train = np.empty((nverts, args.n_splits))
        phase_combined_test = np.empty((nverts, args.n_splits))
        train_subjs, test_subjs = [], []
        kf = KFold(n_splits=args.n_splits, shuffle=False)
        for i, (train_idx, test_idx) in enumerate(kf.split(subj_ids_valid)):
            print(f"Split {i+1}/{args.n_splits}")
            print("Computing FC")
            fc_train[:, :, i] = np.mean(fc_indiv_mm[:, :, train_idx], axis=2)
            fc_test[:, :, i] = np.mean(fc_indiv_mm[:, :, test_idx], axis=2)

            fcd_train.append(np.vstack([fcd_indiv_mm[:, j] for j in train_idx], dtype=np.float32).T)
            fcd_test.append(np.vstack([fcd_indiv_mm[:, j] for j in test_idx], dtype=np.float32).T)

            print("Computing CPCs")
            _, s_train, V_train = fbpca.pca(
                np.hstack([complex_indiv_mm[:, :, i] for i in train_idx]).T, 
                k=ncpcs, 
                n_iter=20, 
                l=l
            )
            _, s_test, V_test = fbpca.pca(
                np.hstack([complex_indiv_mm[:, :, i] for i in test_idx]).T, 
                k=ncpcs, 
                n_iter=20, 
                l=l
            )
            cpcs_train[:, :, i] = V_train.T
            cpcs_test[:, :, i] = V_test.T
            svals_train[:, i] = s_train
            svals_test[:, i] = s_test

            train_subjs.append(subj_ids_valid[train_idx])
            test_subjs.append(subj_ids_valid[test_idx])

        fcd_train = np.dstack(fcd_train).astype(np.float32)

    # Save the results as h5 files
    print("Saving the results...")
    if args.parc is not None:
        space_desc = f"space-fsLR_den-{args.den}_parc-{args.parc}"
    else:
        space_desc = f"space-fsLR_den-{args.den}"

    subj_ids_valid = np.array(subj_ids_valid, dtype=np.int32)
    train_subjs = np.array(pad_sequences(train_subjs), dtype=np.int32) if args.crossval else None
    test_subjs = np.array(pad_sequences(test_subjs), dtype=np.int32) if args.crossval else None
    with h5py.File(f"{PROJ_DIR}/data/empirical/{args.species}/{data_desc}_desc-bold_{space_desc}_hemi-L_nt-{args.nt_emp}.h5", 'w') as f:
        f.create_dataset('bold', data=bold_all, dtype=np.float32)
        f.create_dataset('subj_ids', data=subj_ids_valid)
        f.create_dataset('medmask', data=medmask)

    with h5py.File(f"{PROJ_DIR}/data/empirical/{args.species}/{data_desc}_desc-fc_{space_desc}_hemi-L_nt-{args.nt_emp}.h5", 'w') as h5f:
        h5f.create_dataset('fc_indiv', data=fc_indiv_mm, dtype=np.float32)
        h5f.create_dataset('fc_group', data=fc_group, dtype=np.float32)
        h5f.create_dataset('medmask', data=medmask)
        h5f.create_dataset('subj_ids', data=subj_ids_valid)

    with h5py.File(f"{PROJ_DIR}/data/empirical/{args.species}/{data_desc}_desc-fcd_{space_desc}_hemi-L_freql-{args.band_freq[0]:.1g}_freqh-{args.band_freq[1]:.1g}_nt-{args.nt_emp}.h5", 'w') as h5f:
        h5f.create_dataset('fcd_group', data=fcd_indiv_mm, dtype=np.float32)
        h5f.create_dataset('medmask', data=medmask)
        h5f.create_dataset('subj_ids', data=subj_ids_valid)

    with h5py.File(f"{PROJ_DIR}/data/empirical/{args.species}/{data_desc}_desc-cpcs_{space_desc}_hemi-L_freql-{args.band_freq[0]:.1g}_freqh-{args.band_freq[1]:.1g}_nt-{args.nt_emp}.h5", 'w') as h5f:
        h5f.create_dataset('cpcs_group', data=cpcs_group, dtype=np.complex64)
        h5f.create_dataset('svals_group', data=svals_group, dtype=np.float32)
        h5f.create_dataset('medmask', data=medmask)
        h5f.create_dataset('subj_ids', data=subj_ids_valid)

    if args.crossval:
        with h5py.File(f"{PROJ_DIR}/data/empirical/{args.species}/{data_desc}_desc-fc-kfold{args.n_splits}_{space_desc}_hemi-L_nt-{args.nt_emp}.h5", 'w') as h5f:
            h5f.create_dataset('fc_train', data=fc_train, dtype=np.float32)
            h5f.create_dataset('fc_test', data=fc_test, dtype=np.float32)
            h5f.create_dataset('medmask', data=medmask)
            h5f.create_dataset('subj_ids_train', data=train_subjs)
            h5f.create_dataset('subj_ids_test', data=test_subjs)

        with h5py.File(f"{PROJ_DIR}/data/empirical/{args.species}/{data_desc}_desc-fcd-kfold{args.n_splits}_{space_desc}_hemi-L_freql-{args.band_freq[0]:.1g}_freqh-{args.band_freq[1]:.1g}_nt-{args.nt_emp}.h5", 'w') as h5f:
            h5f.create_dataset('fcd_train', data=fcd_train, dtype=np.float32)
            h5f.create_dataset('fcd_test', data=fcd_test, dtype=np.float32)
            h5f.create_dataset('medmask', data=medmask)
            h5f.create_dataset('subj_ids_train', data=train_subjs)
            h5f.create_dataset('subj_ids_test', data=test_subjs)

        with h5py.File(f"{PROJ_DIR}/data/empirical/{args.species}/{data_desc}_desc-cpcs-kfold{args.n_splits}_{space_desc}_hemi-L_freql-{args.band_freq[0]:.1g}_freqh-{args.band_freq[1]:.1g}_nt-{args.nt_emp}.h5", 'w') as h5f:
            h5f.create_dataset('cpcs_train', data=cpcs_train, dtype=np.complex64)
            h5f.create_dataset('svals_train', data=svals_train, dtype=np.float32)
            h5f.create_dataset('cpcs_test', data=cpcs_test, dtype=np.complex64)
            h5f.create_dataset('svals_test', data=svals_test, dtype=np.float32)
            h5f.create_dataset('medmask', data=medmask)
            h5f.create_dataset('subj_ids_train', data=train_subjs)
            h5f.create_dataset('subj_ids_test', data=test_subjs)
    
    # Remove memmap files
    os.remove(f"{PROJ_DIR}/data/temp/fc_memmap.dat")
    os.remove(f"{PROJ_DIR}/data/temp/phase_memmap.dat")
