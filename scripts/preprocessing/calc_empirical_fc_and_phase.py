# Description: Calculate empirical FC and phase for each subject
# import packages
import os
import h5py
import fbpca
import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.signal import hilbert
from heteromodes.restingstate import calc_gen_phase, calc_phase_map, filter_bold, calc_phase_cpcs
from heteromodes.utils import pad_sequences


PROJ_DIR = os.getenv("PROJ_DIR")

def fc_and_phase_job(bold, tr, band_freq, phase_type="complex"):
    # Z-score bold data
    scaler = StandardScaler()
    bold_z = scaler.fit_transform(bold.T).T

    # Compute FC
    fc = np.corrcoef(bold_z, dtype=np.float32)
    
    # Bandpass filter the BOLD signal
    bold_filtered = filter_bold(bold_z, tr=tr, lowcut=band_freq[0], highcut=band_freq[1])
    # Compute phase
    if phase_type == "complex":
        phase = hilbert(bold_filtered, axis=1).conj()
    elif phase_type == "generalized":
        phase = calc_gen_phase(bold_filtered, tr=tr, lowcut=band_freq[0], highcut=band_freq[1])

    return fc, phase.astype(np.complex64)

if __name__ == "__main__":
    n_jobs = 10
    n_splits = 5
    freql = 0.01
    freqh = 0.1
    n_subjs = 255
    ncpcs = 10
    ncpcs_comb = 3
    den = "4k"
    phase_type = "complex"

    # Load empirical BOLD data
    print("Loading empirical BOLD data...")
    with h5py.File(f"{PROJ_DIR}/data/empirical/HCP_nsubj-255_bold_parc-None_fsLR{den}_hemi-L.h5", 'r') as bold_data:
        medmask = bold_data['medmask'][:]
        bold_emp = bold_data['bold'][medmask, :, :n_subjs]
        subj_ids = bold_data['subj_ids'][:]
    nverts = np.sum(medmask)

    # Parallelize the calculation of FC and phase
    print("Calculating empirical FC and Phase...")
    results = Parallel(n_jobs=n_jobs, verbose=1, backend="multiprocessing")(
        delayed(fc_and_phase_job)(
            bold=bold_emp[:, :, subj],
            tr=0.72,
            band_freq=(freql, freqh),
            phase_type=phase_type
        )
        for subj in range(n_subjs)
    )
    fc_indiv, phase_indiv = zip(*results)

    # Use memmaps to save memory
    fc_indiv_mm = np.memmap(f"{PROJ_DIR}/data/temp/fc_memmap.dat", mode='w+', dtype=np.float32, shape=(nverts, nverts, n_subjs))
    fc_indiv_mm[:] = np.dstack(fc_indiv)
    fc_group = np.mean(fc_indiv_mm, axis=2, dtype=np.float32)
    phase_indiv_mm = np.memmap(f"{PROJ_DIR}/data/temp/phase_memmap.dat", mode='w+', dtype=np.complex64, shape=(nverts, 1200, n_subjs))
    phase_indiv_mm[:] = np.dstack(phase_indiv)

    print('calculating phase cpcs')
    l = 10 + ncpcs
    _, svals_group, V_group = fbpca.pca(
        np.hstack([phase_indiv_mm[:, :, i] for i in range(n_subjs)], dtype=np.complex64).T, 
        k=ncpcs, 
        n_iter=20, 
        l=l
    )
    phase_cpcs_group = np.real(V_group).T

    fc_train = np.empty((nverts, nverts, n_splits))
    fc_test = np.empty((nverts, nverts, n_splits))
    phase_cpcs_train = np.empty((nverts, ncpcs, n_splits))
    phase_cpcs_test = np.empty((nverts, ncpcs, n_splits))
    svals_train, svals_test = np.empty((ncpcs, n_splits)), np.empty((ncpcs, n_splits))
    phase_combined_train = np.empty((nverts, n_splits))
    phase_combined_test = np.empty((nverts, n_splits))
    train_subjs, test_subjs = [], []
    kf = KFold(n_splits=n_splits, shuffle=False)
    for i, (train_idx, test_idx) in enumerate(kf.split(subj_ids)):
        print(f"Split {i+1}/{n_splits}")
        print("Computing FC")
        fc_train[:, :, i] = np.mean(fc_indiv_mm[:, :, train_idx], axis=2)
        fc_test[:, :, i] = np.mean(fc_indiv_mm[:, :, test_idx], axis=2)

        print("Computing Phase")
        _, s_train, V_train = fbpca.pca(
            np.hstack([phase_indiv_mm[:, :, i] for i in train_idx]).T, 
            k=ncpcs, 
            n_iter=20, 
            l=l
        )
        _, s_test, V_test = fbpca.pca(
            np.hstack([phase_indiv_mm[:, :, i] for i in test_idx]).T, 
            k=ncpcs, 
            n_iter=20, 
            l=l
        )
        phase_cpcs_train[:, :, i] = np.real(V_train).T
        phase_cpcs_test[:, :, i] = np.real(V_test).T
        svals_train[:, i] = s_train
        svals_test[:, i] = s_test

        train_subjs.append(subj_ids[train_idx])
        test_subjs.append(subj_ids[test_idx])

    # Save the results as h5 file
    print("Saving the results...")
    # Save data here to free up memory
    with h5py.File(f"{PROJ_DIR}/data/empirical/HCP_nsubj-{n_subjs}_fc_parc-None_fsLR{den}_hemi-L.h5", 'w') as h5f:
        h5f.create_dataset('fc_indiv', data=fc_indiv_mm, dtype=np.float32)
        h5f.create_dataset('fc_group', data=fc_group, dtype=np.float32)
        h5f.create_dataset('medmask', data=medmask)
        h5f.create_dataset('subj_ids', data=subj_ids)

    with h5py.File(f"{PROJ_DIR}/data/empirical/HCP_nsubj-{n_subjs}_complexPhase_parc-None_fsLR{den}_hemi-L_freql-{freql:.1g}_freqh-{freqh:.1g}.h5", 'w') as h5f:
        h5f.create_dataset('phase', data=phase_indiv_mm, dtype=np.complex64)
        h5f.create_dataset('phase_cpcs_group', data=phase_cpcs_group, dtype=np.float32)
        h5f.create_dataset('svals_group', data=svals_group, dtype=np.float32)
        h5f.create_dataset('medmask', data=medmask)
        h5f.create_dataset('subj_ids', data=subj_ids)

    with h5py.File(f"{PROJ_DIR}/data/empirical/HCP_nsubj-{n_subjs}_fc-kfold{n_splits}_parc-None_fsLR{den}_hemi-L_freql-{freql:.1g}_freqh-{freqh:.1g}.h5", 'w') as h5f:
        h5f.create_dataset('fc_train', data=fc_train, dtype=np.float32)
        h5f.create_dataset('fc_test', data=fc_test, dtype=np.float32)
        h5f.create_dataset('medmask', data=medmask)
        h5f.create_dataset('subj_ids_train', data=pad_sequences(train_subjs))
        h5f.create_dataset('subj_ids_test', data=pad_sequences(test_subjs))

    with h5py.File(f"{PROJ_DIR}/data/empirical/HCP_nsubj-{n_subjs}_phasecpcs-kfold{n_splits}_parc-None_fsLR{den}_hemi-L_freql-{freql:.1g}_freqh-{freqh:.1g}.h5", 'w') as h5f:
        h5f.create_dataset('phase_cpcs_train', data=phase_cpcs_train, dtype=np.float32)
        h5f.create_dataset('svals_train', data=svals_train, dtype=np.float32)
        h5f.create_dataset('phase_cpcs_test', data=phase_cpcs_test, dtype=np.float32)
        h5f.create_dataset('svals_test', data=svals_test, dtype=np.float32)
        h5f.create_dataset('medmask', data=medmask)
        h5f.create_dataset('subj_ids_train', data=pad_sequences(train_subjs))
        h5f.create_dataset('subj_ids_test', data=pad_sequences(test_subjs))
    
    # Remove memmap files
    os.remove(f"{PROJ_DIR}/data/temp/fc_memmap.dat")
    os.remove(f"{PROJ_DIR}/data/temp/phase_memmap.dat")
