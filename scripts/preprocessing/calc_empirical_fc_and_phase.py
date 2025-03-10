# Description: Calculate empirical FC and phase for each subject
# import packages
import os
import h5py
import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from brainspace.utils.parcellation import reduce_by_labels
from heteromodes.restingstate import calc_phase


PROJ_DIR = os.getenv("PROJ_DIR")

def calc_fc_and_phase(bold, tr, band_freq, parc=None):
    # Z-score bold data
    scaler = StandardScaler()
    bold_z = scaler.fit_transform(bold.T).T

    # Compute phase on vertex data
    phase = calc_phase(bold_z, tr=tr, lowcut=band_freq[0], highcut=band_freq[1])

    # Compute FC on parcellated data
    if parc is not None:
        bold_z = reduce_by_labels(bold_z, parc, axis=1)
    fc = np.corrcoef(bold_z)

    return fc, phase

if __name__ == "__main__":
    n_jobs = 10
    band_freq = (0.01, 0.1)
    n_subjs = 384

    # Load empirical BOLD data
    with h5py.File(f"{PROJ_DIR}/data/empirical/HCP_unrelated-445_rfMRI_hemi-L_nsubj-{n_subjs}_parc-None_fsLR4k_hemi-L_BOLD.hdf5", 'r') as bold_data:
        medmask = bold_data['medmask'][:]
        bold_emp = bold_data['bold'][:]
        subj_ids = bold_data['subj_ids'][:]

    bold_emp = bold_emp[medmask, :, :]

    # Parallelize the calculation of FC and phase
    print("Calculating empirical FC and Phase...")
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(calc_fc_and_phase)(
            bold=bold_emp[:, :, subj], 
            tr=0.72,
            band_freq=(band_freq[0], band_freq[1]),
        )
        for subj in range(n_subjs)
    )
    fc_emp_all, phase_emp_all = zip(*results)
    fc_emp_all = np.dstack(fc_emp_all)
    phase_emp_all = np.dstack(phase_emp_all)

    print("Shape of FC empirical data: ", fc_emp_all.shape)
    print("Shape of phase empirical data: ", phase_emp_all.shape)

    # Save the results as h5 file
    print("Saving the results...")
    with h5py.File(f"{PROJ_DIR}/data/empirical/HCP_unrelated-445_rfMRI_hemi-L_nsubj-{n_subjs}_parc-None_fsLR4k_hemi-L_FCandPhase.hdf5", 'w') as h5f:
        h5f.create_dataset('fc', data=fc_emp_all)
        h5f.create_dataset('phase', data=phase_emp_all)
        h5f.create_dataset('medmask', data=medmask)
        h5f.create_dataset('subj_ids', data=subj_ids)