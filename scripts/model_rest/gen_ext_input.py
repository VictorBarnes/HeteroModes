import os
import h5py
import numpy as np
import nibabel as nib
from dotenv import load_dotenv


load_dotenv()
PROJ_DIR = os.getenv("PROJ_DIR")

den = "4k"
parc_name = None
if parc_name is None:
    bold_data = h5py.File(f"{PROJ_DIR}/data/empirical/HCP_unrelated-445_rfMRI_hemi-L_nsubj-384_parc-None_fsLR{den}_hemi-L_BOLD.hdf5", 'r')
    bold_emp = bold_data['bold'][:]
    medmask = np.where(bold_emp[:, 0, 0] != 0, True, False)
else:
    parc_file = f"{PROJ_DIR}/data/parcellations/parc-{parc_name}_space-fsLR_den-{den}_hemi-L.label.gii"
    parc = nib.load(parc_file).darrays[0].data.astype(int)
    medmask = np.where(parc != 0, True, False)

nruns = 5
nT = 1200
nverts = np.sum(medmask)

dt = 0.09
tmax = 50 + (nT - 1) * 0.72
t = np.arange(0, tmax + dt, dt)
n_timepoints = len(t)

for i in range(nruns):
    print(f"Generating external input {i}. Shape: {nverts} x {n_timepoints}")
    file = f"{PROJ_DIR}/data/resting_state/extInput_parc-{parc_name}_den-{den}_nT-{nT}_randseed-{i}.npy"

    np.random.seed(i)
    ext_input = np.random.randn(nverts, n_timepoints)
    np.save(file, ext_input)
