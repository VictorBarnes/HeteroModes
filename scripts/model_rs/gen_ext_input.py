import os
import numpy as np
import nibabel as nib
from dotenv import load_dotenv

load_dotenv()
PROJ_DIR = os.getenv("PROJ_DIR")

den = "4k"
parc_name = "schaefer400"
parc_file = f"{PROJ_DIR}/data/parcellations/parc-{parc_name}_space-fsLR_den-{den}_hemi-L.label.gii"
parc = nib.load(parc_file).darrays[0].data.astype(int)
medmask = np.where(parc != 0, True, False)

nruns = 5
nverts = np.sum(medmask)

dt = 0.09
tmax = 50 + 1199 * 0.72
t = np.arange(0, tmax + dt, dt)
n_timepoints = len(t)

for i in range(nruns):
    print(f"Generating external input {i}. Shape: {nverts} x {n_timepoints}")
    file = f"{PROJ_DIR}/data/resting_state/extInput_parc-{parc_name}_den-{den}_hemi-L_randseed-{i}.npy"

    np.random.seed(i)
    ext_input = np.random.randn(nverts, n_timepoints)
    np.save(file, ext_input)
