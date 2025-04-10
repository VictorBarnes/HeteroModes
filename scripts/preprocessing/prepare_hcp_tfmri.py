import os
import h5py
import numpy as np
import nibabel as nib
from neuromaps.datasets import fetch_fslr
from heteromodes.utils import load_project_env

load_project_env()
PROJ_DIR = os.getenv("PROJ_DIR")
medmask = nib.load(fetch_fslr("32k")["medial"][0]).darrays[0].data.astype(bool)

# Load the .mat file
with h5py.File("/fs03/kg98/jamesp/github_repos/BrainEigenmodes/data/empirical/S255_tfMRI_ALLTASKS_raw_lh.mat", "r") as f:
    zstat_group = f["zstat"]
    
    # Extract dataset names
    contrast_names = list(zstat_group.keys())

    # Compute mean along axis 0 for each dataset
    contrast_maps = np.column_stack([np.nanmean(zstat_group[name][:, medmask], axis=0) for name in contrast_names])

print("contrast_names:", contrast_names)
print("contrast_maps shape:", contrast_maps.shape)

# Save the contrast maps to a h5py file
with h5py.File(f"{PROJ_DIR}/data/empirical/hcp-s255_tfMRI.hdf5", "w") as f:
    f.create_dataset("contrast_names", data=np.array(contrast_names, dtype="S"))
    f.create_dataset("contrast_maps", data=contrast_maps)
    f.create_dataset("medmask", data=medmask)
