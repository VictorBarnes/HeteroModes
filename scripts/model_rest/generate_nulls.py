# %%
import os
import json
import numpy as np
import nibabel as nib
from neuromaps.datasets import fetch_atlas
from eigenstrapping import SurfaceEigenstrapping
from heteromodes.utils import load_hmap, get_project_root

PROJ_DIR = get_project_root()

with open(f"{PROJ_DIR}/data/heteromaps/human/heteromap_labels.json", "r") as f:
    config = json.load(f)
heteromap_labels = config["heteromap_labels"]
resample = True
n_nuls = 1000
n_modes = 500
den = "4k"

fslr = fetch_atlas("fsLR", den)
surff = str(fslr["midthickness"][0])
medmask = nib.load(f"{PROJ_DIR}/data/empirical/human/space-fsLR_den-4k_hemi-L_desc-nomedialwall.func.gii").darrays[0].data.astype(bool)

for hmap_label in heteromap_labels.keys():
    print(f"Processing {heteromap_labels[hmap_label]}...")
    hmap = load_hmap(hmap_label, species="human", trg_den=den)
    eigen = SurfaceEigenstrapping(surface=surff, data=hmap, medial=medmask, seed=365, num_modes=n_modes, 
                                resample=resample)
    nulls = eigen(n=n_nuls)

    fname = f"{PROJ_DIR}/data/nulls/data-{hmap_label}_space-fsLR_den-4k_hemi-L_nmodes-{n_modes}_nnulls-{n_nuls}_nulls_resample-{resample}"
    np.save(f"{fname}.npy", nulls)
