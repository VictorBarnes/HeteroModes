# %%
import os
import numpy as np
import nibabel as nib
from neuromaps.datasets import fetch_atlas
from eigenstrapping import SurfaceEigenstrapping
from nsbtools.utils import load_project_env
from heteromodes.utils import load_hmap

load_project_env()
PROJ_DIR = os.getenv("PROJ_DIR")

hetero_labels = {
    "myelinmap": "T1w/T2w",
    "thickness": "Cortical thickness",
    "synapticden": "Synaptic density",
    "odi": "ODI",
    "ndi": "NDI",
    "genel4PC1": "Layer IV",
    "eiratio1.2": "E:I ratio",
    "megtimescale": "MEG timescale"
}
resample = True
n_nuls = 1000
n_modes = 500
den = "4k"

fslr = fetch_atlas("fsLR", den)
surff = str(fslr["midthickness"][0])
medmask = nib.load(f"{PROJ_DIR}/data/empirical/human/space-fsLR_den-4k_hemi-L_desc-nomedialwall.func.gii").darrays[0].data.astype(bool)

for hmap_label in hetero_labels.keys():
    print(f"Processing {hetero_labels[hmap_label]}...")
    hmap = load_hmap(hmap_label, species="human", trg_den=den)
    eigen = SurfaceEigenstrapping(surface=surff, data=hmap, medial=medmask, seed=365, num_modes=n_modes, 
                                resample=resample)
    nulls = eigen(n=n_nuls)

    fname = f"{PROJ_DIR}/data/nulls/data-{hmap_label}_space-fsLR_den-4k_hemi-L_nmodes-{n_modes}_nnulls-{n_nuls}_nulls_resample-{resample}"
    np.save(f"{fname}.npy", nulls)
