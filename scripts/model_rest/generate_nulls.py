# %%
import os
import json
import numpy as np
import nibabel as nib

from neuromodes.mesh import unmask_data
from neuromodes.eigen import EigenSolver
from neuromodes.nulls import eigenstrap
from heteromodes.utils import get_project_root, load_hmap

PROJ_DIR = get_project_root()
den = "32k"
species = "human"
n_nulls = 1000
n_modes = 500

#%%


surf = (
    f'{PROJ_DIR}/data/empirical/{species}/space-fsLR_den-{den}_'
    f'hemi-L_desc-midthickness.surf.gii'
)
medmask = nib.load(
    f"{PROJ_DIR}/data/empirical/{species}/space-fsLR_den-{den}_"
    f"hemi-L_desc-nomedialwall.func.gii"
).darrays[0].data.astype(bool)

with open(f"{PROJ_DIR}/data/heteromaps/{species}/heteromaps_config.json", "r") as f:
    config = json.load(f)
config.pop("SAaxis", None)
config.pop("curvature", None)
config.pop("hcpep-hc-myelinmap", None)
config.pop("hcpep-scz-myelinmap", None)

for hmap_label in config.keys():
    print(f"Generating nulls for {hmap_label}...")
    hmap = load_hmap(hmap_label, species=species, density=den)

    solver = EigenSolver(
        geometry=surf,
        mask=medmask
    ).solve(n_modes=n_modes)
    nulls = solver.eigenstrap(
        data=hmap[medmask], 
        n_nulls=n_nulls, 
        resample="exact",
        seed=365
    )
    nulls_out = unmask_data(nulls, mask=medmask)

    fname = f"{PROJ_DIR}/data/nulls/{species}/data-{hmap_label}_space-fsLR_den-{den}_hemi-L_nmodes-{n_modes}_nnulls-{n_nulls}_nulls_resample-True.npy"
    np.save(fname, nulls_out)
