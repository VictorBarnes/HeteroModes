import os
import h5py
import nibabel as nib
import numpy as np
from brainspace.utils.parcellation import reduce_by_labels
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from scipy.io import loadmat
from neuromaps.datasets import fetch_fslr
from heteromodes.utils import load_project_env

load_project_env()
PROJ_DIR = os.getenv("PROJ_DIR")
DENSITIES = {"4k": 4002, "32k": 32492}

parc_name = None
den = "4k"
fetch_sc = False

if parc_name is not None:
    try:
        parc_file = f"{PROJ_DIR}/data/parcellations/parc-{parc_name}_space-fsLR_den-{den}_hemi-L.label.gii"
        parc = nib.load(parc_file).darrays[0].data.astype(int)
    except:
        raise ValueError(f"Parcellation '{parc_name}' with den {den} not found.")
    medmask = np.where(parc != 0, True, False)
    n_parcels = len(np.unique(parc[medmask]))
else:
    if den == "32k":
        medmask = nib.load(fetch_fslr(den)["medial"][0]).darrays[0].data.astype(bool)
    else:
        medmask = nib.load(f"{PROJ_DIR}/data/empirical/fsLR-{den}_medmask.label.gii").darrays[0].data.astype(bool)


# Load subject IDs
subj_ids = np.loadtxt(f"{PROJ_DIR}/data/empirical/hcp-255_subj-ids.txt", dtype=int)

# Load bold data as matrix
scaler = StandardScaler()
bold_all = []
sc_all = []
subj_count = 0
for i, subj in enumerate(subj_ids):
    # if subj_count == nsubj_desired:
    #     break

    print(f"Processing subject {subj} ({i+1}/{len(subj_ids)})")

    ### Load BOLD data ###
    if den == "32k":
        bold_folder = f"/fs03/kg98/vbarnes/HCP/{subj}/MNINonLinear/Results/rfMRI_REST1_LR"
        bold_file = f"rfMRI_REST1_LR_Atlas_hp2000_clean.L.func.gii"
    else:
        bold_folder = f"/fs03/kg98/vbarnes/HCP/{subj}/MNINonLinear/Results/rfMRI_REST1_LR/resampled"
        bold_file = f"rfMRI_REST1_LR_Atlas_hp2000_clean_{den}.L.func.gii"
    try:
        bold = np.array(nib.load(Path(bold_folder, bold_file)).agg_data(), dtype=np.float32).T
    except:
        print(f"Skipping subject {subj} due to missing bold data")
        # Drop sub from sub_ids
        subj_ids = np.delete(subj_ids, i)
        continue
    
    # Check that shape is valid
    if np.shape(bold) != (DENSITIES[den], 1200):
        print(f"Skipping subject {subj} due to invalid shape")
        # Drop sub from sub_ids
        subj_ids = np.delete(subj_ids, i)
        continue
    # Parcellate
    if parc_name is not None:
        bold = reduce_by_labels(bold[medmask, :], parc[medmask], axis=1)
    # Z-score
    bold = scaler.fit_transform(bold.T).T
    bold_all.append(bold)

    ### Load SC data ###
    if fetch_sc:
        sc_folder = f"/fs03/kg98/vbarnes/HCP/{subj}/MNINonLinear/Results/connectome"
        sc_file = f"parc-HCPMMP1ANDfslatlas20acpc_tractalgo-iFOD2_sifttype-SIFT2_networktype-standard.mat"
        sc_all.append(loadmat(Path(sc_folder, sc_file))['connectome_subj'][:180, :180])

    subj_count += 1

bold_all = np.dstack(bold_all)

if parc_name is None:
    parc_name = f"None_fsLR{den}"

with h5py.File(f"{PROJ_DIR}/data/empirical/HCP_nsubj-{subj_count}_bold_parc-{parc_name}_hemi-L.h5", 'w') as f:
    f.create_dataset('bold', data=bold_all, dtype=np.float32)
    f.create_dataset('subj_ids', data=subj_ids)
    f.create_dataset('medmask', data=medmask)

if fetch_sc:
    sc_all = np.dstack(sc_all)
    sc_avg = np.nanmean(sc_all, axis=2)

    with h5py.File(f"{PROJ_DIR}/data/empirical/HCP_nsubj-{subj_count}_sc_hemi-L_nsubj-{subj_count}_parc-{parc_name}_hemi-L.hdf5", 'w') as f:
        f.create_dataset('sc', data=sc_avg)
        f.create_dataset('subj_ids', data=subj_ids)
