import os
import h5py
import pandas as pd
import nibabel as nib
import numpy as np
from brainspace.utils.parcellation import reduce_by_labels
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from pathlib import Path
from scipy.io import loadmat

load_dotenv()
PROJ_DIR = os.getenv("PROJ_DIR")

nsubj_desired = 384
parc_name = 'schaefer400'
try:
    parc_file = f"{PROJ_DIR}/data/parcellations/parc-{parc_name}_space-fsLR_den-32k_hemi-L.label.gii"
    parc = nib.load(parc_file).darrays[0].data.astype(int)
except:
    raise ValueError(f"Parcellation '{parc_name}' with den '32k' not found.")
medmask = np.where(parc != 0, True, False)

n_parcels = len(np.unique(parc[medmask]))

# Only keep subjects that have both FC and SC data
hcp_data = pd.read_csv('/fs03/kg98/vbarnes/HCP/restricted_1110_unrelated-445.csv', index_col=False)
hcp_data = hcp_data[hcp_data['has_fc'] & hcp_data['has_sc']].reindex()
subj_ids = hcp_data['Subject'].values

# Load bold data as matrix
scaler = StandardScaler()
bold_all = []
sc_all = []
subj_count = 0
for i, subj in enumerate(subj_ids):
    if subj_count == nsubj_desired:
        break

    print(f"Processing subject {subj} ({i+1}/{len(subj_ids)})")

    ### Load BOLD data ###
    bold_folder = f"/fs03/kg98/vbarnes/HCP/{subj}/MNINonLinear/Results/rfMRI_REST1_LR"
    bold_file = f"rfMRI_REST1_LR_Atlas_hp2000_clean.L.func.gii"
    bold = np.array(nib.load(Path(bold_folder, bold_file)).agg_data()).T
    # Check that shape is valid
    if np.shape(bold) != (32492, 1200):
        print(f"Skipping subject {subj} due to invalid shape")
        # Drop sub from sub_ids
        subj_ids = np.delete(subj_ids, i)
        continue
    # Process BOLD: 1) parcellate, 2) z-score
    bold_parc = reduce_by_labels(bold[medmask, :], parc[medmask], axis=1)
    bold_parc = scaler.fit_transform(bold_parc.T).T
    bold_all.append(bold_parc)

    ### Load SC data ###
    sc_folder = f"/fs03/kg98/vbarnes/HCP/{subj}/MNINonLinear/Results/connectome"
    sc_file = f"parc-HCPMMP1ANDfslatlas20acpc_tractalgo-iFOD2_sifttype-SIFT2_networktype-standard.mat"
    sc_all.append(loadmat(Path(sc_folder, sc_file))['connectome_subj'][:180, :180])

    subj_count += 1

bold_all = np.dstack(bold_all)
sc_all = np.dstack(sc_all)

sc_avg = np.nanmean(sc_all, axis=2)

with h5py.File(f"{PROJ_DIR}/data/empirical/HCP_unrelated-445_rfMRI_hemi-L_nsubj-{subj_count}_parc-{parc_name}_hemi-L_BOLD.hdf5", 'w') as f:
    f.create_dataset('bold', data=bold_all)
    f.create_dataset('subj_ids', data=subj_ids)

# with h5py.File(f"{PROJ_DIR}/data/empirical/HCP_unrelated-445_SC_hemi-L_nsubj-{subj_count}_parc-{parc_name}_hemi-L_SCavg.hdf5", 'w') as f:
#     f.create_dataset('sc', data=sc_avg)
#     f.create_dataset('subj_ids', data=subj_ids)
