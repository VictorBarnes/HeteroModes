# %%
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nsbtools.utils import unmask
from nsbtools.plotting import plot_brain
from heteromodes.utils import get_project_root


PROJ_DIR = get_project_root()

medmask = nib.load(f"{PROJ_DIR}/data/empirical/macaque/space-fsLR_den-4k_hemi-L_desc-nomedialwall.func.gii").darrays[0].data.astype(bool)
exh = nib.load(f"{PROJ_DIR}/data/heteromaps/macaque/desc-exh_space-fsLR_den-4k_hemi-L.func.gii").darrays[0].data[medmask]
inh = nib.load(f"{PROJ_DIR}/data/heteromaps/macaque/desc-inh_space-fsLR_den-4k_hemi-L.func.gii").darrays[0].data[medmask]

exh_scaled = (exh - exh.min()) / (exh.max() - exh.min()) * (2 - 1) + 1
inh_scaled = (inh - inh.min()) / (inh.max() - inh.min()) * (2 - 1) + 1

eiratio = exh_scaled / inh_scaled

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
surf = f"{PROJ_DIR}/data/empirical/macaque/space-fsLR_den-4k_hemi-L_desc-midthickness.surf.gii"
plot_brain(surf, unmask(exh_scaled, medmask), cmap="turbo", cbar=True, ax=axs[0])
plot_brain(surf, unmask(inh_scaled, medmask), cmap="turbo", cbar=True, ax=axs[1])
plot_brain(surf, unmask(eiratio, medmask), cmap="turbo", cbar=True, ax=axs[2])
axs[0].set_title("Excitatory receptor density (scaled)")
axs[1].set_title("Inhibitory receptor density (scaled)")
axs[2].set_title("E:I ratio")
plt.tight_layout()
plt.show()

# %% Save eiratio as gifti
from neuromaps.images import construct_shape_gii

eiratio_gii = construct_shape_gii(unmask(eiratio, medmask))
nib.save(eiratio_gii, f"{PROJ_DIR}/data/heteromaps/macaque/desc-eiratio_space-fsLR_den-4k_hemi-L.func.gii")