#%%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from heteromodes.restingstate import calc_edge_fc
from heteromodes.utils import get_project_root
from nsbutils.plotting_pyvista import plot_surf
from neuromodes.io import read_surf
from nsbutils.utils import unmask


PROJ_DIR = get_project_root()
hc_fc_file = f"{PROJ_DIR}/data/hcp-ep/hcp-ep_diagnosis-hc_run-1-2_fc_space-fsLR_den-4k_hemi-L.h5"
scz_fc_file = f"{PROJ_DIR}/data/hcp-ep/hcp-ep_diagnosis-scz_run-1-2_fc_space-fsLR_den-4k_hemi-L.h5"

with h5py.File(hc_fc_file, "r") as f:
    hc_fc = f["fc_group"][:]
    medmask = f["medmask"][:].astype(bool)
with h5py.File(scz_fc_file, "r") as f:
    scz_fc = f["fc_group"][:]

print(hc_fc.shape)
print(scz_fc.shape)


#%% Plot average FC matrices

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
im0 = axs[0].imshow(hc_fc, vmin=-1, vmax=1, cmap="seismic")
axs[0].set_title("HC Average FC")
plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
im1 = axs[1].imshow(scz_fc, vmin=-1, vmax=1, cmap="seismic")
axs[1].set_title("SCZ Average FC")
plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
plt.show()

#%% Compute edge fc corr

hc_edge_fc = calc_edge_fc(hc_fc, fisher_z=True)
scz_edge_fc = calc_edge_fc(scz_fc, fisher_z=True)
r_edge = np.corrcoef(hc_edge_fc, scz_edge_fc)[0, 1]

plt.scatter(hc_edge_fc, scz_edge_fc, alpha=0.5)
plt.xlabel("HC Edge FC (Fisher Z)")
plt.ylabel("SCZ Edge FC (Fisher Z)")
plt.title(f"r_edge: r = {r_edge:.4f}")
plt.show()

#%%

diff = hc_fc - scz_fc

vmin = -np.max(np.abs(diff))
vmax = np.max(np.abs(diff))

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
im0 = ax.imshow(diff, vmin=vmin, vmax=vmax, cmap="seismic")
plt.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
ax.set_title("HC - SCZ FC Difference")
plt.show()

surf = read_surf(
    f"{PROJ_DIR}/data/empirical/human/space-fsLR_den-4k_hemi-L_desc-midthickness.surf.gii"
)
surf = {"vertices": surf.vertices, "faces": surf.faces}

node_diff = np.mean(diff, axis=0)
print(node_diff.shape)

#%%

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
vmin = -np.max(np.abs(node_diff))
vmax = np.max(np.abs(node_diff))
plot_surf(
    surf={"lh": surf},
    data={"lh": unmask(node_diff, medmask)},
    rois={"lh": medmask},
    cmap="seismic",
    cbar=True,
    ax=ax,
    views=["lateral", "medial"],
    clim=(vmin, vmax),
)
plt.show()
