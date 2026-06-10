#%%
import numpy as np
import nibabel as nib
from heteromodes.utils import get_project_root
from nsbutils.plotting_pyvista import plot_surf
import matplotlib.pyplot as plt
from neuromodes.io import read_surf
from nsbutils.utils import unmask
from neuromaps.images import construct_shape_gii

PROJ_DIR = get_project_root()
data_dir = "/fs03/kg98/vbarnes/hcp-ep/"

medmask = nib.load(
    f"{PROJ_DIR}/data/empirical/human/hcp-ep_space-fsLR_den-4k_hemi-L_desc-nomedialwall.func.gii"
).darrays[0].data.astype(bool)

hc_ids = np.loadtxt(PROJ_DIR / "data" / "empirical" / "human" / "hcp-ep-hc-run-1-2_nsubj-45_desc-subjects.txt").astype(int)
scz_ids = np.loadtxt(PROJ_DIR / "data" / "empirical" / "human" / "hcp-ep-scz-run-1-2_nsubj-74_desc-subjects.txt").astype(int)

print(len(hc_ids), len(scz_ids))

#%% Load one map at a time and compute ongoing average

for i, cohort in enumerate(["hc", "scz"]):
    if cohort == "hc":
        cohort_ids = hc_ids
    else:
        cohort_ids = scz_ids

    all_maps = []
    for sub_id in cohort_ids:
        data_dir = f"/fs03/kg98/vbarnes/hcp-ep/sub-{sub_id}/MNINonLinear/Results"
        map_path = f"{data_dir}/space-fsLR_den-4k_hemi-L_desc-myelinmap-smooth-bc.func.gii"
        myelin_map = nib.load(map_path).darrays[0].data
        all_maps.append(myelin_map[medmask])
    
    avg_map = np.mean(all_maps, axis=0)
    gii = construct_shape_gii(data=unmask(avg_map, medmask))
    nib.save(
        gii, PROJ_DIR / f"data/heteromaps/human/source-hcpep_desc-{cohort}-myelinmap_space-fsLR_den-4k_hemi-L.func.gii"
    )

#%% load computed maps and plot

t1t2_hc = nib.load(
    PROJ_DIR / "data/heteromaps/human/source-hcpep_desc-hc-myelinmap_space-fsLR_den-4k_hemi-L.func.gii"
).darrays[0].data
t1t2_scz = nib.load(
    PROJ_DIR / "data/heteromaps/human/source-hcpep_desc-scz-myelinmap_space-fsLR_den-4k_hemi-L.func.gii"
).darrays[0].data
diff = t1t2_scz - t1t2_hc

maps = np.stack([t1t2_hc, t1t2_scz], axis=0)

#%%


surf = read_surf(
    f"{PROJ_DIR}/data/empirical/human/space-fsLR_den-4k_hemi-L_desc-midthickness.surf.gii"
)
surf = {"vertices": surf.vertices, "faces": surf.faces}


fig, axs = plt.subplots(1, 2, figsize=(15, 5))
cohort_maps = np.stack([t1t2_hc, t1t2_scz], axis=0)
vmin = np.nanmin(cohort_maps)
vmax = np.nanmax(cohort_maps)
for i, map in enumerate(maps):
    plot_surf(
        surf={"lh": surf},
        data={"lh": map},
        rois={"lh": medmask},
        cmap="turbo", 
        cbar=True, 
        ax=axs[i],
        views=["lateral", "medial"],
        clim=(vmin, vmax),
    )
plt.show()


#%% plot correlation of maps

plt.scatter(t1t2_hc, t1t2_scz, alpha=0.5)
plt.xlabel("HC Average T1w/T2w")
plt.ylabel("SCZ Average T1w/T2w")
plt.title("Correlation of Average T1w/T2w Maps")

r = np.corrcoef(t1t2_hc[medmask], t1t2_scz[medmask])[0, 1]
plt.text(0.05, 0.95, f"r = {r:.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.show()


#%%

vmin = -np.max(np.abs(diff[medmask]))
vmax = np.max(np.abs(diff[medmask]))
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
plot_surf(
    surf={"lh": surf},
    data={"lh": diff},
    rois={"lh": medmask},
    cmap="seismic",
    cbar=True,
    ax=ax,
    views=["lateral", "medial"],
    clim=(vmin, vmax),
)
plt.show()
