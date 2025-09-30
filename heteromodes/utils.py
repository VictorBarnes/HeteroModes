import trimesh
import numpy as np
import nibabel as nib
from pathlib import Path
from nsbtools.eigen import check_surf


def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    raise RuntimeError("Project root not found.")

def load_hmap(hmap_label, species="human", trg_den="32k", data_dir=None):
    """Load heterogeneity map and transforms it to the target density.

    Parameters
    ----------
    hmap_label : str
        The label of the heterogeneity map to load.
    trg_den : str, optional
        The target density to which the map should be transformed, by default "32k".

    Returns
    -------
    numpy.ndarray
        The heterogeneity map data as a numpy array.

    Raises
    ------
    FileNotFoundError
        If no heterogeneity map is found for the given label.
    """

    # Check if data_dir is provided, otherwise use the default directory
    if data_dir is None:
        PROJ_DIR = get_project_root()
        data_dir = Path(PROJ_DIR, "data", "heteromaps", species)
    else:
        data_dir = Path(data_dir)

    matches = list((data_dir).glob(f"*desc-{hmap_label}*den-{trg_den}*.func.gii"))
    if matches:
        file = str(matches[0])
        hmap = nib.load(file).darrays[0].data
    else:
        raise FileNotFoundError(f"No heterogeneity map found for label '{hmap_label}' "
                                f" and density {trg_den}.")
    return hmap

def pad_sequences(sequences, val=-1):
    """
    Pads a list of sequences to the length of the longest sequence.
    """

    # Find the length of the longest sequence
    max_length = max(len(seq) for seq in sequences)
    # Pad sequences
    padded_sequences = [
        np.pad(seq, pad_width=(0, max_length - len(seq)), mode='constant', constant_values=val)
        if len(seq) < max_length else seq
        for seq in sequences
    ]

    return np.array(padded_sequences)

def intersect_medmasks(surf, medmask1, medmask2):
    """Find intersection of medmasks and remove unreferenced vertices."""
    surf = check_surf(surf)

    medmask1, medmask2 = np.asarray(medmask1, dtype=bool), np.asarray(medmask2, dtype=bool)
    medmask = np.logical_and(medmask1, medmask2)
    
    # Create masked mesh to find unreferenced vertices
    v_masked = surf.vertices[medmask]
    idx_map = np.full(len(medmask), -1); idx_map[medmask] = np.arange(np.sum(medmask))
    f_masked = idx_map[surf.faces[np.all(medmask[surf.faces], axis=1)]]
    
    mesh = trimesh.Trimesh(vertices=v_masked, faces=f_masked, process=False)
    referenced = np.zeros(len(mesh.vertices), dtype=bool); referenced[mesh.faces.flatten()] = True
    
    if not np.all(referenced):
        medmask[np.where(medmask)[0][~referenced]] = False
        
    return medmask
