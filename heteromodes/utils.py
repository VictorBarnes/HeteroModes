import json
import numpy as np
import nibabel as nib
from pathlib import Path

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

    config_file = data_dir / "heteromap_labels.json"
    with open(config_file, "r") as f:
        hetero_file = json.load(f).get("heteromap_files", {}).get(hmap_label, None)

    if hetero_file is not None:
        hmap = nib.load(data_dir / hetero_file).darrays[0].data
    else:
        raise FileNotFoundError(f"No heterogeneity map file specified for label "
                                f"'{hmap_label}' in {config_file}.")
        
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
