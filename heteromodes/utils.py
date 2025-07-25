import os
import re
import numpy as np
import nibabel as nib
from pathlib import Path
from neuromaps.transforms import fslr_to_fslr


def load_hmap(hmap_label, trg_den="32k", data_dir=None):
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
        data_dir = Path(os.getenv("PROJ_DIR"), "data", "heteromaps")
    else:
        data_dir = Path(data_dir)

    hmap_file = list((data_dir).glob(f"*desc-{hmap_label}_*.func.gii"))
    if len(hmap_file) == 0:
        raise FileNotFoundError(f"No heterogeneity map found for label '{hmap_label}'.")

    # Extract the source density from the file name    
    src_den = re.search(r'den-(.*?)_', str(hmap_file)).group(1)

    # Load the heterogeneity map and transform it to the desired density if necessary
    if src_den == trg_den:
        hmap = nib.load(hmap_file[0]).darrays[0].data
    else:
        hmap = fslr_to_fslr(hmap_file[0], trg_den, hemi="L")[0].darrays[0].data

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

