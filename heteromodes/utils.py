import os
import numpy as np
import nibabel as nib
from pathlib import Path
from dotenv import load_dotenv
from neuromaps.transforms import fslr_to_fslr
from neuromaps.datasets import fetch_atlas


load_dotenv()
PROJ_DIR = os.getenv("PROJ_DIR")

def check_orthogonal(matrix, tol=0.1):
    """
    Check if a matrix is orthogonal.

    Parameters
    ----------
    matrix : array_like
        The matrix to be checked for orthogonality.
    tol : float, optional
        The tolerance value for comparing diagonal elements to 1.0. Default is 0.1.

    Returns
    -------
    bool
        True if the matrix is orthogonal, False otherwise.
    """
    # Convert the matrix to a numpy array
    matrix = np.array(matrix)
    dot_product = np.dot(matrix.T, matrix)
    
    # Check if the diagonal elements are close to 1
    if not np.allclose(np.diagonal(dot_product), 1.0, atol=tol):
        return False
    
    return True


def check_normal(matrix, axis=0, tol=0.01):
    """
    Check if the columns of a matrix are normalized.

    Parameters
    ----------
    matrix : array_like
        The input matrix.
    axis : int, optional
        The axis along which to calculate the norm. By default, axis=0.
    tol : float, optional
        The tolerance for comparing the column norms to 1. By default, tol=0.01.

    Returns
    -------
    bool
        True if all column norms are close to 1 within the given tolerance, False otherwise.
    """
    # Convert the matrix to a numpy array
    matrix = np.array(matrix)
    
    # Calculate the norm of each column
    column_norms = np.linalg.norm(matrix, axis=axis)
    
    # Check if all column norms are close to 1
    if not np.allclose(column_norms, 1.0, atol=tol):
        return False
    
    return True


def standardise_modes(emodes):
    """
    Perform standardisation by flipping the modes such that the first element of each mode is 
    positive.

    Parameters
    ----------
    emodes : numpy.ndarray
        The input array containing the modes.

    Returns
    -------
    numpy.ndarray
        The standardized modes with the first element of each mode set to be positive.
    """
    # Find the sign of the first non-zero element in each column
    signs = np.sign(emodes[np.argmax(emodes != 0, axis=0), np.arange(emodes.shape[1])])
    
    # Apply the sign to the modes
    standardized_modes = emodes * signs
    
    return standardized_modes


def load_hmap(hmap_label, den="32k"):
    """Load heterogeneity map.

    Parameters
    ----------
    hmap_label : _type_
        _description_
    den : str, optional
        _description_, by default "32k"

    Returns
    -------
    _type_
        _description_
    """
    hmap_file = list(Path(PROJ_DIR, "data", "heteromaps").glob(f"*desc-{hmap_label}_*den-32k_*.func.gii"))
    medial = nib.load(fetch_atlas("fsLR", den)["medial"][0]).darrays[0].data.astype(bool)
    
    # Load the heterogeneity map and transform it to the desired density if necessary
    if den == "32k":
        hmap = nib.load(hmap_file[0]).darrays[0].data
    else:
        hmap = fslr_to_fslr(hmap_file[0], den, hemi="L")[0].darrays[0].data

    # hmap[~medial] = np.nan

    return hmap


def unmask(data, medmask, val=np.nan):
    medmask = medmask.astype(bool)

    if data.ndim == 1:
        nverts = len(medmask)
        map_reshaped = np.full(nverts, val)
        map_reshaped[medmask] = data
    elif data.ndim == 2:
        nverts = len(medmask)
        nfeatures = np.shape(data)[1]
        map_reshaped = np.full((nverts, nfeatures), val)
        map_reshaped[medmask, :] = data

    return map_reshaped

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