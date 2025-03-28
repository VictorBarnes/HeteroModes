import os
import re
import numpy as np
import nibabel as nib
from pathlib import Path
from dotenv import load_dotenv
from neuromaps.transforms import fslr_to_fslr


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

def load_hmap(hmap_label, trg_den="32k"):
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

    hmap_file = list(Path(PROJ_DIR, "data", "heteromaps").glob(f"*desc-{hmap_label}_*.func.gii"))
    if len(hmap_file) == 0:
        raise FileNotFoundError(f"No heterogeneity map found for label '{hmap_label}'.")

    # Extract the source density from the file name    
    src_den = re.search(r'den-(.*?)_', str(hmap_file)).group(1)

    # Load the heterogeneity map and transform it to the desired density if necessary
    if src_den == trg_den:
        hmap = nib.load(hmap_file[0]).darrays[0].data
    else:
        hmap = fslr_to_fslr(hmap_file[0], trg_den, hemi="L")[0].darrays[0].data

    # hmap[~medial] = np.nan

    return hmap

def unmask(data, medmask, val=np.nan):
    """
    Unmasks data by inserting it into a full array with the same length as the medial mask.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be unmasked (n_verts, n_data). Can be 1D or 2D.
    medmask : numpy.ndarray
        A boolean array where True indicates the positions of the data in the full array.
    val : float, optional
        The value to fill in the positions outside the mask. Default is np.nan.

    Returns
    -------
    numpy.ndarray
        The unmasked data, with the same shape as the medial mask.
    """
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

def unparcellate(data, parc):
    return np.array([data[x-1] for x in parc if x > 0])
