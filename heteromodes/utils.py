"""Utility functions for data loading and preprocessing."""

import io
from PIL import Image
import json
import numpy as np
import nibabel as nib
from pathlib import Path


def get_project_root() -> Path:
    """
    Find the project root directory by searching for markers.
    
    Searches parent directories for pyproject.toml or .git to identify
    the project root.
    
    Returns
    -------
    Path
        Absolute path to the project root directory.
        
    Raises
    ------
    RuntimeError
        If no project root markers are found in parent directories.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    raise RuntimeError("Project root not found. No pyproject.toml or .git found.")


def load_hmap(hmap_label, species="human", density="4k", data_dir=None):
    """
    Load a heterogeneity map from the data directory.

    Parameters
    ----------
    hmap_label : str
        Label identifying the heterogeneity map in heteromap_labels.json.
    species : str, default="human"
        Species identifier ("human", "macaque", etc.).
    density : str, default="4k"
        Surface density ("4k" or "32k").
    data_dir : str or Path, optional
        Custom data directory path. If None, uses default project structure
        (data/heteromaps/{species}/).

    Returns
    -------
    np.ndarray
        Heterogeneity map data as a 1D numpy array.

    Raises
    ------
    FileNotFoundError
        If the heterogeneity map label is not found in the configuration file
        or if the referenced file does not exist.
    ValueError
        If an invalid density value is provided.
    """
    # Validate density parameter
    valid_densities = ["4k", "32k"]
    if density not in valid_densities:
        raise ValueError(
            f"Invalid density '{density}'. Must be one of {valid_densities}."
        )
    
    # Use default data directory if not provided
    if data_dir is None:
        proj_dir = get_project_root()
        data_dir = proj_dir / "data" / "heteromaps" / species
    else:
        data_dir = Path(data_dir)

    # Load heterogeneity map file path from configuration
    config_file = data_dir / "heteromap_labels.json"
    with open(config_file, "r") as f:
        config = json.load(f)
    
    hetero_file = config.get("heteromap_files", {}).get(hmap_label)
    
    if hetero_file is None:
        raise FileNotFoundError(
            f"No heterogeneity map file specified for label '{hmap_label}' "
            f"in {config_file}."
        )
    
    # Replace density in filename
    hetero_file = hetero_file.replace("den-4k", f"den-{density}")
    
    # Load GIFTI file and extract data from first data array
    hmap = nib.load(data_dir / hetero_file).darrays[0].data
    
    return hmap


def pad_sequences(sequences, val=-1):
    """
    Pad variable-length sequences to uniform length.
    
    All sequences are padded to match the length of the longest sequence
    in the input list. Useful for creating rectangular arrays from ragged data.

    Parameters
    ----------
    sequences : list of array-like
        List of 1D sequences with potentially different lengths.
    val : scalar, default=-1
        Padding value to use for shorter sequences.

    Returns
    -------
    np.ndarray
        2D array with shape (n_sequences, max_length) containing all
        sequences padded to uniform length.
        
    Examples
    --------
    >>> sequences = [[1, 2], [3, 4, 5], [6]]
    >>> pad_sequences(sequences, val=0)
    array([[1, 2, 0],
           [3, 4, 5],
           [6, 0, 0]])
    """
    max_length = max(len(seq) for seq in sequences)
    
    padded_sequences = [
        np.pad(seq, (0, max_length - len(seq)), mode='constant', constant_values=val)
        if len(seq) < max_length else seq
        for seq in sequences
    ]

    return np.array(padded_sequences)

def fig_to_array(fig, dpi=None, pad_inches=0.0):
    """
    Convert a matplotlib figure to a numpy array.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The matplotlib figure to convert.
    dpi : float, optional
        Resolution in dots per inch. If None, uses the figure's dpi.
    pad_inches : float, default=0.0
        Amount of padding around the figure when saving.
    
    Returns
    -------
    np.ndarray
        RGB image array with shape (height, width, 3).
    """
    # Save figure to a bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=pad_inches)
    buf.seek(0)
    
    # Load image from buffer and convert to numpy array
    img = Image.open(buf)
    img_array = np.array(img)
    buf.close()
    
    return img_array
