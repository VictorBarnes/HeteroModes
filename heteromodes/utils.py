import os
import tempfile
import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
from scipy.signal import butter, filtfilt, detrend, hilbert
from scipy.linalg import norm
from scipy.special import erf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from surfplot import Plot
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

def filter_bold(bold, tr=0.72, lowcut=0.04, highcut=0.07):
    """Filter the BOLD signal using a bandpass filter.

    Parameters
    ----------
    bold : numpy.ndarray
        The BOLD signal data of shape (N, T), where N is the number of regions and T is the number of time points.
    tr : float
        The repetition time (TR) in seconds.
    lowcut : float
        The lowcut frequency for the bandpass filter.
    highcut : float
        The highcut frequency for the bandpass filter.

    Returns
    -------
    numpy.ndarray
        The filtered BOLD signal data.
    """
    # Define parameters
    k = 2  # 2nd order butterworth filter
    fnq = 0.5 * 1/tr  # nyquist frequency
    Wn = [lowcut/fnq, highcut/fnq]  # butterworth bandpass non-dimensional frequency
    bfilt2, afilt2 = butter(k, Wn, btype="bandpass")  # construct the filter

    # Standardise and detrend the data if it isn't already
    if not np.allclose(np.mean(bold, axis=1), 0) or not np.allclose(np.std(bold, axis=1), 1.0):
        scaler = StandardScaler()
        bold = scaler.fit_transform(bold.T).T
    bold = detrend(bold, axis=1, type='constant')
    # Apply the filter to the data
    bold_filtered = filtfilt(bfilt2, afilt2, bold, axis=1)

    return bold_filtered

def calc_phase_fcd(bold, tr=0.72):
    """Calculate phase-based functional connectivity dynamics (phFCD).

    This function calculates the phase-based functional connectivity dynamics (phFCD) 
    between regions of interest using the given bold signal and repetition time (TR).

    Parameters
    ----------
    bold : numpy.ndarray
        The bold signal data of shape (N, T), where N is the number of regions and T is the number of time points.
    tr : float
        The repetition time (TR) in seconds.

    Returns
    -------
    numpy.ndarray
        The phFCD matrix of shape (M,), where M is the number of unique pairs of regions.
    """

    # Define parameters
    # k = 2  # 2nd order butterworth filter
    # fnq = 0.5 * 1/tr  # nyquist frequency
    # flp = 0.04  # low-pass frequency of filter
    # fhi = 0.07  # high-pass frequency of the filter
    # Wn = [flp/fnq, fhi/fnq]  # butterworth bandpass non-dimensional frequency
    # bfilt2, afilt2 = butter(k, Wn, btype="bandpass")  # construct the filter

    # # Standardise and detrend the data if it isn't already
    # if not np.allclose(np.mean(bold, axis=1), 0) or not np.allclose(np.std(bold, axis=1), 1.0):
    #     scaler = StandardScaler()
    #     bold = scaler.fit_transform(bold.T).T
    # bold = detrend(bold, axis=1, type='constant')
    # Apply the filter to the data

    n_regions, t = np.shape(bold)  
    # Bandpass filter the BOLD signal
    bold_filtered = filter_bold(bold, tr=tr)
    # Calculate phase for each region
    phase_bold = np.angle(hilbert(bold_filtered))

    # Remove first 9 and last 9 time points to avoid edge effects from filtering, as the bandpass 
    # filter may introduce distortions near the boundaries of the time series. The cutoff is 
    # arbitrarily chosen
    t_trunc = np.arange(9, t - 9)  

    # Calculate synchrony
    tril_ind = np.tril_indices(n_regions, -1)
    nt = len(t_trunc)
    synchrony_vec = np.zeros((nt, len(tril_ind[0])))
    for t_ind, t in enumerate(t_trunc):
        phase_diff = phase_bold[:, t][:, None] - phase_bold[:, t]
        synchrony_mat = np.cos(phase_diff)
        synchrony_vec[t_ind, :] = synchrony_mat[tril_ind]
    # for t_ind in range(nt):
    #     t = t_trunc[t_ind]
    #     phase_bold_current = np.tile(phase_bold[:, t], (n_regions, 1))
    #     synchrony_mat = np.cos(phase_bold_current - phase_bold_current.T)
    #     synchrony_vec[t_ind, :] = synchrony_mat[tril_ind]

    # Calculate phase from synchrony at each time point
    p_mat = np.zeros((nt - 2, synchrony_vec.shape[1]))
    for t_ind in range(nt - 2):
        p_mat[t_ind, :] = np.mean(synchrony_vec[t_ind:t_ind + 3, :], axis=0)
        p_mat[t_ind, :] = p_mat[t_ind, :] / norm(p_mat[t_ind, :])

    # Calculate phase for every time pair
    fcd_mat = p_mat @ p_mat.T

    triu_ind = np.triu_indices(fcd_mat.shape[0], k=1)
    fcd = fcd_mat[triu_ind]

    return fcd

def scale_hmap(hmap, alpha=1.0, beta=1.0, r=28.9, gamma=0.116, sigma=0, method="zscore", verbose=True):
    """Scale the heterogeneity map using the given parameters.

    This function takes a heterogeneity map and scales it based on the provided parameters.
    The scaling is done by normalizing the map, applying a density transformation, and
    checking the range of the resulting values.

    Parameters
    ----------
    hetero : numpy.ndarray
        The heterogeneity map to be scaled.
    alpha : float
        The scaling factor for the density transformation.
    beta : float
        The exponent for the density transformation.
    cmean : float
        The mean wave speed value.
    method : str, optional
        The method to use for scaling the heterogeneity map. Options are "norm" for normalizing
        the map between 0 and 1, and "erf" for applying the error function. By default, 
        method="norm".

    Returns
    -------
    numpy.ndarray
        The scaled heterogeneity map.

    Raises
    ------
    ValueError
        If the density values are not positive.
    ValueError
        If the resulting wave speed values are not within the range [0.1, 150] m/s.
    """
    cmean = r * gamma
    
    if method in ["norm", "erf", "sigmoid"]:
        if method == "norm":
            # Normalize the heterogeneity map
            scaler = MinMaxScaler(feature_range=(0, 1))
            hmap_n = scaler.fit_transform(hmap.reshape(-1, 1)).flatten()
        elif method == "erf":
            hmap_lin = erf((hmap - np.nanmean(hmap)) / hmap.std() / np.sqrt(2))
            hmap_n = ((hmap_lin + 1) / 2)
        elif method == "sigmoid":
            hmap_lin = (hmap - np.nanmean(hmap)) / hmap.std()
            hmap_n = (1 / (1 + np.exp(-hmap_lin)))
        rho = sigma + alpha*(hmap_n - np.nanmean(hmap_n))
    elif method == "zscore":
        # Standardize the heterogeneity map
        scaler = StandardScaler()
        rho = sigma + alpha * scaler.fit_transform(hmap.reshape(-1, 1)).flatten()
    else:
        raise ValueError("Invalid scaling method. Options are 'norm', 'erf', 'sigmoid', or 'zscore'.")

    # Set lower bound on rho
    if sigma == 1:
        rho_min = 0.5/cmean
        rho_max = 150/cmean
        rho[rho < rho_min] = rho_min
        rho[rho > rho_max] = rho_max

    # if np.min(rho) < 0:
    #     raise ValueError("Density values must be positive")
    # rho = rho ** beta
    
    # c = cmean * rho # 1
    # c = cmean * np.sqrt(rho) # 2
    
    # if np.nanmin(c) < 0.49999 or np.nanmax(c) > 150:
    #     raise ValueError(f"cs values must be in the range [0.5, 150] m/s but the range is"
    #                         f"[{np.min(c):.1f}, {np.max(c):.1f}] m/s")
    
    # if verbose:
    #     print(f"cs range: {np.min(c):.2f} m/s to {np.max(c):.2f} m/s")

    return rho


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


def plot_brain(surf, data, labels=None, layout="row", views=["lateral", "medial"], clim_q=None, 
               cmap="viridis", cbar=False, cbar_label=None, cbar_kws=None, outline=False, dpi=100):
    """Plot multiple surfaces with associated data.

    Parameters
    ----------
    surf : object
        The brain surface object.
    data : numpy.ndarray (n_verts, n_data)
        The data to be plotted on the surfaces. Medial wall indices should be set to NaN.
    labels : list, optional
        The labels for each surface, by default None.
    layout : str, optional
        The layout of the subplots, either "row" or "col", by default "row".
    views : list, optional
        The views of the brain surfaces to be plotted, by default ["lateral", "medial"].
    clim_q : list, optional
        The percentiles for the color range of the data, by default [5, 95].

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the subplots.
    """

    # Set dpi
    mpl.rcParams['figure.dpi'] = dpi

    # Set default colorbar keyword arguments and update with user-specified values
    cbar_kws_default = dict(pad=0.01, fontsize=15, shrink=1, decimals=2)
    cbar_kws_default.update(cbar_kws or {})

    # Check if the data is 1D or 2D
    data = np.squeeze(data)
    if np.ndim(data) == 1 or np.shape(data)[1] == 1:
        data = data.reshape(-1, 1)
        fig = plt.figure(figsize=(len(views)*1.5, 2.5))
        axs = [plt.gca()]
    else:
        if layout == "row":
            fig, axs = plt.subplots(1, np.shape(data)[1], figsize=(len(views)*np.shape(data)[1]*1.5, 2))
        elif layout == "col":
            fig, axs = plt.subplots(np.shape(data)[1], 1, figsize=(3, np.shape(data)[1]*1.25))
        fig.subplots_adjust(wspace=0.01, hspace=0.01)
        axs = axs.flatten()

    # Create a temporary directory to save the figures
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, ax in enumerate(axs):

            # Plot brain surface
            p = Plot(surf_lh=surf, views=views, size=(500, 250), zoom=1.25)
            if clim_q is not None:
                color_range = [np.nanpercentile(data[:, i], clim_q[0]), np.nanpercentile(data[:, i], clim_q[1])]
            else:
                color_range = None
            p.add_layer(data=data[:, i], cmap=cmap, cbar=cbar, color_range=color_range)

            if outline:
                p.add_layer(data[:, i], as_outline=True, cmap="gray", cbar=False, color_range=(1, 2))

            # Save the plot as a temporary file
            temp_file = f"{temp_dir}/figure_{i}.png"
            fig = p.build(cbar_kws=cbar_kws_default)
            if cbar:
                # Plot cbar label underneath the cbar
                fig.get_axes()[1].set_xlabel(cbar_label, fontsize=cbar_kws_default["fontsize"], 
                                             labelpad=5)
            plt.close(fig)  # Close the figure to avoid automatically displaying it
            fig.savefig(temp_file, bbox_inches='tight', dpi=dpi)

            # Load the figure and plot it as a subplot
            ax.imshow(plt.imread(temp_file))
            # Remove axes and ticklabels
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            # Plot labels
            if labels is not None:
                if layout == "row":
                    ax.set_title(labels[i], pad=0)
                elif layout == "col":
                    ax.set_ylabel(labels[i], labelpad=0, rotation=0, ha="right")

    return fig

def load_parc(parc_file):
    # Get parc_file extension
    ext = parc_file.split('.')[-1]
    if ext == 'gii':
        parc = nib.load(parc_file).darrays[0].data
    elif ext == 'txt':
        parc = np.loadtxt(parc_file).astype(int)
    else:
        raise ValueError("Parcellation file must be a .gii or .txt file")
    
    return parc