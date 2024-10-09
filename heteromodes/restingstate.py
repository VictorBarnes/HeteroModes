import numpy as np
from scipy.stats import ks_2samp
from scipy.linalg import norm
from scipy.signal import butter, filtfilt, detrend, hilbert
from sklearn.preprocessing import StandardScaler
from heteromodes.models import WaveModel, BalloonModel

def simulate_bold(evals, emodes, ext_input, solver_method, eig_method, r=28.9, gamma=0.116, B=None, tstep=0.09 * 1e3):
    """
    Function that simulates resting-state fMRI BOLD data.
    """
    # Set simulation parameters
    tr = 0.72 * 1e3 # HCP data TR in ms
    tpre = 50 * 1e3 # burn time to remove transient
    tmax = tpre + 1199 * tr # match number of timepoints in empirical data

    # Wave model to simulate neural activity
    wave = WaveModel(emodes, evals, r=r, gamma=gamma, tstep=tstep, tmax=tmax)
    ntsteps_tr = int(tr // wave.tstep)
    tsteady_ind = np.abs(wave.t - tpre).argmin()

    # Balloon model to simulate BOLD activity
    balloon = BalloonModel(emodes, tstep=tstep, tmax=tmax)

    # Simulate neural activity
    _, neural_activity = wave.solve(ext_input, solver_method, eig_method, B=B)
    
    # Simulate BOLD activity
    _, bold_activity = balloon.solve(neural_activity, solver_method, eig_method, B=B)

    return bold_activity[:, tsteady_ind::ntsteps_tr]

def filter_bold(bold, tr, lowcut=0.04, highcut=0.07):
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

def calc_phase_fcd(bold, tr=0.72, n_avg=3):
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
    # Ensure n_avg > 0
    if n_avg < 1:
        raise ValueError("n_avg must be greater than 0")

    n_regions, t = np.shape(bold)  
    # Bandpass filter the BOLD signal
    bold_filtered = filter_bold(bold, tr=tr)
    # Calculate phase for each region
    phase_bold = np.angle(hilbert(bold_filtered))

    # Remove first 9 and last 9 time points to avoid edge effects from filtering, as the bandpass 
    # filter may introduce distortions near the boundaries of the time series.
    t_trunc = np.arange(9, t - 9)  

    # Calculate synchrony
    tril_ind = np.tril_indices(n_regions, -1)
    nt = len(t_trunc)
    synchrony_vec = np.zeros((nt, len(tril_ind[0])))
    for t_ind, t in enumerate(t_trunc):
        phase_diff = phase_bold[:, t][:, None] - phase_bold[:, t]
        synchrony_mat = np.cos(phase_diff)
        synchrony_vec[t_ind, :] = synchrony_mat[tril_ind]

    # Pre-calculate phase vectors
    p_mat = np.zeros((nt - n_avg-1, synchrony_vec.shape[1]))
    for t_ind in range(nt - n_avg-1):
        p_mat[t_ind, :] = np.mean(synchrony_vec[t_ind : t_ind+n_avg, :], axis=0)
        p_mat[t_ind, :] = p_mat[t_ind, :] / norm(p_mat[t_ind, :])

    # Calculate phase for every time pair
    fcd_mat = p_mat @ p_mat.T

    triu_ind = np.triu_indices(fcd_mat.shape[0], k=1)
    fcd = fcd_mat[triu_ind]

    return fcd

def calc_fc_fcd(bold, tr, band_freq=(0.04, 0.07)):
    # Ensure data is standardised
    if not np.isclose(np.mean(bold, axis=1), 0).all() or not np.isclose(np.std(bold, axis=1), 1.0).all():
        scaler = StandardScaler()
        bold = scaler.fit_transform(bold.T).T
    # Bandpass filter the data
    if band_freq is None:
        bold = bold
    elif len(band_freq) == 2:
        bold = filter_bold(bold, tr=tr, lowcut=band_freq[0], highcut=band_freq[1])
    else:
        raise ValueError("Filter must be a tuple of length 2")
    
    # Caculate FC and FCD
    fc = np.corrcoef(bold)
    fcd = calc_phase_fcd(bold, tr=tr, n_avg=10)

    return fc, fcd

def evaluate_model(empirical, model):
    """
    Evaluate accuracy of model by calculating functional connectivity metrics on BOLD data.

    Notes
    -----
    The empirical and model BOLD data should have the same dimensions. 
    """

    fc_emp = empirical["fc"]
    fc_model = model["fc"]
    fcd_emp = empirical["fcd"]
    fcd_model = model["fcd"]

    if np.shape(fc_emp)[0] != np.shape(fc_model)[0]:
        raise ValueError("Empirical and model data do not have the same number of regions")
    nparcels = np.shape(fc_emp)[0]

    triu_inds = np.triu_indices(nparcels, k=1)

    # Compute Edge-level FC
    edge_fc = np.corrcoef(np.arctanh(fc_emp[triu_inds]), np.arctanh(fc_model[triu_inds]))[0, 1]

    # Compute Node-level FC (exclude diagonal elements by setting them to NaN)
    fc_model_run_nandiag = np.copy(fc_model)
    np.fill_diagonal(fc_model_run_nandiag, np.nan)
    fc_emp_nandiag = np.copy(fc_emp)
    np.fill_diagonal(fc_emp_nandiag, np.nan)
    node_fc = np.corrcoef(np.nanmean(np.arctanh(fc_model_run_nandiag), axis=1), 
                          np.nanmean(np.arctanh(fc_emp_nandiag), axis=1))[0, 1]

    # Calculate FCD of model BOLD data
    fcd_ks = ks_2samp(np.hstack(fcd_emp), np.hstack(fcd_model))[0]

    return edge_fc, node_fc, fcd_ks
