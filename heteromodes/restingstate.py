import fbpca
import numpy as np
from scipy.linalg import norm
from scipy.signal import butter, filtfilt, detrend, hilbert
from scipy.fft import fft, ifft
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import label
from sklearn.preprocessing import StandardScaler
from heteromodes.models import WaveModel, BalloonModel


def simulate_bold(evals, emodes, ext_input, solver_method, eig_method, r=28.9, gamma=0.116, mass=None, tstep=0.09 * 1e3, nsteps=1200):
    """
    Function to simulate resting-state fMRI BOLD data using the NFT wave model and the Balloon
    haemodynamic model.
    """
    # Set simulation parameters
    tr = 0.72 * 1e3 # HCP data TR in ms
    tpre = 50 * 1e3 # burn time to remove transient
    tmax = tpre + (nsteps - 1) * tr # match number of timepoints in empirical data

    # Wave model to simulate neural activity
    wave = WaveModel(emodes, evals, r=r, gamma=gamma, tstep=tstep, tmax=tmax)
    ntsteps_tr = int(tr // wave.tstep)
    tsteady_ind = np.abs(wave.t - tpre).argmin()

    # Balloon model to simulate BOLD activity
    balloon = BalloonModel(emodes, tstep=tstep, tmax=tmax)

    # Simulate neural activity
    _, neural_activity = wave.solve(ext_input, solver_method, eig_method, mass=mass)
    
    # Simulate BOLD activity
    _, bold_activity = balloon.solve(neural_activity, solver_method, eig_method, mass=mass)

    # Extract BOLD activity to match empirical data timepoints
    bold_activity_tr = bold_activity[:, tsteady_ind::ntsteps_tr]

    return bold_activity_tr

def filter_bold(bold, tr, lowcut=0.01, highcut=0.1):
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

def generalized_phase_vector(x, Fs, lp):
    """
    GENERALIZED PHASE VECTOR calculate the generalized phase of input vector

    INPUT:
    x - data column vector
    Fs - sampling rate (Hz)
    lp - low-frequency data cutoff (Hz)

    OUTPUT:
    xgp - output datacube
    wt - instantaneous frequency estimate
    """

    # parameters
    nwin = 3

    # handle input
    npts = len(x)

    # anonymous functions
    rewrap = lambda xp: (xp - 2 * np.pi * np.floor((xp - np.pi) / (2 * np.pi)) - 2 * np.pi)
    def naninterp(xp):
        # Find indices where xp is not NaN and where it is NaN
        not_nan = np.where(~np.isnan(xp))[0]
        nan_indices = np.where(np.isnan(xp))[0]
        
        # Interpolate at the NaN positions using 'pchip' interpolation
        interpolator = PchipInterpolator(not_nan, xp[not_nan], extrapolate=True)
        xp[nan_indices] = interpolator(nan_indices)
        
        return xp

    # init
    dt = 1 / Fs

    # analytic signal representation (single-sided Fourier approach, cf. Marple 1999)
    xo = fft(x, npts)
    h = np.zeros(npts)
    if npts > 0 and npts % 2 == 0:
        h[[0, npts // 2 + 1]] = 1
        h[1:npts // 2 + 1] = 2
    else:
        h[0] = 1
        h[1:(npts + 1) // 2 + 1] = 2
    xo = ifft(xo * h)

    ph = np.angle(xo)
    md = np.abs(xo)

    # calculate IF
    wt = np.zeros_like(xo)
    wt[:-1] = np.angle(xo[1:] * np.conj(xo[:-1])) / (2 * np.pi * dt)

    # account for sign of IF
    sign_if = np.sign(np.nanmean(wt))
    if sign_if == -1:
        modulus = np.abs(xo)
        ang = sign_if * np.angle(xo)
        xo = modulus * np.exp(1j * ang)
        ph = np.angle(xo)
        md = np.abs(xo)
        wt[:-1] = np.angle(xo[1:] * np.conj(xo[:-1])) / (2 * np.pi * dt)

    # check if nan channel
    if np.all(np.isnan(ph)):
        return np.full_like(xo, np.nan), wt

    # find negative frequency epochs (i.e. less than LP cutoff)
    idx = wt < lp
    idx[0] = False
    L, G = label(idx)
    for kk in range(1, G + 1):
        idxs = np.where(L == kk)[0]  # Find indices of the current labeled region
        start = idxs[0]              # Start index of the current region
        end = start + int((idxs[-1] - start) * nwin) # End index (not inclusive)
        idx[start:end+1] = True        # Set elements in the calculated range to True

    # "stitch over" negative frequency epochs
    p = ph.copy()
    p[idx] = np.nan
    if np.all(np.isnan(p)):
        return np.full_like(xo, np.nan), wt  # check if all NaNs

    # Only unwrap the non-NaN values
    non_nan_indices = ~np.isnan(p)
    p[non_nan_indices] = np.unwrap(p[non_nan_indices])

    p = naninterp(p)
    p = rewrap(p)

    # output
    xgp = md * np.exp(1j * p)

    return xgp, wt

def calc_edgefc_corr(model_fc, emp_fc, eps=1e-7):
    triu_inds = np.triu_indices(np.shape(model_fc)[0], k=1)
    edge_fc_corr = np.corrcoef(np.arctanh(np.clip(model_fc[triu_inds], -1 + eps, 1 - eps)), 
                               np.arctanh(np.clip(emp_fc[triu_inds], -1 + eps, 1 - eps)))[0, 1]

    return edge_fc_corr

def calc_nodefc_corr(model_fc, emp_fc, eps=1e-7):
    # Compute Node-level FC (exclude diagonal elements by setting them to NaN)
    fc_model_nandiag = np.clip(model_fc, -1 + eps, 1 - eps)
    np.fill_diagonal(fc_model_nandiag, np.nan)
    fc_emp_nandiag = np.clip(emp_fc, -1 + eps, 1 - eps)
    np.fill_diagonal(fc_emp_nandiag, np.nan)
    node_fc_corr = np.corrcoef(np.nanmean(np.arctanh(fc_model_nandiag), axis=1), 
                               np.nanmean(np.arctanh(fc_emp_nandiag), axis=1))[0, 1]
    
    return node_fc_corr

def calc_gen_phase(bold, tr=0.72, lowcut=0.01, highcut=0.1):
    # Calculate phase delay
    phase_delay = np.empty(bold.shape, dtype=np.complex128)
    for i in range(bold.shape[0]):
        phase_delay[i, :] = generalized_phase_vector(bold[i, :], Fs=1/0.72, lp=lowcut)[0].conj()

    return np.angle(phase_delay)

def calc_phase_map(phase, n_components=4):
    # Compute SVD
    l = 10 + n_components
    U, s, V = fbpca.pca(phase, k=n_components, n_iter=20, l=l)

    # Calculate weighted sum of first n principal components
    combined_phase_map = np.sum(np.real(V).T @ np.diag(s), axis=1) / np.sum(s)
    # combined_phase_map = np.mean(U @ np.diag(s) @ V, axis=1)

    # return combined_phase_map
    return combined_phase_map

def calc_phase_cpcs(phase, n_components=4):
    """Calculate the complex principal components (CPCs) of the phase data.

    This function computes the complex principal components (CPCs) of the input phase data
    using Singular Value Decomposition (SVD).

    Parameters
    ----------
    phase : numpy.ndarray
        The phase data of shape (T, N), where T is the number of time points and N is the number of regions.
    n_components : int, optional
        The number of principal components to compute, by default 3.

    Returns
    -------
    numpy.ndarray
        The complex principal components (CPCs) shape (N, n_components).
    numpy.ndarray
        The singular values corresponding to the principal components.
    """

    # Compute SVD
    l = 10 + n_components
    U, s, V = fbpca.pca(phase, k=n_components, n_iter=20, l=l)
    # combined_phase_map = np.sum(np.real(V.T) * s, axis=1) / np.sum(s)

    return np.real(V).T, s


def calc_fcd(bold, tr=0.72, lowcut=0.04, highcut=0.07, n_avg=3):
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
    bold_filtered = filter_bold(bold, tr=tr, lowcut=lowcut, highcut=highcut)
    # Calculate phase for each region
    phase_bold = np.angle(hilbert(bold_filtered))

    # Remove first 9 and last 9 time points to avoid edge effects from filtering, as the bandpass 
    # filter may introduce distortions near the boundaries of the time series.
    t_trunc = np.arange(9, t - 9)

    # Calculate synchrony
    triu_inds = np.triu_indices(n_regions, k=1)
    nt = len(t_trunc)
    synchrony_vecs = np.zeros((nt, len(triu_inds[0])))
    for t_ind, t in enumerate(t_trunc):
        phase_diff = np.subtract.outer(phase_bold[:, t], phase_bold[:, t])
        synchrony_mat = np.cos(phase_diff)
        synchrony_vecs[t_ind, :] = synchrony_mat[triu_inds]

    # Pre-calculate phase vectors
    p_mat = np.zeros((nt - n_avg-1, synchrony_vecs.shape[1]))
    for t_ind in range(nt - n_avg-1):
        p_mat[t_ind, :] = np.mean(synchrony_vecs[t_ind : t_ind+n_avg, :], axis=0)
        p_mat[t_ind, :] = p_mat[t_ind, :] / norm(p_mat[t_ind, :])

    # Calculate phase for every time pair
    fcd_mat = p_mat @ p_mat.T

    triu_ind = np.triu_indices(fcd_mat.shape[0], k=1)
    fcd = fcd_mat[triu_ind]

    return fcd
