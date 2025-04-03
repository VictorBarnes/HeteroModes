import os
import fbpca
import numpy as np
from scipy.linalg import norm
from scipy.signal import butter, filtfilt, detrend, hilbert
from scipy.fft import fft, ifft
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import label
from scipy.stats import zscore
from heteromodes.solver import EigenSolver

DENSITIES = {3619: "4k", 29696: "32k"}

def run_simulation(
    # Solver parameters
    surf, 
    medmask=None, 
    hetero=None, 
    alpha=0, 
    r=28.9, 
    gamma=0.116, 
    scaling="sigmoid", 
    q_norm=None, 
    lump=False, 
    smoothit=10,
    nmodes=500, 
    # Simulation parameters
    nruns=5, 
    dt_emp=0.72 * 1e3, 
    nt_emp=1200, 
    dt=0.1, 
    tsteady=0, 
    solver_method="Fourier", 
    eig_method="orthogonal", 
    # Phase parameters
    phase_type=None,
    band_freq=(0.01, 0.1), 
    n_comps=4
):
    """
    Runs a neural field model simulation and computes functional connectivity (FC) and phase maps.

    Solver Parameters:
    ------------------
    surf : Surface object
        The cortical surface for the simulation.
    medmask : array-like or None
        Binary mask for medial wall regions.
    hetero : float or None
        Level of heterogeneity in the model.
    alpha : float
        Scaling parameter for wave propagation (default: 0).
    r : float
        Global coupling strength (default: 28.9).
    gamma : float
        Decay parameter for interactions (default: 0.116).
    scaling : str
        Type of scaling function ("sigmoid", "linear", etc.).
    q_norm : float or None
        Normalization factor for activity.
    lump : bool
        Whether to lump parameters together.
    smoothit : int
        Number of smoothing iterations (default: 10).
    nmodes : int, optional
        Number of eigenmodes to use in the solver (default: 500).

    Simulation Parameters:
    ----------------------
    nruns : int, optional
        Number of simulation runs to average over (default: 5).
    dt_emp : float, optional
        Empirical time step in milliseconds (default: 720 ms).
    nt_emp : int, optional
        Number of empirical timepoints (default: 1200).
    dt : float
        Simulation time step in milliseconds (default: 0.1 ms).
    tsteady : float
        Time before starting data collection.
    solver_method : str
        Method for solving the simulation (default: "Fourier").
    eig_method : str
        Method for eigenmode decomposition (default: "orthogonal").

    Phase Parameters:
    -----------------
    phase_type : str, optional
        Type of phase map to compute. Options: {"cpc1", "combined"} (default: None).
    band_freq : tuple of (float, float)
        Frequency band for Hilbert transform (default: (0.01, 0.1)).
    n_comps : int
        Number of principal components to use for phase calculation (default: 4).

    Returns:
    --------
    fc : np.ndarray
        Functional connectivity matrix.
    phase_map : np.ndarray or None
        Phase map of the first complex principal component (CPC1) if requested.
    """
    
    # Try solving eigenvalues and eigenvectors
    try:
        solver = EigenSolver(surf=surf, medmask=medmask, hetero=hetero, alpha=alpha, 
                             r=r, gamma=gamma, scaling=scaling, q_norm=q_norm, 
                             lump=lump, smoothit=smoothit)
        solver.solve(k=nmodes, fix_mode1=True, standardise=False)
    except ValueError as e:
        if "Alpha value results in non-physiological wave speeds" in str(e):
            print(f"Invalid parameter combination: alpha={alpha}, r={r}, gamma={gamma}")
            return None, None
        else:
            raise e

    nt = int((nt_emp - 1) * dt_emp / dt)

    # Pre-allocate BOLD activity array
    bold = np.empty((solver.surf.n_points, nt_emp * nruns), dtype=np.float32)
    for run in range(nruns):
        bold_run = solver.simulate_bold(dt=dt, nt=nt, tsteady=tsteady, solver_method=solver_method, 
                                        eig_method=eig_method, seed=run)

        # Downsample to match empirical time resolution
        bold_run = bold_run[:, ::int(dt_emp // dt)]
        bold[:, run * nt_emp : (run + 1) * nt_emp] = zscore(bold_run, axis=1)

    # Calculate Functional Connectivity (FC)
    fc = solver.calc_fc(bold)

    # Compute phase map if required
    phase_map = None
    if phase_type in ["cpc1", "combined"]:
        fnq = 0.5 * (1 / (dt_emp / 1e3))  # Nyquist frequency

        # Apply Hilbert transform
        complex_data = solver.calc_hilbert(
            bold, fnq=fnq, band_freq=band_freq, conj=True
        )

        # PCA on complex data
        _, s, V = fbpca.pca(complex_data.T, k=10, n_iter=20, l=20)

        if phase_type == "cpc1":
            phase_map = np.real(V[0, :]).T
        elif phase_type == "combined":
            phase_map = np.sum(
                np.real(V[:n_comps, :]).T * s[:n_comps], axis=1
            ) / np.sum(s[:n_comps])

    return fc, phase_map

def evaluate_model(model_fc, model_phase_map, emp_fc, emp_phase_map, metrics):
    # Calculate evaluation metrics
    edge_fc_corr, node_fc_corr, phase_corr = 0, 0, 0
    if "edge_fc" in metrics:
        edge_fc_corr = calc_edgefc_corr(model_fc, emp_fc)
    if "node_fc" in metrics:
        node_fc_corr = calc_nodefc_corr(model_fc, emp_fc)
    if "phase" in metrics:
        phase_corr = np.abs(np.corrcoef(model_phase_map, emp_phase_map)[0, 1])

    return edge_fc_corr, node_fc_corr, phase_corr

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
