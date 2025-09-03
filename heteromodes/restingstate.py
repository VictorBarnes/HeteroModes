import os
import fbpca
import numpy as np
from scipy.linalg import norm
from scipy.signal import butter, filtfilt, hilbert
from scipy.fft import fft, ifft
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import label
from scipy.stats import zscore, ks_2samp
from brainspace.utils.parcellation import reduce_by_labels
from nsbtools.utils import unmask
from nsbtools.eigen import EigenSolver


def run_model(
    # Solver parameters
    surf, 
    medmask=None, 
    hetero=None, 
    alpha=0, 
    beta=1.0,
    r=28.9, 
    gamma=0.116, 
    scaling="sigmoid", 
    q_norm=None, 
    lump=False, 
    smoothit=10,
    n_modes=500, 
    # Simulation parameters
    n_runs=5, 
    nt_emp=1200, 
    dt_emp=720, 
    dt_model=90, 
    tsteady=0, 
    eig_method="orthogonal", 
    parc=None
):
    """
    Runs a neural field model simulation and computes functional connectivity (FC) and phase maps.
    This function has been written with the intent of being parallelized which is why the 
    EigenSolver is called within the function instead of being passed as an argument.

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
    n_modes : int, optional
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
        solver = EigenSolver(surf=surf, medmask=medmask, hetero=hetero, n_modes=n_modes, alpha=alpha, beta=beta,
                             r=r, gamma=gamma, scaling=scaling, q_norm=q_norm, 
                             lump=lump, smoothit=smoothit)
        solver.solve(fix_mode1=True, standardise=False)
    except ValueError as e:
        if "Alpha value results in non-physiological wave speeds" in str(e):
            print(f"Invalid parameter combination: alpha={alpha}, r={r}, gamma={gamma}")
            return None, None
        else:
            raise e

    nt_model = int((nt_emp - 1) * dt_emp / dt_model)

    # Pre-allocate BOLD activity array
    n_regions = solver.surf.n_points if parc is None else len(np.unique(parc[medmask]))
    bold = np.empty((n_regions, nt_emp, n_runs), dtype=np.float32)
    for i in range(n_runs):
        bold_i = solver.simulate_waves(dt=dt_model, nt=nt_model, tsteady=tsteady, seed=i, bold_out=True,
                                         eig_method=eig_method).astype(np.float32)

        # Downsample to match empirical time resolution
        bold_i = bold_i[:, ::int(dt_emp // dt_model)]
        
        # Parcellate
        if parc is not None:
            bold_i = reduce_by_labels(bold_i, parc[medmask], axis=1)

        bold[:, :, i] = zscore(bold_i, axis=1).astype(np.float32)

    return bold

def analyze_bold(bold, dt_emp=720, band_freq=(0.01, 0.1), 
                 metrics=["edge_fc_corr", "node_fc_corr", "fcd_ks"]):
    """Compute all derivatives from BOLD data."""
    outputs = {}
    
    # Ensure input is in efficient dtype
    bold = bold.astype(np.float32, copy=False)
    
    # Calculate FC
    if "edge_fc_corr" in metrics or "node_fc_corr" in metrics:
        bold_concat = np.hstack([bold[:, :, i] for i in range(bold.shape[2])], dtype=np.float32)
        outputs['fc'] = calc_fc(bold_concat)
    
    if "cpc1" in metrics or 'fcd_ks' in metrics:
        fnq = 0.5 * (1 / (dt_emp / 1e3))   

        if "cpc1" in metrics:
            outputs['analytic'] = calc_hilbert(bold, fnq=fnq, band_freq=band_freq, k=2).astype(np.complex64)
        if "fcd_ks" in metrics:
            fcd = []
            for i in range(bold.shape[2]):
                fcd.append(calc_fcd_efficient(bold[:, :, i], fnq=fnq, band_freq=band_freq).astype(np.float32))
            outputs['fcd'] = np.array(fcd, dtype=np.float32).T

    return outputs


def evaluate_model(model_outputs, emp_outputs, 
                   metrics=["edge_fc_corr", "node_fc_corr", "fcd_ks"]):
    """
    Evaluate model against empirical data using pre-computed results.
    
    Parameters:
    -----------
    model_results : dict
        Dictionary containing model derivatives (fc, phase_map, fcd, etc.)
    emp_results : dict  
        Dictionary containing empirical derivatives (fc_mean, phase_map_mean, etc.)
    metrics : list
        List of metrics to compute results for

    Returns:
    --------
    results : dict
        Dictionary with correlation values for each metric
    """
    results = {}
    
    if "edge_fc_corr" in metrics:
        model_edge_fc = calc_edge_fc(model_outputs['fc'], fisher_z=True)
        emp_edge_fc = calc_edge_fc(emp_outputs['fc'], fisher_z=True)
        results['edge_fc_corr'] = np.corrcoef(model_edge_fc, emp_edge_fc)[0, 1]
    
    if "node_fc_corr" in metrics:
        model_node_fc = calc_node_fc(model_outputs['fc'])
        emp_node_fc = calc_node_fc(emp_outputs['fc'])
        results['node_fc_corr'] = np.corrcoef(model_node_fc, emp_node_fc)[0, 1]
    
    if "phase" in metrics:
        pass
    
    if "fcd_ks" in metrics:
        results['fcd_ks'] = ks_2samp(model_outputs['fcd'].flatten(), emp_outputs['fcd'].flatten())[0]
    
    return results

def filter_bold(bold, fnq, band_freq=(0.01, 0.1), k=2):
    """Filter the BOLD signal using a bandpass filter."""
    # Define parameters
    Wn = [band_freq[0]/fnq, band_freq[1]/fnq]
    bfilt2, afilt2 = butter(k, Wn, btype="bandpass")

    bold_z = zscore(bold, axis=1).astype(np.float32)
    bold_filtered = filtfilt(bfilt2, afilt2, bold_z, axis=1)

    return bold_filtered

def calc_fc(bold):
    bold_z = zscore(bold, axis=1).astype(np.float32)
    fc_matrix = np.corrcoef(bold_z, dtype=np.float32)

    return fc_matrix

def calc_hilbert(bold, fnq, band_freq=(0.01, 0.1), k=2):
    bold_filtered = filter_bold(bold, fnq, k=k, band_freq=band_freq)
    
    # Use scipy's hilbert which is optimized with FFTW if available
    return hilbert(bold_filtered, axis=1)

def calc_fcd(bold, fnq, band_freq=(0.01, 0.1), n_avg=3):
    # Ensure n_avg > 0
    if n_avg < 1:
        raise ValueError("n_avg must be greater than 0")

    _, nt = np.shape(bold)  
    # Bandpass filter the BOLD signal
    phase = np.angle(calc_hilbert(bold, fnq, band_freq=band_freq, k=2))

    # Remove first 9 and last 9 time points to avoid edge effects from filtering, as the bandpass 
    # filter may introduce distortions near the boundaries of the time series.
    t_trunc = np.arange(9, nt - 9)

    # Calculate synchrony
    triu_inds = np.triu_indices(np.shape(bold)[0], k=1)
    nt_trunc = len(t_trunc)
    synchrony_vecs = np.zeros((nt_trunc, len(triu_inds[0])))
    for t_ind, t in enumerate(t_trunc):
        phase_diff = np.subtract.outer(phase[:, t], phase[:, t])
        synchrony_mat = np.cos(phase_diff)
        synchrony_vecs[t_ind, :] = synchrony_mat[triu_inds]

    # Pre-calculate phase vectors
    p_mat = np.zeros((nt_trunc - n_avg-1, synchrony_vecs.shape[1]))
    for t_ind in range(nt_trunc - n_avg-1):
        p_mat[t_ind, :] = np.mean(synchrony_vecs[t_ind : t_ind+n_avg, :], axis=0)
        p_mat[t_ind, :] = p_mat[t_ind, :] / norm(p_mat[t_ind, :])

    # Calculate phase for every time pair
    fcd_mat = p_mat @ p_mat.T

    triu_ind = np.triu_indices(fcd_mat.shape[0], k=1)
    fcd = fcd_mat[triu_ind]

    return fcd

def calc_fcd_efficient(bold, fnq, band_freq=(0.01, 0.1), n_avg=3, chunk_size=50, metric="phase"):
    if n_avg < 1:
        raise ValueError("n_avg must be greater than 0")

    bold = bold.astype(np.float32, copy=False)

    n_regions, nt = bold.shape
    if metric == "amplitude":
        # Calculate amplitude of the BOLD signal
        phase = np.abs(calc_hilbert(bold, fnq, band_freq=band_freq, k=2))
    elif metric == "phase":
        phase = np.angle(calc_hilbert(bold, fnq, band_freq=band_freq, k=2))
    else:
        raise ValueError("Invalid metric. Choose 'amplitude' or 'phase'.")

    t_trunc = np.arange(9, nt - 9)
    nt_trunc = len(t_trunc)
    phase_trunc = phase[:, t_trunc]  # Preloaded for efficient access

    triu_i, triu_j = np.triu_indices(n_regions, k=1)
    n_edges = len(triu_i)

    p_mat = np.empty((nt_trunc - n_avg - 1, n_edges), dtype=np.float32)
    avg_sync = np.empty(n_edges, dtype=np.float32)  # Preallocated for reuse

    for chunk_start in range(0, nt_trunc, chunk_size):
        chunk_end = min(chunk_start + chunk_size, nt_trunc)
        chunk_size_actual = chunk_end - chunk_start

        synchrony_chunk = np.empty((chunk_size_actual, n_edges), dtype=np.float32)

        for i in range(chunk_size_actual):
            x = phase_trunc[:, chunk_start + i][:, None]
            y = phase_trunc[:, chunk_start + i][None, :]
            synchrony_mat = np.cos(x - y)
            synchrony_chunk[i, :] = synchrony_mat[triu_i, triu_j]

        for i in range(chunk_size_actual):
            global_t = chunk_start + i
            if global_t < nt_trunc - n_avg - 1:
                start_idx = max(0, global_t - chunk_start)
                end_idx = min(chunk_size_actual, global_t + n_avg - chunk_start)

                if end_idx > start_idx:
                    avg_sync = np.mean(synchrony_chunk[start_idx:end_idx, :], axis=0)

                    norm_val = np.sqrt(np.sum(avg_sync ** 2))
                    if norm_val > 1e-6:
                        p_mat[global_t, :] = avg_sync / norm_val
                    else:
                        p_mat[global_t, :] = 0.0

    fcd_mat = p_mat @ p_mat.T
    triu_ind = np.triu_indices(fcd_mat.shape[0], k=1)
    fcd_upper = fcd_mat[triu_ind]
    
    return fcd_upper

def calc_edge_fc(fc, eps=1e-7, fisher_z=False):
    triu_inds = np.triu_indices(np.shape(fc)[0], k=1)
    edge_fc = np.clip(fc[triu_inds], -1 + eps, 1 - eps)
    if fisher_z:
        edge_fc = np.arctanh(edge_fc)

    return edge_fc

def calc_node_fc(fc, eps=1e-7):
    # Compute Node-level FC (exclude diagonal elements by setting them to NaN)
    fc_nandiag = np.clip(fc, -1 + eps, 1 - eps)
    np.fill_diagonal(fc_nandiag, np.nan)
    node_fc = np.nanmean(np.arctanh(fc_nandiag), axis=1)
    
    return node_fc

def calc_gen_phase(bold, tr=0.72, lowcut=0.01, highcut=0.1):
    # Calculate phase delay
    phase_delay = np.empty(bold.shape, dtype=np.complex128)
    for i in range(bold.shape[0]):
        phase_delay[i, :] = generalized_phase_vector(bold[i, :], Fs=1/0.72, lp=lowcut)[0]

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
