"""Neural field model simulation and resting-state fMRI analysis."""

import fbpca
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import zscore, ks_2samp
from brainspace.utils.parcellation import reduce_by_labels
from neuromodes.eigen import EigenSolver


def run_model(
    # Solver parameters
    surf, 
    medmask=None, 
    hetero=None,
    alpha=0, 
    aniso=None,
    beta=1.0,
    r=28.9, 
    gamma=116, 
    scaling="sigmoid", 
    lump=False, 
    smoothit=10,
    n_modes=500, 
    # Simulation parameters
    n_runs=5, 
    nt_emp=1200, 
    dt_emp=0.72, 
    dt_model=0.09, 
    tsteady=500, 
    decomp_method="project",
    pde_method="fourier",
    parc=None
):
    """
    Run neural field model simulation and generate BOLD time series.
    
    This function is designed for parallel execution, instantiating the EigenSolver
    internally rather than accepting it as an argument.

    Parameters
    ----------
    surf : Surface object
        Cortical surface for the simulation.
    medmask : array-like, optional
        Binary mask for medial wall regions.
    hetero : array-like, optional
        Heterogeneity map for spatial variation in model parameters.
    alpha : float, default=0
        Scaling parameter for wave propagation speed.
    beta : float, default=1.0
        Spatial smoothing parameter.
    r : float, default=28.9
        Global coupling strength (mm).
    gamma : float, default=116
        Decay parameter for spatial interactions (s^-1)
    scaling : str, default="sigmoid"
        Scaling function type for spatial interactions.
    lump : bool, default=False
        Whether to use mass lumping in finite element method.
    smoothit : int, default=10
        Number of spatial smoothing iterations.
    n_modes : int, default=500
        Number of eigenmodes for spectral decomposition.
    n_runs : int, default=5
        Number of independent simulation runs.
    nt_emp : int, default=1200
        Number of empirical timepoints to return.
    dt_emp : float, default=0.72
        Empirical sampling interval in seconds.
    dt_model : float, default=0.09
        Model simulation time step in seconds.
    tsteady : int, default=500
        Number of initial timepoints to discard as steady-state burn-in.
    decomp_method : str, default="project"
        Eigenmode decomposition method.
    parc : array-like, optional
        Parcellation labels for aggregating vertex-wise activity.

    Returns
    -------
    bold : np.ndarray
        BOLD time series with shape (n_regions, nt_emp, n_runs).
        Returns (None, None) if parameters result in non-physiological wave speeds.
    """
    
    # Initialize eigenmode solver with model parameters
    solver = EigenSolver(
        surf=surf, 
        mask=medmask, 
        hetero=hetero,
        alpha=alpha,
        scaling=scaling, 
        # aniso=aniso, 
        # beta=beta, 
    )
    solver = solver.solve(
        n_modes=n_modes, 
        fix_mode1=True, 
        standardize=False,
        seed=365,
        lump=lump,
        smoothit=smoothit,
    )

    # Calculate total model timepoints including steady-state period
    downsample_factor = int(dt_emp / dt_model)
    nt_model = int(nt_emp * downsample_factor) + tsteady

    # Determine output dimensions (vertex-wise or parcellated)
    n_regions = solver.n_verts if parc is None else len(np.unique(parc[medmask]))
    bold = np.empty((n_regions, nt_emp, n_runs), dtype=np.float32)
    
    # Run multiple simulations with different random seeds
    for i in range(n_runs):
        # Simulate neural activity and convert to BOLD signal
        bold_i = solver.simulate_waves(
            r=r, 
            gamma=gamma, 
            dt=dt_model, 
            nt=nt_model, 
            seed=i, 
            bold_out=True, 
            decomp_method=decomp_method,
            pde_method=pde_method
        ).astype(np.float32)
        
        # Remove steady-state period
        bold_i = bold_i[:, tsteady:]
        
        # Downsample to empirical sampling rate
        bold_i = bold_i[:, ::downsample_factor]

        # Apply parcellation if provided
        if parc is not None:
            bold_i = reduce_by_labels(bold_i, parc[medmask], axis=1)

        # Z-score normalize each region's time series
        bold[:, :, i] = zscore(bold_i, axis=1).astype(np.float32)

    return bold


def analyze_bold(bold, dt_emp=0.72, band_freq=(0.01, 0.1), 
                 metrics=["edge_fc_corr", "node_fc_corr", "fcd_ks"]):
    """
    Compute functional connectivity metrics from BOLD time series.

    Parameters
    ----------
    bold : np.ndarray
        BOLD time series with shape (n_regions, n_timepoints, n_runs).
    dt_emp : float, default=720
        Empirical sampling interval in seconds.
    band_freq : tuple, default=(0.01, 0.1)
        Frequency band (low, high) in Hz for bandpass filtering.
    metrics : list, default=["edge_fc_corr", "node_fc_corr", "fcd_ks"]
        Metrics to compute: "edge_fc_corr", "node_fc_corr", "cpc1", "fcd_ks".

    Returns
    -------
    outputs : dict
        Dictionary containing requested metrics:
        - 'fc': Functional connectivity matrix
        - 'analytic': Analytic signal from Hilbert transform
        - 'fcd': Functional connectivity dynamics
    """
    outputs = {}
    bold = bold.astype(np.float32, copy=False)
    bold_concat = np.hstack([bold[:, :, i] for i in range(bold.shape[2])], dtype=np.float32)
    
    # Compute functional connectivity if needed
    if "edge_fc_corr" in metrics or "node_fc_corr" in metrics:
        outputs['fc'] = calc_fc(bold_concat)
    
    # Compute phase-based metrics if needed
    if "cpc1_corr" in metrics or 'fcd_ks' in metrics:
        fnq = 0.5 / dt_emp  # Nyquist frequency in Hz

        if "cpc1_corr" in metrics:
            analytic = calc_hilbert(
                bold_concat, fnq=fnq, band_freq=band_freq, k=2
            ).astype(np.complex64)
            ncpcs = 10
            l = 10 + ncpcs
            _, _, V = fbpca.pca(analytic.T, k=ncpcs, n_iter=20, l=l)
            outputs['cpcs'] = V.T.astype(np.complex64)
            
        if "fcd_ks" in metrics:
            fcd = [calc_fcd_efficient(bold[:, :, i], fnq=fnq, band_freq=band_freq).astype(np.float32)
                   for i in range(bold.shape[2])]
            outputs['fcd'] = np.array(fcd, dtype=np.float32).T

    return outputs


def evaluate_model(model_outputs, emp_outputs, 
                   metrics=["edge_fc_corr", "node_fc_corr", "fcd_ks"]):
    """
    Evaluate model fit to empirical data using multiple metrics.
    
    Parameters
    ----------
    model_outputs : dict
        Model-derived metrics (keys: 'fc', 'fcd', etc.).
    emp_outputs : dict  
        Empirical data metrics (keys: 'fc', 'fcd', etc.).
    metrics : list, default=["edge_fc_corr", "node_fc_corr", "fcd_ks"]
        Metrics to compute: "edge_fc_corr", "node_fc_corr", "fcd_ks".

    Returns
    -------
    results : dict
        Dictionary containing computed metric values:
        - 'edge_fc_corr': Pearson correlation between edge FC vectors
        - 'node_fc_corr': Pearson correlation between node FC vectors
        - 'fcd_ks': Kolmogorov-Smirnov statistic for FCD distributions
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
    
    if "fcd_ks" in metrics:
        results['fcd_ks'] = ks_2samp(
            model_outputs['fcd'].flatten(), 
            emp_outputs['fcd'].flatten()
        )[0]

    if "cpc1_corr" in metrics:
        model_cpc1 = np.imag(model_outputs['cpcs'][:, 0])
        emp_cpc1 = np.imag(emp_outputs['cpcs'][:, 0])
        results['cpc1_corr'] = np.corrcoef(model_cpc1, emp_cpc1)[0, 1]
    
    return results


def filter_bold(bold, fnq, band_freq=(0.01, 0.1), k=2):
    """
    Apply Butterworth bandpass filter to BOLD signal.
    
    Parameters
    ----------
    bold : np.ndarray
        BOLD time series to filter.
    fnq : float
        Nyquist frequency in Hz.
    band_freq : tuple, default=(0.01, 0.1)
        Frequency band (low, high) in Hz.
    k : int, default=2
        Filter order.
    
    Returns
    -------
    bold_filtered : np.ndarray
        Bandpass filtered BOLD signal.
    """
    # Normalize frequency band to Nyquist frequency
    Wn = [band_freq[0] / fnq, band_freq[1] / fnq]
    b, a = butter(k, Wn, btype="bandpass")

    # Z-score and apply zero-phase filter
    bold_z = zscore(bold, axis=1).astype(np.float32)
    bold_filtered = filtfilt(b, a, bold_z, axis=1)

    return bold_filtered


def calc_fc(bold):
    """
    Calculate functional connectivity matrix.
    
    Parameters
    ----------
    bold : np.ndarray
        BOLD time series with shape (n_regions, n_timepoints).
    
    Returns
    -------
    fc_matrix : np.ndarray
        Pearson correlation matrix (n_regions, n_regions).
    """
    bold_z = zscore(bold, axis=1).astype(np.float32)
    fc_matrix = np.corrcoef(bold_z, dtype=np.float32)
    return fc_matrix


def calc_hilbert(bold, fnq, band_freq=(0.01, 0.1), k=2):
    """
    Compute analytic signal using Hilbert transform.
    
    Parameters
    ----------
    bold : np.ndarray
        BOLD time series.
    fnq : float
        Nyquist frequency in Hz.
    band_freq : tuple, default=(0.01, 0.1)
        Frequency band (low, high) in Hz for bandpass filtering.
    k : int, default=2
        Butterworth filter order.
    
    Returns
    -------
    analytic_signal : np.ndarray
        Complex-valued analytic signal.
    """
    bold_filtered = filter_bold(bold, fnq, k=k, band_freq=band_freq)
    return hilbert(bold_filtered, axis=1)


def calc_fcd_efficient(bold, fnq, band_freq=(0.01, 0.1), n_avg=3, 
                       chunk_size=50, metric="phase"):
    """
    Calculate FCD using memory-efficient chunked processing.
    
    This implementation processes time series in chunks to reduce memory usage
    for large datasets.
    
    Parameters
    ----------
    bold : np.ndarray
        BOLD time series with shape (n_regions, n_timepoints).
    fnq : float
        Nyquist frequency in Hz.
    band_freq : tuple, default=(0.01, 0.1)
        Frequency band (low, high) in Hz for bandpass filtering.
    n_avg : int, default=3
        Number of timepoints to average for sliding window.
    chunk_size : int, default=50
        Number of timepoints to process per chunk.
    metric : str, default="phase"
        Metric to use: "phase" or "amplitude".
    
    Returns
    -------
    fcd_upper : np.ndarray
        Upper triangular FCD matrix values.
    """
    if n_avg < 1:
        raise ValueError("n_avg must be greater than 0")

    bold = bold.astype(np.float32, copy=False)
    n_regions, nt = bold.shape
    
    # Extract phase or amplitude from analytic signal
    analytic = calc_hilbert(bold, fnq, band_freq=band_freq, k=2)
    if metric == "amplitude":
        signal = np.abs(analytic)
    elif metric == "phase":
        signal = np.angle(analytic)
    else:
        raise ValueError("Invalid metric. Choose 'amplitude' or 'phase'.")

    # Remove edge artifacts
    t_trunc = np.arange(9, nt - 9)
    nt_trunc = len(t_trunc)
    signal_trunc = signal[:, t_trunc]

    triu_i, triu_j = np.triu_indices(n_regions, k=1)
    n_edges = len(triu_i)

    # Pre-allocate normalized synchrony matrix
    n_windows = nt_trunc - n_avg - 1
    p_mat = np.empty((n_windows, n_edges), dtype=np.float32)

    # Process in chunks to reduce memory footprint
    for chunk_start in range(0, nt_trunc, chunk_size):
        chunk_end = min(chunk_start + chunk_size, nt_trunc)
        chunk_size_actual = chunk_end - chunk_start

        # Compute synchrony for current chunk
        synchrony_chunk = np.empty((chunk_size_actual, n_edges), dtype=np.float32)
        for i in range(chunk_size_actual):
            signal_t = signal_trunc[:, chunk_start + i]
            synchrony_mat = np.cos(signal_t[:, None] - signal_t[None, :])
            synchrony_chunk[i, :] = synchrony_mat[triu_i, triu_j]

        # Apply sliding window averaging within chunk
        for i in range(chunk_size_actual):
            global_t = chunk_start + i
            if global_t < n_windows:
                start_idx = max(0, global_t - chunk_start)
                end_idx = min(chunk_size_actual, global_t + n_avg - chunk_start)

                if end_idx > start_idx:
                    avg_sync = np.mean(synchrony_chunk[start_idx:end_idx, :], axis=0)
                    norm_val = np.sqrt(np.sum(avg_sync ** 2))
                    
                    p_mat[global_t, :] = avg_sync / norm_val if norm_val > 1e-6 else 0.0

    # Compute temporal correlation of synchrony patterns
    fcd_mat = p_mat @ p_mat.T
    triu_ind = np.triu_indices(fcd_mat.shape[0], k=1)
    
    return fcd_mat[triu_ind]


def calc_edge_fc(fc, eps=1e-7, fisher_z=False):
    """
    Extract upper triangular edge FC values with optional Fisher Z-transform.
    
    Parameters
    ----------
    fc : np.ndarray
        Functional connectivity matrix (n_regions, n_regions).
    eps : float, default=1e-7
        Small value to avoid arctanh singularities at ±1.
    fisher_z : bool, default=False
        Whether to apply Fisher Z-transformation.
    
    Returns
    -------
    edge_fc : np.ndarray
        Vector of upper triangular FC values.
    """
    triu_i, triu_j = np.triu_indices(fc.shape[0], k=1)
    edge_fc = np.clip(fc[triu_i, triu_j], -1 + eps, 1 - eps)
    
    if fisher_z:
        edge_fc = np.arctanh(edge_fc)

    return edge_fc


def calc_node_fc(fc, eps=1e-7):
    """
    Calculate node-level FC as mean Fisher Z-transformed connectivity.
    
    Parameters
    ----------
    fc : np.ndarray
        Functional connectivity matrix (n_regions, n_regions).
    eps : float, default=1e-7
        Small value to avoid arctanh singularities at ±1.
    
    Returns
    -------
    node_fc : np.ndarray
        Mean Fisher Z-transformed FC for each node.
    """
    # Clip values and exclude diagonal
    fc_clipped = np.clip(fc, -1 + eps, 1 - eps)
    np.fill_diagonal(fc_clipped, np.nan)
    
    # Apply Fisher Z-transform and compute mean
    node_fc = np.nanmean(np.arctanh(fc_clipped), axis=1)
    
    return node_fc

