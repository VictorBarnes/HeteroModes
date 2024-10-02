import numpy as np
from scipy.stats import ks_2samp
from scipy.linalg import norm
from scipy.signal import butter, filtfilt, detrend, hilbert
from sklearn.preprocessing import StandardScaler
from brainspace.mesh import mesh_io
from heteromodes.solver import HeteroSolver
from heteromodes.models import WaveModel, BalloonModel

class ModelBOLD(object):
    def __init__(self, surf, medmask, hmap=None, alpha=1.0, sigma=0,
                 r=28.9, gamma=0.116, scale_method="zscore"):
        
        # Load surface template
        self.surf = surf
        self.medmask = medmask
        self.hmap = hmap

        self.scale_method = scale_method
        self.alpha = alpha
        self.sigma = sigma
        self.r = r
        self.gamma = gamma

    def calc_modes(self, n_modes=500, method="hetero"):
        """Calculate heterogeneous modes."""
        self.solver = HeteroSolver(
            surf=self.surf, 
            medmask=self.medmask, 
            hmap=self.hmap, 
            alpha=self.alpha, 
            method=method, 
            sigma=self.sigma, 
            scale_method=self.scale_method
        )
        self.evals, self.emodes = self.solver.solve(k=n_modes, fix_mode1=True, standardise=True)

    # def run(self, sim_seed=None, solver_method='Fourier', eig_method='orthonormal'):
    def run_rest(self, ext_input=None, sim_seed=None, solver_method='Fourier', 
                 eig_method='orthonormal', tstep=0.09*1e3):
        """Model resting-state fMRI BOLD data."""

        # Calculate simulated BOLD data
        TR = 0.72 * 1e3     # HCP data TR in ms

        # Set wave model parameters (all time units are in ms)
        tpre = 50 * 1e3         # burn time to remove transient
        tmax = tpre + 1199*TR  # match number of timepoints in empirical data
        wave = WaveModel(self.emodes, self.evals, r=self.r, gamma=self.gamma, tstep=tstep,
                         tmax=tmax)
        ntsteps_tr = int(TR // wave.tstep)        # Number of time steps in a TR
        # Get steady-state index
        tsteady_ind = np.abs(wave.t - tpre).argmin()

        # Set balloon model parameters
        balloon = BalloonModel(self.emodes, tstep=tstep, tmax=tmax)

        # random external input (to mimic resting state)
        if ext_input is None:
            sim_seed = np.random.randint(0, 2**32-1) if sim_seed is None else sim_seed
            np.random.seed(sim_seed)
            ext_input = np.random.randn(np.shape(self.emodes)[0], len(wave.t))
        else:
            # Check shape of external input
            if np.shape(ext_input) != (np.shape(self.emodes)[0], len(wave.t)):
                raise ValueError("External input must have shape (n_vertices, n_timepoints)")

        # simulate neural activity
        _, neural_activity = wave.solve(ext_input, solver_method, eig_method, B=self.solver.mass)
        # simulate BOLD activity from the neural activity
        _, bold_activity = balloon.solve(neural_activity, solver_method, eig_method,
                                        B=self.solver.mass)

        bold_activity_out = bold_activity[:, tsteady_ind::ntsteps_tr]

        return bold_activity_out
    
    def run_stimulate(self, solver_method='Fourier', eig_method='orthonormal'):
        # Get V1 indices
        roi_labels = [roi.label for roi in self.parc.labeltable.labels]
        v1_inds = np.where(self.parc.darrays[0].data == roi_labels.index("L_V1_ROI"), True, False)[self.medmask]

        # Initialise wave model
        wave = WaveModel(self.emodes, self.evals)

        # Create a 1 ms external input with amplitude = 20 to V1 (results are robust to amplitude)
        ext_trange = [1, 2]     # in ms
        ext_amp = 20
        ext_inds = np.where((wave.t >= ext_trange[0]) & (wave.t <= ext_trange[1]))[0]
        ext_input = np.zeros((np.shape(self.emodes)[0], len(wave.t)))
        ext_input[np.ix_(v1_inds, ext_inds)] = ext_amp

        # Simulate neural activity
        _, neural_sim = wave.solve(ext_input, solver_method, eig_method, B=self.solver.mass)

        return neural_sim

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

    # Pre-calculate phase vectors
    p_mat = np.zeros((nt - 2, synchrony_vec.shape[1]))
    for t_ind in range(nt - 2):
        p_mat[t_ind, :] = np.mean(synchrony_vec[t_ind:t_ind + 3, :], axis=0)
        p_mat[t_ind, :] = p_mat[t_ind, :] / norm(p_mat[t_ind, :])

    # Calculate phase for every time pair
    fcd_mat = p_mat @ p_mat.T

    triu_ind = np.triu_indices(fcd_mat.shape[0], k=1)
    fcd = fcd_mat[triu_ind]

    return fcd

def calc_fc_fcd(bold, tr, filter=False):
    # Ensure data is standardised
    if not np.isclose(np.mean(bold, axis=1), 0).all() or not np.isclose(np.std(bold, axis=1), 1.0).all():
        scaler = StandardScaler()
        bold = scaler.fit_transform(bold.T).T
    # Bandpass filter the data
    if filter:
        bold = filter_bold(bold, tr=tr)
    
    # Caculate FC and FCD
    fc = np.corrcoef(bold)
    fcd = calc_phase_fcd(bold, tr=tr)

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

    return edge_fc, node_fc, fcd_ks, fc_model, fcd_model
