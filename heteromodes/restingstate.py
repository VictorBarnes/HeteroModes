import numpy as np
import nibabel as nib
from scipy.stats import ks_2samp
from sklearn.preprocessing import StandardScaler
from brainspace.utils.parcellation import reduce_by_labels
from brainspace.mesh import mesh_io
from heteromodes.solver import HeteroSolver
from heteromodes.models import WaveModel, BalloonModel
from heteromodes.utils import scale_hmap, load_hmap, calc_phase_fcd, unmask, load_parc

# Turn off VTK warning when using importing brainspace.mesh_operations:  
# "vtkThreshold.cxx:99 WARN| vtkThreshold::ThresholdBetween was deprecated for VTK 9.1 and will be removed in a future version."
# import vtk
# vtk.vtkObject.GlobalWarningDisplayOff()


class ModelBOLD(object):
    def __init__(self, surf_file, medmask, hmap=None, alpha=1.0, sigma=0,
                 r=28.9, gamma=0.116, scale_method="zscore"):
        
        # Load surface template
        surf = mesh_io.read_surface(surf_file)
        # Load parcellation and extract medial mask (where parcellation != 0)
        # parc = load_parc(parc_file)
        # if len(parc) != surf.n_points:
        #     raise ValueError("Parcellation file must have the same number of elements as the number of vertices in the surface template")
        # medmask = np.where(parc != 0, True, False)
        # n_parcels = len(np.unique(parc[medmask]))
        # Mask surface template
        # medmask = nib.load(medmask).darrays[0].data
        # surf_masked = mesh_operations.mask_points(surf, medmask)

        # Load heterogeneity map
        # if hmap_file is None:
        #     # No heterogeneity is encoded by an array of ones
        #     hmap = None
        #     # Alpha must be equal to 0 in the homogeneous case
        #     self.alpha = 0.0
        # else:
        #     hmap = nib.load(hmap_file).darrays[0].data
            # hmap = load_hmap(hmap, medmask=medmask, den=den)
            # hmap = scale_hmap(hmap, alpha, beta, r, gamma, method=scale_method, verbose=False)

        self.surf = surf
        self.medmask = medmask
        # self.parc = parc
        # self.n_parcels = n_parcels
        self.hmap = hmap

        self.scale_method = scale_method
        self.alpha = alpha
        self.sigma = sigma
        self.r = r
        self.gamma = gamma

    def calc_modes(self, n_modes=500, method="hetero"):
        """Calculate heterogeneous modes."""
        self.solver = HeteroSolver(surf=self.surf, medmask=self.medmask, hmap=self.hmap, alpha=self.alpha, 
                                   method=method, sigma=self.sigma, scale_method=self.scale_method)
        self.evals, self.emodes = self.solver.solve(k=n_modes, fix_mode1=True, standardise=True)

    # def run(self, sim_seed=None, solver_method='Fourier', eig_method='orthonormal'):
    def run_rest(self, ext_input=None, sim_seed=None, solver_method='Fourier', eig_method='orthonormal',
            tstep=0.09*1e3):
        """Model resting-state fMRI BOLD data."""

        # Calculate simulated BOLD data
        TR = 0.72 * 1e3     # HCP data TR in ms

        # Set wave model parameters (all time units are in ms)
        # tstep = 0.09 * 1e3
        tpre = 50 * 1e3         # burn time to remove transient
        tmax = tpre + 1199*TR  # match number of timepoints in empirical data
        wave = WaveModel(self.emodes, self.evals, r=self.r, gamma=self.gamma, tstep=tstep,
                         tmax=tmax)
        # tpost = 863.28 * 1e3    # time during steady-state
        # wave.tmax = tpre + wave.tstep + tpost
        ntsteps_tr = int(TR // wave.tstep)        # Number of time steps in a TR
        # wave.tmax = tpost + wave.tstep * (ntsteps_tr//2 + 1)
        # wave.tspan = [0, tpost]
        # wave.t = np.arange(0, wave.tmax, wave.tstep)
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


def evaluate_model(empirical, model, TR=0.72):
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

    # scaler = StandardScaler()

    if np.shape(fc_emp)[0] != np.shape(fc_model)[0]:
        raise ValueError("Empirical and model data do not have the same number of regions")
    nparcels = np.shape(fc_emp)[0]

    triu_inds = np.triu_indices(nparcels, k=1)
    # bold_sim = scaler.fit_transform(bold_sim.T).T

    # # Calculate model FC
    # fc_model_run = np.corrcoef(bold_sim)
    # fc_model = fc_model_run

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
    # fcd_model_run = calc_phase_fcd(bold_sim, TR)
    # fcd_model = fcd_model_run
    fcd_ks = ks_2samp(np.hstack(fcd_emp), np.hstack(fcd_model))[0]

    return edge_fc, node_fc, fcd_ks, fc_model, fcd_model

