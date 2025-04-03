import os
import pathlib
import numpy as np
from joblib import Memory
from pathlib import Path
from dotenv import load_dotenv
from lapy import Solver, TriaMesh
from lapy.utils._imports import import_optional_dependency
from scipy.ndimage import gaussian_filter
from scipy.stats import zscore
from scipy.signal import butter, filtfilt, hilbert
from sklearn.preprocessing import QuantileTransformer
from brainspace.vtk_interface.wrappers import BSPolyData
from brainspace.mesh.mesh_operations import mask_points
from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh.mesh_elements import get_cells, get_points
from heteromodes.models import WaveModel, BalloonModel
from heteromodes.utils import standardise_modes

# Turn off VTK warning when using importing brainspace.mesh_operations:  
# "vtkThreshold.cxx:99 WARN| vtkThreshold::ThresholdBetween was deprecated for VTK 9.1 and will be removed in a future version."
import vtk
vtk.vtkObject.GlobalWarningDisplayOff()

# Set up joblib memory caching
load_dotenv()
CACHE_DIR = os.getenv("CACHE_DIR")
if CACHE_DIR is None or not os.path.exists(CACHE_DIR):
    CACHE_DIR = Path.cwd() / "cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
memory = Memory(Path(CACHE_DIR), verbose=0)

def _check_surf(surf):
    """Validate surface type and load if a file name. Adapted from `surfplot`."""
    if isinstance(surf, (str, pathlib.Path)):
        return read_surface(str(surf))
    elif isinstance(surf, BSPolyData) or (surf is None):
        return surf
    else:
        raise ValueError('Surface be a path-like string, an instance of '
                        'BSPolyData, or None')

@memory.cache
def gen_random_input(n_points, n_timepoints, seed=None):
    """Generates external input with caching to avoid redundant recomputation."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(n_points, n_timepoints)

class EigenSolver(Solver):
    def __init__(self, surf, medmask=None, hetero=None, alpha=0, r=28.9, gamma=0.116, scaling="sigmoid", 
                 q_norm=None, lump=False, smoothit=10, verbose=False):
        self._r = r
        self._gamma = gamma
        self.alpha = alpha
        self.scaling = scaling
        self.q_norm = q_norm
        self.verbose = verbose

        # Initialise surface and convert to TriaMesh object
        surf = _check_surf(surf)
        if medmask is not None:
            surf = mask_points(surf, medmask)
            if hetero is not None:
                hetero = hetero[medmask]
        self.geometry = TriaMesh(get_points(surf), get_cells(surf))
        self.surf = surf  
        self.hetero = hetero

        # Calculate the two matrices of the Laplace-Beltrami operator
        self.stiffness, self.mass = self.laplace_beltrami(lump=lump, smoothit=smoothit)

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, r):
        self.check_hetero(hetero=self.hetero, r=r, gamma=self.gamma)
        self._r = r

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        self.check_hetero(hetero=self.hetero, r=self.r, gamma=gamma)
        self._gamma = gamma

    @property
    def hetero(self):
        return self._hetero

    @hetero.setter
    def hetero(self, hetero):
        # Handle None case by setting to ones
        if hetero is None:
            if self.alpha != 0:
                print("Setting `alpha` to 0 because `hetero` is None.")
                self.alpha = 0
            self._hetero = np.ones(self.surf.n_points)
        else:
            # Ensure hetero is valid
            if not isinstance(hetero, np.ndarray):
                raise ValueError("Heterogeneity map must be a numpy array or None")
            if len(hetero) != self.surf.n_points:
                raise ValueError("Heterogeneity map must have the same number of elements as the number of vertices in the surface template.")
            if np.isnan(hetero).any() or np.isinf(hetero).any():
                raise ValueError("Heterogeneity map must not contain NaNs or Infs.")

            # Scale the heterogeneity map
            hetero = self.scale_hetero(hetero=hetero, alpha=self.alpha, scaling=self.scaling, q_norm=self.q_norm)

            # Check the heterogeneity does not result in non-physiological wave speeds
            self.check_hetero(hetero=hetero, r=self.r, gamma=self.gamma)

            # Assign to private attribute
            self._hetero = hetero

    @staticmethod
    def check_hetero(hetero, r, gamma):
        # Check hmap values are physiologically plausible
        if np.max(r * gamma * np.sqrt(hetero)) > 150:
            raise ValueError("Alpha value results in non-physiological wave speeds (> 150 m/s). Try" 
                             " using a smaller alpha value.")

    @staticmethod
    def scale_hetero(hetero=None, alpha=1.0, scaling="sigmoid", q_norm=None): 
        # Z-score the heterogeneity map
        hetero = zscore(hetero)

        # Apply quantile normalisation
        if q_norm is not None:
            scaler = QuantileTransformer(output_distribution=q_norm, random_state=0)
            hetero = scaler.fit_transform(hetero.reshape(-1, 1)).flatten()

        # Scale the heterogeneity map
        if scaling == "exponential":
            hetero = np.exp(alpha * hetero)
        elif scaling == "sigmoid":
            hetero = 2 / (1 + np.exp(-alpha * hetero))
        else:
            raise ValueError("Invalid scaling function. Must be 'exponential' or 'sigmoid'.")

        return hetero

    def laplace_beltrami(self, lump=False, smoothit=10):        
        hetero_tri = self.geometry.map_vfunc_to_tfunc(self.hetero)
        # Check that the length of the heterogeneity map matches the number of triangles
        if len(hetero_tri) != self.geometry.t.shape[0]:
            raise ValueError(f"Wrong hetero length: {len(hetero_tri)}. Should be: "
                                f"{self.geometry.t.shape[0]}")

        # heterogneous Laplace
        if self.verbose:
            print("TriaMesh with heterogeneous Laplace-Beltrami")
        u1, u2, _, _ = self.geometry.curvature_tria(smoothit=smoothit)

        hetero_mat = np.tile(hetero_tri[:, np.newaxis], (1, 2))
        a, b = self._fem_tria_aniso(self.geometry, u1, u2, hetero_mat, lump)
        
        return a, b

    def solve(self, k=10, fix_mode1=False, standardise=False, use_cholmod=False):
        """
        Solve for eigenvalues and eigenmodes

        Parameters
        ----------

        Returns
        -------

        """

        self.use_cholmod = use_cholmod
        if self.use_cholmod:
            self.sksparse = import_optional_dependency("sksparse", raise_error=True)
            importlib.import_module(".cholmod", self.sksparse.__name__)
        else:
            self.sksparse = None
        
        # Solve the eigenvalue problem
        self.evals, evecs = self.eigs(k=k, verbose=self.verbose)
        
        # Set first mode to be constant (mean of first column)
        if fix_mode1:
            evecs[:, 0] = np.mean(evecs[:, 0])

        # Standardise sign of modes
        if standardise:
            evecs = standardise_modes(evecs)

        # Check for NaNs
        if np.isnan(evecs).any():
            raise AssertionError("`evecs` contain NaNs") 

        self.evecs = evecs
        
    def simulate_bold(self, ext_input=None, dt=0.1, nt=1000, tsteady=0, solver_method="Fourier", eig_method="orthogonal", seed=None):
        tmax = dt * nt + tsteady
        wave = WaveModel(self.evecs, self.evals, r=self.r, gamma=self.gamma, tstep=dt, tmax=tmax)
        self.t = wave.t
        
        # Check if external input is provided, otherwise generate random input
        if ext_input is None:
            ext_input = gen_random_input(self.surf.n_points, len(self.t), seed=seed)
        # Ensure the external input has the correct shape
        if ext_input.shape != (self.surf.n_points, len(self.t)):
            raise ValueError(f"External input shape {ext_input.shape} does not have the correct shape ({self.surf.n_points}, {len(self.t)}).")
        
        # Simulate neural activity
        _, neural = wave.solve(ext_input, solver_method, eig_method, mass=self.mass)
        
        # Simulate BOLD activity
        balloon = BalloonModel(self.evecs, tstep=dt, tmax=tmax)
        _, bold = balloon.solve(neural, solver_method, eig_method, mass=self.mass)

        # Return only the steady state part
        tsteady_ind = np.abs(self.t - tsteady).argmin()

        return bold[:, tsteady_ind:]  

    def filter_bold(self, bold, fnq, band_freq=(0.01, 0.1), k=2):
        """Filter the BOLD signal using a bandpass filter."""
        # Define parameters
        Wn = [band_freq[0]/fnq, band_freq[1]/fnq]
        bfilt2, afilt2 = butter(k, Wn, btype="bandpass")

        bold_z = zscore(bold, axis=1)
        bold_filtered = filtfilt(bfilt2, afilt2, bold_z, axis=1)

        return bold_filtered

    def calc_fc(self, bold):
        bold_z = zscore(bold, axis=1)
        fc_matrix = np.corrcoef(bold)

        return fc_matrix

    def calc_hilbert(self, bold, fnq, band_freq=(0.01, 0.1), k=2, conj=False):
        bold_filtered = self.filter_bold(bold, fnq, k=k, band_freq=band_freq)
        
        if conj:
            return hilbert(bold_filtered, axis=1).conj()
        else:
            return hilbert(bold_filtered, axis=1)

    def calc_fcd(self, bold, fnq, band_freq=(0.01, 0.1), n_avg=3):
        # Ensure n_avg > 0
        if n_avg < 1:
            raise ValueError("n_avg must be greater than 0")

        _, nt = np.shape(bold)  
        # Bandpass filter the BOLD signal
        phase = np.angle(self.calc_hilbert(bold_filtered, fnq, band_freq=band_freq, k=k, conj=False))

        # Remove first 9 and last 9 time points to avoid edge effects from filtering, as the bandpass 
        # filter may introduce distortions near the boundaries of the time series.
        t_trunc = np.arange(9, t - 9)

        # Calculate synchrony
        triu_inds = np.triu_indices(self.surf.n_points, k=1)
        nt_trunc = len(t_trunc)
        synchrony_vecs = np.zeros((nt_trunc, len(triu_inds[0])))
        for t_ind, t in enumerate(t_trunc):
            phase_diff = np.subtract.outer(phase[:, nt], phase[:, nt])
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


