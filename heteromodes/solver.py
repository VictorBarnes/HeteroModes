import pathlib
import numpy as np
from lapy import Solver, TriaMesh
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from brainspace.vtk_interface.wrappers import BSPolyData
from brainspace.mesh.mesh_operations import mask_points
from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh.mesh_elements import get_cells, get_points
from heteromodes.utils import standardise_modes

# Turn off VTK warning when using importing brainspace.mesh_operations:  
# "vtkThreshold.cxx:99 WARN| vtkThreshold::ThresholdBetween was deprecated for VTK 9.1 and will be removed in a future version."
import vtk
vtk.vtkObject.GlobalWarningDisplayOff()

def _check_surf(surf):
    """Validate surface type and load if a file name. Adapted from `surfplot`."""
    if isinstance(surf, (str, pathlib.Path)):
        return read_surface(str(surf))
    elif isinstance(surf, BSPolyData) or (surf is None):
        return surf
    else:
        raise ValueError('Surface be a path-like string, an instance of '
                         'BSPolyData, or None')

class EigenSolver(Solver):
    """
    Class to solve the heterogeneous Helmholtz equation on a surface mesh.

    Parameters
    ----------
    surf : str
        Path to the surface template file or BSPolyData. 
        See `brainspace.mesh.mesh_io.read_surface` for acceptable file types.
    medmask : np.ndarray, optional
        Mask to apply to the surface vertices, by default None. If None, no mask will be 
        applied.
    hetero : np.ndarray, optional
        Array of the heterogeneity map, by default None. If None, hmap will be set to an array
        of zeros of length (n_vertices,) and this will results in homogeneous modes.
        If hmap is a numpy array, then it must contain the same number of elements as the number
        of vertices in `surf`.
    alpha : float, optional
        Scaling factor for the heterogeneity map, by default 0. If alpha is 0, the heterogeneity
        map will be set to an array of ones of length (n_vertices,).
    sigma : int, optional
        The number of iterations to smooth the heterogeneity map, by default 0. If sigma is 0,
        no smoothing will be applied.
    scaling : str, optional
        The scaling function to apply to the heterogeneity map, by default "exponential".
        Must be either "exponential" or "sigmoid".
    q_norm : str, optional
        The quantile normalisation method to apply to the heterogeneity map, by default None.
        Must be either "uniform" or "normal". If None, no quantile normalisation will be applied.
    verbose : bool, optional
        Flag indicating whether to print the solver information, by default False.
    lapy_kwargs : keyword arguments
        Additional keyword arguments to pass to the lapy.Solver class.

    Attributes
    ----------
    mesh : TriaMesh
        The surface mesh object.
    rho : np.ndarray
        The heterogeneity map after scaling and smoothing.
    """
    def __init__(self, surf, medmask=None, hetero=None, alpha=0, sigma=0, scaling="exponential", 
                 q_norm=None, verbose=False, **lapy_kwargs):  
        CMEAN = 3.3524

        # Initialise surface and convert to TriaMesh object
        surf = _check_surf(surf)
        if medmask is not None:
            surf = mask_points(surf, medmask)
            if hetero is not None:
                hetero = hetero[medmask]
        mesh = TriaMesh(get_points(surf), get_cells(surf))

        if hetero is None:
            if alpha != 0:
                print("Setting `alpha` to 0 because `hetero` is None.")
                alpha = 0
            rho = np.ones(surf.n_points)
        elif isinstance(hetero, np.ndarray):
            if len(hetero) != surf.n_points:
                raise ValueError("Heterogeneity map must have the same number of elements as the number of vertices in the surface template")
            if np.isnan(hetero).any():
                raise ValueError("Heterogeneity map must not contain NaNs")
                
            # Z-score the heterogeneity map
            scaler = StandardScaler()
            hetero = scaler.fit_transform(hetero.reshape(-1, 1)).flatten()

            # Apply quartile normalisation
            if q_norm is not None:
                scaler = QuantileTransformer(output_distribution=q_norm, random_state=0)
                hetero = scaler.fit_transform(hetero.reshape(-1, 1)).flatten()
            
            # Smooth the heterogeneity map
            if sigma > 0:
                hetero = mesh.smooth_vfunc(hetero, n=sigma)

            # Scale the heterogeneity map
            if scaling == "exponential":
                rho = np.exp(alpha * hetero)
            elif scaling == "sigmoid":
                rho = 2 / (1 + np.exp(-alpha * hetero))
            else:
                raise ValueError("Invalid scaling function. Must be 'exponential' or 'sigmoid'.")

        else:
            raise ValueError("Heterogeneity map must be a numpy array or None")

        # Check hmap values are physiologically plausible
        if np.max(CMEAN * np.sqrt(rho)) > 150:
            raise ValueError("Alpha value results in non-physiological wave speeds (> 150 m/s). Try" 
                             " using a smaller alpha value.")

        # Map hmap from vertices to triangles
        rho_tri = mesh.map_vfunc_to_tfunc(rho)

        # Initialise the Solver object
        super().__init__(mesh, aniso=(0, 0), hetero=rho_tri, verbose=verbose, **lapy_kwargs)
        
        # Store the parameters
        self.mesh = mesh
        self.rho = rho
        
    def solve(self, n_modes=10, fix_mode1=False, standardise=False):
        """
        Solve for eigenvalues and eigenmodes

        Parameters
        ----------
        n_modes : int, optional
            The number of eigenmodes to compute. Defaults to 10.
        fix_mode1 : bool, optional
            Flag indicating whether to set the first eigenmode to be constant (mean of the first
            eigenmodes). The first eigenmode should always be constant by definition but is not due 
            to numerical errors. Defaults to True.
        standardise : bool, optional
            Flag indicating whether to perform standardisation by flipping the modes such that the 
            first element of each mode is positive. Defaults to True.

        Returns
        -------
        Tuple
            A tuple containing the eigenvalues and eigenmodes.

        Raises
        ------
        AssertionError
            Raised if the cortical indices of the eigenmodes contain NaNs.
        """
        evals, emodes = self.eigs(k=n_modes)
        
        # Set first mode to be constant (mean of first column)
        if fix_mode1:
            emodes[:, 0] = np.mean(emodes[:, 0])

        # Standardise sign of modes
        if standardise:
            emodes = standardise_modes(emodes)

        # Check for NaNs
        if np.isnan(emodes).any():
            raise AssertionError("Cortical indices of `emodes` contain NaNs") 

        return evals, emodes
    