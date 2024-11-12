import pathlib
import numpy as np
from lapy import Solver, TriaMesh
from sklearn.preprocessing import StandardScaler
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

def _check_hmap(surf, hmap):
    """Validate the heterogeneity map and return as a numpy array."""
    if hmap is None:
        return np.zeros(surf.n_points)
    elif isinstance(hmap, np.ndarray):
        if len(hmap) != surf.n_points:
            raise ValueError("Heterogeneity map must have the same number of elements as the number of vertices in the surface template")
        # Replace nan values with 1
        hmap[np.isnan(hmap)] = np.nanmean(hmap)
        
        return hmap
    else:
        raise ValueError("Heterogeneity map must be a numpy array or None")

class HeteroSolver(Solver):
    """
    Class to solve the heterogeneous Helmholtz equation on a surface mesh.
    """
    def __init__(self, surf, hmap=None, medmask=None, alpha=1.0, verbose=False, **lapy_kwargs):  
        """
        Initialize the HeteroSolver object.

        Parameters
        ----------
        surf : str
            Path to the surface template file or BSPolyData. 
            See `brainspace.mesh.mesh_io.read_surface` for acceptable file types.
        hmap : np.ndarray, optional
            Array of the heterogeneity map, by default None. If None, hmap will be set to an array
            of zeros of length (n_vertices,) and this will results in homogeneous modes.
            If hmap is a numpy array, then it must contain the same number of elements as the number
            of vertices in `surf`.
        medmask : np.ndarray, optional
            Mask to apply to the surface vertices, by default None. If None, no mask will be 
            applied.
        alpha : float, optional
            Scaling factor for the heterogeneity map, by default 1.0.
        verbose : bool, optional
            Flag indicating whether to print the solver information, by default False.
        """

        # Initialise surface and convert to TriaMesh object
        surf = _check_surf(surf)
        if medmask is not None:
            surf = mask_points(surf, medmask)
            if hmap is not None:
                hmap = hmap[medmask]
        mesh = TriaMesh(get_points(surf), get_cells(surf))

        # Check and scale heterogeneity map
        if hmap is None:
            alpha = 0
        hmap = _check_hmap(surf, hmap)
        rho = scale_hmap(hmap, alpha=alpha)       

        # Map hmap from vertices to triangles
        rho_tri = mesh.map_vfunc_to_tfunc(rho)

        # Initialise the Solver object
        super().__init__(mesh, aniso=(0, 0), hetero=rho_tri, verbose=verbose, **lapy_kwargs)
        
        # Store the parameters
        self.mesh = mesh
        self.rho = rho
        
    def solve(self, n_modes=10, fix_mode1=False, standardise=False):
        """
        Solve for eigenvalues and eigenmodes of the HeteroModes problem.

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
    
def scale_hmap(hmap, alpha=1.0):
    """
    Scale the heterogeneity map using the given parameters.

    Parameters
    ----------
    hmap : numpy.ndarray
        The heterogeneity map to be scaled.
    alpha : float
        The scaling factor.

    Returns
    -------
    rho : numpy.ndarray
        The scaled heterogeneity map.
    """

    # z-score the heterogeneity map
    scaler = StandardScaler()
    rho = np.exp(alpha * scaler.fit_transform(hmap.reshape(-1, 1)).flatten())

    return rho
