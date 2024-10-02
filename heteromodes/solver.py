import pathlib
import numpy as np
from lapy import Solver, TriaMesh
from brainspace.vtk_interface.wrappers import BSPolyData
from brainspace.mesh.mesh_operations import mask_points
from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh.mesh_elements import get_cells, get_points
from heteromodes.utils import standardise_modes, scale_hmap

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
        return np.ones(surf.n_points)
    elif isinstance(hmap, np.ndarray):
        if len(hmap) != surf.n_points:
            raise ValueError("Heterogeneity map must have the same number of elements as the number of vertices in the surface template")
        # Replace nan values with 1
        hmap[np.isnan(hmap)] = np.nanmean(hmap)
        
        return hmap

# def _check_scaling(mesh, rho):
#     """Check if scaling is valid."""
#     CMEAN = 3.3524

#     rho_tri = mesh.map_vfunc_to_tfunc(rho)
#     _, _, c1, c2 = mesh.curvature_tria(smoothit=10)
#     aniso_mat = np.empty((mesh.t.shape[0], 2))
#     aniso_mat[:, 0] = np.exp(rho_tri * np.abs(c1))
#     aniso_mat[:, 1] = np.exp(rho_tri * np.abs(c2))
    
#     c_s = CMEAN * np.sqrt(aniso_mat)
#     if np.min(c_s) < 0.01 or np.max(c_s) > 300:
#         raise ValueError("c_s values should be in the range [0.01, 300] m/s")

class HeteroSolver(Solver):
    """
    Class to solve the heterogeneous Helmholtz equation on a surface mesh.
    """
    def __init__(self, surf, hmap=None, medmask=None, alpha=1.0, sigma=0, method="hetero", 
                 scale_method="norm", verbose=False, **lapy_kwargs):  
        """
        Initialize the HeteroSolver object.

        Parameters
        ----------
        surf : str
            Path to the surface template file or BSPolyData. 
            See `brainspace.mesh.mesh_io.read_surface` for acceptable file types.
        hmap : np.ndarray, optional
            Array of the heterogeneity map, by default None. If None, hmap will be set to an array
            of ones of length (n_vertices,) and this will results in homogeneous modes.
            If hmap is a numpy array, then it must contain the same number of elements as the number
            of vertices in `surf`.
        """

        # Initialise surface and convert to TriaMesh object
        surf = _check_surf(surf)
        if medmask is not None:
            surf = mask_points(surf, medmask)
            if hmap is not None:
                hmap = hmap[medmask]
        mesh = TriaMesh(get_points(surf), get_cells(surf))

        # Check and scale heterogeneity map
        hmap = _check_hmap(surf, hmap)
        rho = scale_hmap(hmap, alpha=alpha, method=scale_method, sigma=sigma, verbose=verbose)       

        # Map hmap from vertices to triangles
        rho_tri = mesh.map_vfunc_to_tfunc(rho)

        # Initialise the Solver object
        if method == "hetero":
            super().__init__(mesh, aniso=(0, 0), hetero=rho_tri, **lapy_kwargs)
        elif method == "aniso":
            super().__init__(mesh, aniso=np.tile(rho_tri, (2, 1)).T, hetero=None, **lapy_kwargs)
        else:
            raise ValueError("Method must be either `hetero` or `aniso`")
        
        # Store the parameters
        self.mesh = mesh
        self.rho = rho
        
    def solve(self, k=10, fix_mode1=False, standardise=False):
        """
        Solve for eigenvalues and eigenmodes of the HeteroModes problem.

        Parameters
        ----------
        n : int, optional
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
        evals, emodes = self.eigs(k=k)
        
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
    