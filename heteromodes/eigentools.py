import numpy as np


def calc_eigendecomposition(data, evecs, method='matrix', B=None):
    """
    Calculate the eigendecomposition of the given data using the specified method.

    Parameters
    ----------
    data : array-like
        The input data for the eigendecomposition.
    evecs : array-like
        The eigenmodes used for the eigendecomposition.
    method : str, optional
        The method used for the eigendecomposition. Default is 'matrix'.
    B : array-like, optional
        The mass matrix used for the eigendecomposition when method is 'orthonormal'.

    Returns
    -------
    array-like
        The beta coefficients obtained from the eigendecomposition.
    """

    if not np.allclose(evecs[:, 0], np.full_like(evecs[:, 0], evecs[0, 0])):
        print("Warning: `evecs` should contain a constant eigenvector.")

    # Solve the linear system to get the beta coefficients
    if method == 'matrix':
        beta_coeffs = np.linalg.solve((evecs.T @ evecs), (evecs.T @ data))
    elif method == 'orthonormal':
        if B is None:
            raise ValueError("B must be specified when method is 'orthonormal'")
        beta_coeffs = evecs.T @ B @ data
    else:
        raise ValueError("Invalid method; must be 'matrix', or 'orthonormal'")
    
    return beta_coeffs


def calc_eigenreconstruction(data, evecs, method='matrix', B=None):
    """
    Calculate the eigen reconstruction of the given data using the provided eigen modes.

    Parameters
    ----------
    data : array-like
        The input data array of shape (n_verts, n_data), where n_verts is the number of vertices and 
        n_data is the number of data points.
    emodes : array-like
        The eigen modes array of shape (n_verts, n_modes), where n_modes is the number of eigen 
        modes.
    method : str, optional
        The method used for eigen decomposition. Defaults to 'matrix'.
    B : array-like, optional
        The matrix B used in the eigen decomposition. Defaults to None.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        - beta_coeffs : list
            A list of beta coefficients calculated for each eigenmode.
        - recon : numpy.ndarray
            The reconstructed data array of shape (n_verts, n_modes, n_data).
        - corr_coeffs : numpy.ndarray
            The correlation coefficients array of shape (n_modes, n_data).
    """

    # Get the number of vertices and data points
    n_verts, n_data = np.shape(data)
    _, n_modes = np.shape(evecs)

    # Initialize the output arrays
    beta_coeffs = []
    recon = np.empty((n_verts, n_modes, n_data))
    corr_coeffs = np.empty((n_modes, n_data))

    # Perform the eigenreconstruction
    for i in range(n_modes):
        beta_coeffs.append(calc_eigendecomposition(data, evecs[:, :i+1], method=method, B=B))
        recon[:, i, :] = evecs[:, :i+1] @ beta_coeffs[i]
        corr_coeffs[i, :] = [np.corrcoef(data[:, j], np.squeeze(recon[:, i, j]))[0, 1] for j in range(n_data)]

    return beta_coeffs, recon, corr_coeffs
