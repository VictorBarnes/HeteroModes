import numpy as np

def calc_eigendecomposition(data, evecs, method='orthogonal', mass=None):
    """
    Calculate the eigen-decomposition of the given data using the specified method.

    Parameters
    ----------
    data : array-like
        The input data for the eigen-decomposition.
    evecs : array-like
        The eigenmodes used for the eigen-decomposition.
    method : str, optional
        The method used for the eigen-decomposition. Default is 'matrix'.
    mass : array-like, optional
        The mass matrix used for the eigen-decomposition when method is 'orthogonal'. Default is 
        None.

    Returns
    -------
    beta : numpy.ndarray of shape (n_modes, n_data)
        The beta coefficients obtained from the eigen-decomposition.
    """

    if not np.allclose(evecs[:, 0], np.full_like(evecs[:, 0], evecs[0, 0])):
        print("Warning: `evecs` should contain a constant eigenvector.")

    # Solve the linear system to get the beta coefficients
    if method == 'matrix':
        beta = np.linalg.solve((evecs.T @ evecs), (evecs.T @ data))
    elif method == 'orthogonal':
        if mass is None:
            raise ValueError("B must be specified when method is 'orthogonal'")

        beta = evecs.T @ mass @ data
    else:
        raise ValueError("Invalid method; must be 'matrix' or 'orthogonal'.")

    return beta

def calc_eigenreconstruction(data, evecs, method='orthogonal', modesq=None, mass=None, data_type="maps", metric="pearsonr", return_all=False):
    """
    Calculate the eigen-reconstruction of the given data using the provided eigenmodes.

    Parameters
    ----------
    data : array-like
        The input data array of shape (n_verts, n_data), where n_verts is the number of vertices and 
        n_data is the number of data points.
    evecs : array-like
        The eigenmodes array of shape (n_verts, n_modes), where n_modes is the number of eigenmodes.
    method : str, optional
        The method used for eigen-decomposition. Default is 'matrix'.
    modesq : array-like, optional
        The sequence of modes to be used for reconstruction. Default is None, which uses all modes.
    mass : array-like, optional
        The mass matrix used for the eigen-decomposition when method is 'orthogonal'. Default is None.
    data_type : str, optional
        The type of data, either "maps" or "timeseries". Default is "maps".
    metric : str, optional
        The metric used for calculating reconstruction accuracy. Default is "pearsonr".
    return_all : bool, optional
        Whether to return the reconstructed timepoints when data_type is "timeseries". Default is False.

    Returns
    -------
    beta : list of numpy.ndarray
        A list of beta coefficients calculated for each mode.
    recon : numpy.ndarray
        The reconstructed data array of shape (n_verts, nq, n_data).
    recon_score : numpy.ndarray
        The correlation coefficients array of shape (nq, n_data).
    fc_recon : numpy.ndarray, optional
        The functional connectivity reconstructed data array of shape (n_verts, n_verts, nq). 
        Returned only if data_type is "timeseries".
    fc_recon_score : numpy.ndarray, optional
        The functional connectivity correlation coefficients array of shape (nq,). Returned only if 
        data_type is "timeseries".
    """

    if np.shape(data)[0] != np.shape(evecs)[0]:
        raise ValueError("The number of vertices in `data` and `evecs` must be the same.")
    if method == "orthogonal" and mass is None:
        raise ValueError("B must be specified when method is 'orthogonal'")
    if metric not in ["pearsonr", "mse"]:
        raise ValueError("Invalid metric; must be 'pearsonr' or 'mse'.")
    if data_type not in ["maps", "timeseries"]:
        raise ValueError("Invalid data_type; must be 'maps' or 'timeseries'.")
    
    # Get the number of vertices and data points
    n_verts, n_data = np.shape(data)

    if modesq is None:
        # Use all modes if not specified (except the first constant mode)
        modesq = np.arange(1, np.shape(evecs)[1] + 1)
    nq = len(modesq)

    # If data is timeseries, calculate the FC of the original data and initialize the output arrays
    if data_type == "timeseries":
        triu_inds = np.triu_indices(n_verts, k=1)
        fc_orig = np.corrcoef(data)[triu_inds]
        fc_recon = np.empty((n_verts, n_verts, nq))
        fc_recon_score = np.empty((nq,))

    # If method is 'orthogonal', then beta coefficients can be calucated at once
    if method == "orthogonal":
        tmp = calc_eigendecomposition(data, evecs[:, :np.max(modesq)], method=method, mass=mass)
        beta = [tmp[:mq] for mq in modesq]
    else:
        beta = [None] * nq

    # Initialize the output arrays
    recon = np.empty((n_verts, nq, n_data))
    recon_score = np.empty((nq, n_data))
    for i in range(nq):
        if method != "orthogonal":
            beta[i] = calc_eigendecomposition(data, evecs[:, :modesq[i]], method=method, mass=mass)

        # Reconstruct the data using the beta coefficients
        recon[:, i, :] = evecs[:, :modesq[i]] @ beta[i]
        if data_type == "maps":
            # Avoid division by zero
            if modesq[i] == 1:  
                recon_score[i, :] = 0
            else:
                if metric == "pearsonr":
                    recon_score[i, :] = [np.corrcoef(data[:, j], np.squeeze(recon[:, i, j]))[0, 1] for j in range(n_data)]
                elif metric == "mse":
                    recon_score[i, :] = np.mean((data - np.squeeze(recon[:, i, :]))**2, axis=0)
                else:
                    raise ValueError("Invalid metric; must be 'pearsonr' or 'mse'")

        # Calculate FC of the reconstructed data
        elif data_type == "timeseries":
            # Calculate the functional connectivity of the reconstructed data
            fc_recon[:, :, i] = np.corrcoef(recon[:, i, :])
            
            # Avoid division by zero
            if modesq[i] == 1:
                if return_all:
                    recon_score[i, :] = 0
                fc_recon_score[i] = 0
            else:
                if return_all:
                    recon_score[i, :] = [np.corrcoef(data[:, j], np.squeeze(recon[:, i, j]))[0, 1] for j in range(n_data)]

                if metric == "pearsonr":
                    fc_recon_score[i] = np.corrcoef(
                        np.arctanh(fc_orig), 
                        np.arctanh(np.squeeze(fc_recon[:, :, i][triu_inds]))
                    )[0, 1]
                elif metric == "mse":
                    fc_recon_score[i] = np.mean((fc_orig - np.squeeze(fc_recon[:, :, i][triu_inds]))**2)
                else:
                    raise ValueError("Invalid metric; must be 'pearsonr' or 'mse'")
                
    if data_type == "timeseries":
        if return_all:
            return beta, recon, recon_score, fc_recon, fc_recon_score
        else:
            return beta, recon, fc_recon, fc_recon_score
    else:
        return beta, recon, recon_score

def calc_eigenreconstruction_topN(data, evecs, mass=None, n=20):
    """
    Perform eigen-reconstruction using the top N eigenmodes with the largest absolute weights.

    Parameters
    ----------
    data : array-like
        The input data array of shape (n_verts, n_data), where n_verts is the number of vertices and 
        n_data is the number of data points.
    evecs : array-like
        The eigenmodes array of shape (n_verts, n_modes), where n_modes is the number of eigenmodes.
    mass : array-like, optional
        The mass matrix used for the eigen-decomposition when method is 'orthogonal'. Default is None.
    n : int, optional
        The number of top eigenmodes to use for reconstruction. Default is 20.

    Returns
    -------
    beta : numpy.ndarray
        The beta coefficients array of shape (n_modes, n_data) obtained from the eigen-decomposition.
    recon : numpy.ndarray
        The reconstructed data array of shape (n_verts, n, n_data), where n is the number of top modes.
    recon_score : numpy.ndarray
        The correlation coefficients array of shape (n, n_data), representing the reconstruction accuracy.
    top_mode_ids : numpy.ndarray
        The indices of the top N eigenmodes for each data point, of shape (n, n_data).
    """
    
    # Get the number of vertices and data points
    if len(np.shape(data)) == 1:
        data = np.expand_dims(data, axis=1)
    n_verts, n_data = np.shape(data)

    # Compute full decomposition coefficients. This only works for "orthogonal" for now
    beta = calc_eigendecomposition(data, evecs, method="orthogonal", mass=mass)
    # Get idxs of top N modes based on the largest absolute weights (skip the first mode)
    top_mode_ids = np.argsort(np.abs(beta), axis=0)[-n:, :][::-1]  # Select top N modes

    # Get corresponding eigenmodes and coefficients
    recon = np.empty((n_verts, n, n_data))
    recon_score = np.empty((n, n_data))
    for i in range(n_data):
        for j in range(n):
            recon[:, j, i] = evecs[:, top_mode_ids[:j+1, i]] @ beta[top_mode_ids[:j+1, i], i]
            if j == 0 and top_mode_ids[j, i] == 0:  # Avoid division by zero
                recon_score[j, i] = 0
            else:
                recon_score[j, i] = np.corrcoef(data[:, i], recon[:, j, i])[0, 1]

    return beta, recon, recon_score, top_mode_ids
