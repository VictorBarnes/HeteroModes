import os
import itertools
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from heteromodes.solver import HeteroSolver
from heteromodes.utils import load_hmap, scale_hmap


load_dotenv()
#def gen_param_combs(hmap_label, alpha=None, beta=None, r=None, gamma=None, medmask=None, scale_method="norm"):
def gen_param_combs(hmap_label, medmask=None, scale_method="norm"):
        """
        Generate valid combinations of parameters for calculating heterogeneous modes.

        Parameters
        ----------
        scale_method : str, optional
            The scaling method for the heterogeneity map, by default "norm".
        medmask : ndarray, optional
            The medial mask, by default None.

        Raises
        ------
        ValueError
            If the hetero file type is not supported.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the valid combinations of parameters.
        """

        # Load surface and hetero map
        hmap = load_hmap(hmap_label=hmap_label, medmask=medmask)

        # alpha = alpha if alpha is not None else np.arange(-10, 10, 0.5)
        # beta = beta if beta is not None else np.arange(0.5, 3.5, 0.5)
        # r = r if r is not None else np.arange(10.0, 110.0, 10.0)
        # gamma = gamma if gamma is not None else [0.116]

        # Generate the parameter combinations
        # alpha = np.arange(-2.0, 2.1, 0.1)
        # alpha = alpha[~np.isclose(alpha, 0)]    # Drop alpha = 0
        beta = [1.0] #np.arange(0.5, 3.5, 0.5)
        r = [28.9] #np.arange(10.0, 110.0, 10.0) 
        gamma = [0.116]
        combs = list(itertools.product(alpha, beta, r, gamma))

        # Check if the combinations are valid
        valid_combs = []
        for alpha, beta, r, gamma in combs:
            try:
                _ = scale_hmap(hmap, alpha, beta, r, gamma, method=scale_method, verbose=False)
                # _ = HeteroSolver(surf=os.getenv("SURF_LH"), hmap=hmap_scale, medmask=medmask, method="aniso")
            except ValueError:
                continue
            valid_combs.append((alpha, beta, r, gamma))
        print(f"Number of valid combinations for {hmap_label} modes: {len(valid_combs)}")

        # Create a DataFrame with the pairs and save
        valid_combs_df = pd.DataFrame(valid_combs, columns=['alpha', 'beta', 'r', 'gamma'])

        return valid_combs_df



