import pandas as pd
import numpy as np
import itertools

# Define the values for the pairs
alpha = np.sort(np.concatenate((np.arange(0.2, 3.2, 0.2), 
                                np.arange(0.2, 3.2, 0.2) * -1)))
beta = np.sort(np.concatenate((np.arange(1.0, 11.0, 1.0), 
                               np.arange(1.0, 11.0, 1.0) * -1, 
                               [-0.5, 0.5])))

# Generate all possible pairs
combs = list(itertools.product(alpha, beta))
combs = [(round(a, 1), round(b, 1)) for a, b in combs]

# Create a DataFrame with the pairs and save
combs_df = pd.DataFrame(combs, columns=['alpha', 'beta'])
combs_df.to_csv("data/csParamCombs_all.csv", index=None)
