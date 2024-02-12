import pandas as pd
import numpy as np
import itertools

# Define the values for the pairs
alpha = np.arange(0.2, 3.2, 0.2)
beta = np.sort(np.append(np.arange(-10.0, 11.0, 1.0), [-0.5, 0.5]))

# Generate all possible pairs
combs = list(itertools.product(alpha, beta))
combs = [(round(a, 1), round(b, 1)) for a, b in combs]

# Create a DataFrame with the pairs and save
combs_df = pd.DataFrame(combs, columns=['alpha', 'beta'])
combs_df.to_csv("data/alpha-beta-combs_all.csv")
