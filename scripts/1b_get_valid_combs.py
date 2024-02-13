# Get valid alpha-beta combinations from generated eigenmodes

import os
import json
import pandas as pd


with open("scripts/config.json", encoding="UTF-8") as f:
        config = json.load(f)
emode_dir = config["emode_dir"]

combs_valid = []
combs_all = pd.read_csv("data/csParamCombs_all.csv")
for i in range(combs_all.shape[0]):
    alpha = combs_all.loc[i, "alpha"]
    beta = combs_all.loc[i, "beta"]
    filename = (f"hetero-{config['hetero_label']}_atlas-{config['atlas']}_"
                f"space-{config['space']}_den-{config['den']}_surf-{config['surf']}_"
                f"hemi-{config['hemi']}_n-{config['n_modes']}_alpha-{alpha}_beta-{beta}_"
                f"maskMed-{config['mask_medial']}_emodes.txt")
    filepath = os.path.join(emode_dir, filename)
    if os.path.isfile(filepath):
            combs_valid.append((alpha, beta))

# Create a DataFrame with the pairs and save
alphabeta_valid_df = pd.DataFrame(combs_valid, columns=['alpha', 'beta'])
alphabeta_valid_df.to_csv(f"data/hetero-{config['hetero_label']}_csParamCombs_valid.csv", 
                          index=None)
