#%%

import json
import matplotlib.pyplot as plt
import pandas as pd
from heteromodes.utils import get_project_root

PROJ_DIR = get_project_root()
RESULTS_DIR = PROJ_DIR / "results" / "model_rest" / "human" / "hcp-ep"
ID = 1

hmap_labels = ["None", f"myelinmap"]
hmap_config = PROJ_DIR / "data" / "heteromaps" / "human" / "heteromaps_config.json"
cohorts = ["hc", "scz"]

#%% Load results into dataframe

r_edge_data, alpha_data, r_data = [], [], []
for hmap_label in hmap_labels:
    for cohort in cohorts:
        if hmap_label != "None":
            hmap = f"hcpep-{cohort}-{hmap_label}"
        else:
            hmap = hmap_label
        best_json = f"{RESULTS_DIR}/{cohort}/{ID}/hetero-{hmap}_aniso-None/best.json"
        with open(best_json) as f:
            best = json.load(f)
        r_edge_data.append(best["edge_fc_corr"])
        alpha_data.append(best["alpha"])
        r_data.append(best["r"])

df = pd.DataFrame(
    {
        "hmap": [h for h in hmap_labels for _ in cohorts],
        "cohort": cohorts * len(hmap_labels),
        "edge_fc_corr": r_edge_data,
        "alpha": alpha_data,
        "r": r_data,
    }
)


#%% Plot results (grouped by cohort)
import seaborn as sns

sns.set_style("whitegrid")
fig, axs = plt.subplots(1, 3, figsize=(10, 5))
sns.barplot(x="hmap", y="edge_fc_corr", hue="cohort", data=df, ax=axs[0])
axs[0].set_title("Edge FC correlation")
axs[0].set_xlabel("Heteromap")
axs[0].set_ylabel("r_edge")
axs[0].legend(title="Cohort", loc="lower right")

sns.barplot(x="hmap", y="alpha", hue="cohort", data=df, ax=axs[1])
axs[1].set_title("Alpha")
axs[1].set_xlabel("Heteromap")
axs[1].set_ylabel("Alpha")
axs[1].legend(title="Cohort", loc="lower right")   

sns.barplot(x="hmap", y="r", hue="cohort", data=df, ax=axs[2])
axs[2].set_title("r")
axs[2].set_xlabel("Heteromap")
axs[2].set_ylabel("r")
axs[2].legend(title="Cohort", loc="lower right")

plt.tight_layout()

plt.show()
