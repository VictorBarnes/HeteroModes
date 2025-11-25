# Regional Heterogeneity Shapes Macroscopic Wave Dynamics

This repository contains data, code and results for the manuscript: (insert link here) by Victor Barnes et al. We investigated the role of regional heterogeneity in shaping macroscopic wave dynamics across species using a wave model based on Neural Field Theory (NFT). 

## Installation
This repository works with Python 3.10 and above. It can be installed by cloning the repository and installing the required packages listed in `pyproject.toml`. 

```bash
git clone
cd HeteroModes
pip install .
```

## Citing
If you use this code in your work, please cite the following manuscript: (insert citation here)

## Usage
The main script to run the main analyses in the manuscript is `optimisation.py`. It has several configurable parameters at the top of the script. To run a simple optimisation, you could use the following command from `scripts/model_rest/`:

```bash
python optimisation.py --species human --id 1 --hmap_label myelinmap  --alpha -3 3 0.1 --n_runs 1 --evaluation fit --n_subj 10 --metrics edge_fc_corr node_fc_corr 
```

This command runs an optimisation for the human model, with an id of 1 (used to index different optimization runs), using the myelin map as the heterogeneity map, with alpha values of ranging from [-3, 3] with a step of 0.1, with 10 runs per model, evaluating the model by fitting it to 10 subjects using edge-level and node-level functional connectivity correlation as metrics.
