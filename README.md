# Regional Heterogeneity Shapes Macroscopic Wave Dynamics

This repository contains data, code and results for the manuscript: (insert link here) by Victor Barnes et al. We investigated the role of regional heterogeneity in shaping macroscopic wave dynamics across species using a wave model based on Neural Field Theory (NFT). 

## Installation
This repository works with Python 3.10 and above. It can be installed by cloning the repository and installing the required packages listed in `pyproject.toml`. This should take less than a minute on a standard machine.

```bash
git clone
cd HeteroModes
pip install .
```

For development with optional dependencies:
```bash
pip install -e ".[plotting,dev]"
```

## Citing
If you use this code in your work, please cite the following manuscript: (insert citation here)

## Usage
The main script to run the main analyses in the manuscript is `optimisation.py`. It has several configurable parameters at the top of the script. To run a simple optimisation (that takes less than 5 minutes), you could use the following command from `scripts/model_rest/`:

```bash
python optimisation.py --species human --id 0 --hmap_label myelinmap  --alpha -3 3 0.5 --n_runs 1 --evaluation fit --n_subj 10 --metrics edge_fc_corr node_fc_corr 
```

This command runs an optimisation for the human model, with an id of 1 (used to index different optimization runs), using the myelin map as the heterogeneity map, with alpha values of ranging from [-3, 3] with a step of 0.1, with 10 runs per model, evaluating the model by fitting it to 10 subjects using edge-level and node-level functional connectivity correlation as metrics. 

Optimising across multiple parameters can also be done. For example, to optimise across both alpha and r parameters, you could use the following command:

```bash
python optimisation.py --species human --id 2 --hmap_label myelinmap  --alpha -3 3 1 --r 10 50 10 --n_runs 1 --evaluation fit --n_subj 10 --metrics edge_fc_corr node_fc_corr 
```

To run a parcellated model, you can add the `--parc` argument. Note that `--den` must be set to `32k` when using parcellations, for maximal accuracy and comparability with empirical data. For example, to run the same optimisation as above but using the HCP-MMP1 parcellation, you would use the following. Note that you should use a different `--id` for each optimisation run to avoid overwriting results:

```bash
python optimisation.py --species human --id 1 --parc hcpmmp1 --den 32k --hmap_label myelinmap  --alpha -3 3 0.5 --n_runs 1 --evaluation fit --n_subj 10 --metrics edge_fc_corr node_fc_corr 
```

### Important usage notes
In the examples above we don't optimize using FCD KS metric as it is very computationally intensive. If you wish to include this metric, please ensure you have access to a high-performance computing cluster. 

License information
-------------------
This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (``cc-by-nc-sa``). See the `LICENSE <LICENCE-CC-BY-NC-SA-4.0.md>`_ file for details.
