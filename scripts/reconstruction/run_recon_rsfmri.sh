#!/bin/bash
#SBATCH --account=kg98
#SBATCH --job-name=rsRec_2
#SBATCH --output=/fs03/kg98/vbarnes/slurm/%A_%a.out
#SBATCH --error=/fs03/kg98/vbarnes/slurm/%A_%a.out
#SBATCH --time=0-12:00:00
# SBATCH --time=0-0:30:00
# SBATCH --qos=shortq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-user=victor.barnes@monash.edu
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --array=0-1 # Adjust this range based on the number of alpha_vals

# hmap_labels=(None myelinmap)
# hmap_label=${hmap_labels[$SLURM_ARRAY_TASK_ID]}
hmap_label="myelinmap"
id=2
alpha_vals=(-1.5 1.5)
alpha=${alpha_vals[$SLURM_ARRAY_TASK_ID]}

CONFIG_FILE="/fs04/kg98/vbarnes/HeteroModes/results/reconstruction/id-${id}/config.json"
n_modes=$(jq -r '.n_modes' "$CONFIG_FILE")
method=$(jq -r '.method' "$CONFIG_FILE")
metric=$(jq -r '.metric' "$CONFIG_FILE")
scaling=$(jq -r '.scaling' "$CONFIG_FILE")
q_norm=$(jq -r '.q_norm' "$CONFIG_FILE")

# print out the variables
echo "hmap_label: $hmap_label"
echo "n_modes: $n_modes"
echo "alpha: $alpha"
echo "method: $method"
echo "metric: $metric"
echo "scaling: $scaling"
echo "q_norm: $q_norm"

# Activate the conda environment and change to the scripts directory
source /fs03/kg98/vbarnes/miniconda/bin/activate
conda activate HeteroModes_py39
cd /fs04/kg98/vbarnes/HeteroModes/scripts/reconstruction

python recon_rsfmri.py \
    --id $id \
    --hmap_label $hmap_label \
    --n_modes $n_modes \
    --alpha $alpha \
    --method $method \
    --metric $metric \
    --n_jobs $SLURM_CPUS_PER_TASK \
    --scaling $scaling \
    --q_norm $q_norm \
