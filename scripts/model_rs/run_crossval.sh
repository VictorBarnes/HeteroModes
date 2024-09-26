#!/bin/bash
#SBATCH --account=kg98
#SBATCH --job-name=MRS_layer4PC1
#SBATCH --output=/fs03/kg98/vbarnes/slurm/model_rs_crossval_%j.out
#SBATCH --time=0-06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-user=victor.barnes@monash.edu
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

HMAP_LABEL="layer4PC1"
ID=5
# SURF_LH="/fs04/kg98/vbarnes/HeteroModes/data/surfaces/atlas-S1200_space-fsLR_den-32k_surf-midthickness_hemi-L_surface.vtk"
# PARC_LH="/fs04/kg98/vbarnes/HeteroModes/data/parcellations/fsLR_32k_Glasser360-lh.txt"
SCALE_METHOD="zscore"
ANISO_METHOD="hetero"
NRUNS=10
NMODES=500
NSPLITS=5
NSUBJS=384
NJOBS=10
DEN="4k"

# Activate the conda environment and change to the scripts directory
source /fs03/kg98/vbarnes/miniconda/bin/activate
conda activate HeteroModes_py38
cd /fs04/kg98/vbarnes/HeteroModes/scripts/model_rs

python3 model_rs_crossval.py \
    --hmap_label $HMAP_LABEL \
    --id $ID \
    --scale_method $SCALE_METHOD \
    --aniso_method $ANISO_METHOD \
    --n_runs $NRUNS \
    --n_modes $NMODES \
    --n_splits $NSPLITS \
    --n_subjs $NSUBJS \
    --n_jobs $NJOBS \
    --den $DEN \
