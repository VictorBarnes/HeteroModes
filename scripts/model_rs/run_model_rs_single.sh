#!/bin/bash
#SBATCH --account=kg98
#SBATCH --job-name=MRS_my22
#SBATCH --output=/fs03/kg98/vbarnes/slurm/model_rs_%j.out
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-user=victor.barnes@monash.edu
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

# Define variables for arguments to make the script more readable
HMAP_LABEL=None
ID=4
SURF_LH="/fs04/kg98/vbarnes/HeteroModes/data/surfaces/atlas-S1200_space-fsLR_den-32k_surf-midthickness_hemi-L_surface.vtk"
SURF_RH=None
PARC_NAME="Glasser360"
NRUNS=20
NMODES=500
SCALE_METHOD="zscore"
SAVE_ALL=true
ALPHA=1.0
BETA=1.0

# Activate the conda environment and change to the scripts directory
source /fs03/kg98/vbarnes/miniconda/bin/activate
conda activate HeteroModes_py38
cd /fs04/kg98/vbarnes/HeteroModes/scripts/model_rs

python3 model_rs.py \
    --method "run_model" \
    --id $ID \
    --surf_lh "$SURF_LH" \
    $( [ "$SURF_RH" != None ] && echo "--surf_rh '$SURF_RH'" ) \
    --parc_name "$PARC_NAME" \
    $( [ "$HMAP_LABEL" != None ] && echo "--hmap_label $HMAP_LABEL" ) \
    --alpha $ALPHA \
    --beta $BETA \
    --nruns $NRUNS \
    --nmodes $NMODES \
    --scale_method $SCALE_METHOD \
    $( [ "$SAVE_ALL" = true ] && echo "--save_all" )
