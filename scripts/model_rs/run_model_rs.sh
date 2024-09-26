#!/bin/bash
#SBATCH --account=kg98
#SBATCH --job-name=MRS_my55
#SBATCH --output=/fs03/kg98/vbarnes/slurm/model_rs_%A_%a.out
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-user=victor.barnes@monash.edu
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --array=1-21

# Define variables for arguments to make the script more readable
HMAP_LABEL="myelinmap"
ID=55
SURF_LH="/fs04/kg98/vbarnes/HeteroModes/data/surfaces/atlas-S1200_space-fsLR_den-32k_surf-midthickness_hemi-L_surface.vtk"
SURF_RH=None
PARC_NAME="Glasser360"
NRUNS=20
NMODES=500
SCALE_METHOD="zscore"
ANISO_METHOD="hetero"
SAVE_ALL=false
NSUBJS=50
TSTEP=90.0

# nmodes_list=(100 200 300 400 500 600 700 800 900 1000)
# NMODES=${nmodes_list[$SLURM_ARRAY_TASK_ID-1]}
# echo "nmodes: $NMODES"
# tstep_list=(10.0 30.0 60.0 90.0 120.0 180.0)
# TSTEP=${tstep_list[$SLURM_ARRAY_TASK_ID-1]}
# echo "tstep: $TSTEP"

# Load parameter combinations file
combs_file="/fs04/kg98/vbarnes/HeteroModes/results/model_rs/${HMAP_LABEL}/id-${ID}/csParamCombs.csv"
# Skip the header row and select the row based on the SLURM_ARRAY_TASK_ID
row=$(tail -n +2 "${combs_file}" | awk -F',' "NR==${SLURM_ARRAY_TASK_ID} {print}")
alpha=$(echo "${row}" | awk -F',' '{print $1}')
# beta=$(echo "${row}" | awk -F',' '{print $2}')
# r=$(echo "${row}" | awk -F',' '{print $3}')
# gamma=$(echo "${row}" | awk -F',' '{print $4}')

# Activate the conda environment and change to the scripts directory
source /fs03/kg98/vbarnes/miniconda/bin/activate
conda activate HeteroModes_py38
cd /fs04/kg98/vbarnes/HeteroModes/scripts/model_rs

python3 model_rs.py \
    --method "run_model" \
    --surf_lh "$SURF_LH" \
    $( [ "$SURF_RH" != None ] && echo "--surf_rh '$SURF_RH'" ) \
    --parc_name "$PARC_NAME" \
    --hmap_label "$HMAP_LABEL" \
    --alpha $alpha \
    --id $ID \
    --nruns $NRUNS \
    --nmodes $NMODES \
    --aniso_method $ANISO_METHOD \
    --scale_method $SCALE_METHOD \
    --tstep $TSTEP \
    --nsubjs $NSUBJS \
    --slurm_id $SLURM_ARRAY_TASK_ID \
    $( [ "$SAVE_ALL" = true ] && echo "--save_all" )
