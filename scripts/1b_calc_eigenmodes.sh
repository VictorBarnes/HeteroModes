#!/bin/bash
#SBATCH --account=kg98
#SBATCH --job-name=calcEMs
#SBATCH --output=/fs03/kg98/vbarnes/slurm/calcModes_%A_%a.out
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-user=victor.barnes@monash.edu
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --array=1-660

echo -e "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"

hetero_label="myelinmap"
alpha_beta_file="/fs04/kg98/vbarnes/HeteroModes/data/csParamCombs_all.csv"
config_file="/fs04/kg98/vbarnes/HeteroModes/scripts/config.json"

# Skip the header row and select the row based on the SLURM_ARRAY_TASK_ID
row=$(tail -n +2 "${alpha_beta_file}" | awk -F',' "NR==${SLURM_ARRAY_TASK_ID} {print}")
alpha=$(echo "${row}" | awk -F',' '{print $1}')
beta=$(echo "${row}" | awk -F',' '{print $2}')
echo -e "hetero: ${hetero_label} | alpha: ${alpha} | beta ${beta}"

echo "Activating conda environment"
source activate /fs03/kg98/vbarnes/miniconda/conda/envs/HeteroModes

cd /fs04/kg98/vbarnes/HeteroModes/scripts

echo "===== Begin running script ====="
python3 -c "from calc_hetero_modes import calc_modes; calc_modes('${config_file}', '${hetero_label}', ${alpha}, ${beta})"
