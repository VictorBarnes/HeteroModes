#!/bin/bash
#SBATCH --account=kg98
#SBATCH --job-name=simFCopt
#SBATCH --output=/fs03/kg98/vbarnes/slurm/simulateFC_optimise_%A_%a.out
#SBATCH --time=1:00:00
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-user=victor.barnes@monash.edu
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --array=1-222

alpha_beta_file="/fs04/kg98/vbarnes/HeteroModes/data/alpha_beta_valid.csv"
config_file="/fs04/kg98/vbarnes/HeteroModes/scripts/config.json"
hetero_label="myelinmap"
n_runs=50

# Skip the header row and select the row based on the SLURM_ARRAY_TASK_ID
row=$(tail -n +2 "${alpha_beta_file}" | awk -F',' "NR==${SLURM_ARRAY_TASK_ID} {print}")
alpha=$(echo "${row}" | awk -F',' '{print $1}')
beta=$(echo "${row}" | awk -F',' '{print $2}')

echo -e "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID} | alpha: ${alpha} | beta: ${beta}"
echo -e "============================================================================="

echo "Loading matlab"
cd /fs04/kg98/vbarnes/HeteroModes/scripts
module load matlab/r2022b

echo "===== Begin running script ====="
matlab -nodisplay -nosplash -r "simulateFC('${hetero_label}', ${alpha}, ${beta}, '${config_file}', ${n_runs}); exit;"
