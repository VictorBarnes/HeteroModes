#!/bin/bash
#SBATCH --account=kg98
#SBATCH --job-name=calcEMs
#SBATCH --output=/fs03/kg98/vbarnes/slurm/calc_modes_%j.out
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-user=victor.barnes@monash.edu
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --array=1-2

echo -e "${SLURM_ARRAY_TASK_ID}"

hetero_label="SAaxis"
alpha_beta_file="/fs04/kg98/vbarnes/HeteroModes/data/alpha-beta-combs_all.csv"

# Skip the header row and select the row based on the SLURM_ARRAY_TASK_ID
row=$(tail -n +2 "${alpha_beta_file}" | awk -F',' "NR==${SLURM_ARRAY_TASK_ID} {print}")
alpha=$(echo "${row}" | awk -F',' '{print $1}')
beta=$(echo "${row}" | awk -F',' '{print $2}')

echo "Activating conda environment"
conda activate HeteroModes

cd /fs04/kg98/vbarnes/HeteroModes

echo "===== Begin running script ====="
python scripts/1_calc_eigenmodes.py -c ${config_file} -a ${alpha} -b ${beta}
