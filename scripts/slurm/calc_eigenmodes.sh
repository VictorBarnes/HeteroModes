#!/bin/bash

#SBATCH --job-name=calcModes
#SBATCH --output=/fs03/kg98/vbarnes/slurm/calc_modes_%j.out
#SBATCH --mail-user=victor.barnes@monash.edu
# SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=kg98
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=4G

echo "Activating virtual environment"
source /fs03/kg98/vbarnes/miniconda/bin/activate
conda activate HeteroModes

cd /fs04/kg98/vbarnes/HeteroModes/scripts

echo "===== Begin running script ====="
python 1_calc_eigenmodes.py
