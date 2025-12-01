#!/bin/bash

# ========================
# SLURM settings (ignored when run in terminal)
# ========================
# SBATCH --account=kg98
# SBATCH --output=/fs03/kg98/vbarnes/slurm/%A_%a.out
# SBATCH --error=/fs03/kg98/vbarnes/slurm/%A_%a.err
# SBATCH --mail-user=victor.barnes@monash.edu
# SBATCH --mail-type=FAIL,END
# SBATCH --ntasks=1
# SBATCH --time=1-00:00:00
# SBATCH --cpus-per-task=6
# SBATCH --mem-per-cpu=20G
# SBATCH --job-name=OpHum1
# SBATCH --array=0-15  # only used when run via sbatch

# ========================
# USER SETTINGS
# ========================
species="human"      # "human", "marmoset", "macaque"
id=0
evaluation="fit"       # "fit" or "crossval"

# ========================
# Determine if we're running under SLURM or manually
# ========================
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Running in terminal mode (no SLURM job array detected)"
    running_in_slurm=false
else
    running_in_slurm=true
fi

# ========================
# Set HMAP_LABELS by species
# ========================
if [ "$species" == "human" ]; then
    HMAP_LABELS=(None myelinmap thickness sv2a odi ndi genel4PC1 megtimescale eiratio1.2)
elif [ "$species" == "marmoset" ]; then
    if [ "$evaluation" != "fit" ]; then
        echo "Error: Evaluation must be 'fit' for marmoset species."
        exit 1
    fi
    HMAP_LABELS=(None myelinmap thickness nissl)
elif [ "$species" == "macaque" ]; then
    if [ "$evaluation" != "fit" ]; then
        echo "Error: Evaluation must be 'fit' for macaque species."
        exit 1
    fi
    # HMAP_LABELS=(None myelinmap thickness ampa cgp5 damp dpat dpmg exh flum inh kain keta mk80 mod musc oxot pire praz uk14 eiratio)
    HMAP_LABELS=(None myelinmap thickness ampa cgp5 damp dpat flum kain keta mk80 musc oxot pire praz uk14)

else
    echo "Error: Unknown species '$species'"
    exit 1
fi

# ========================
# Load config values from JSON
# ========================
CONFIG_FILE="/fs04/kg98/vbarnes/HeteroModes/results/${species}/model_rest/group/id-${id}/run_config.json"

# Read JSON value or exit if missing
read_json() {
    jq -e -r "$1" "$CONFIG_FILE" 2>/dev/null || { echo "Error: Missing field $1 in run_config.json"; exit 1; }
}
# Read JSON array and convert to space-separated string
read_json_array() {
    jq -e -r "$1 | @sh" "$CONFIG_FILE" 2>/dev/null | tr -d "'" || { echo "Error: Missing field $1 in run_config.json"; exit 1; }
}

nruns=$(read_json '.n_runs')
nsplits=$(read_json '.n_splits')
nmodes=$(read_json '.n_modes')
nsubjs=$(read_json '.n_subjs')
alpha=$(read_json_array '.alpha')
beta=$(read_json_array '.beta')
r=$(read_json_array '.r')
gamma=$(read_json_array '.gamma')
den=$(read_json '.den')
metrics=$(read_json_array '.metrics')
band_freq=$(read_json_array '.band_freq')
scaling=$(read_json '.scaling')
parc=$(read_json '.parc')

# Print variables
echo "id: $id"
echo "nruns: $nruns"
echo "nsplits: $nsplits"
echo "nmodes: $nmodes"
echo "nsubjs: $nsubjs"
echo "njobs: $njobs"
echo "alpha: $alpha"
echo "beta: $beta"
echo "r: $r"
echo "gamma: $gamma"
echo "den: $den"
echo "metrics: $metrics"
echo "band_freq: $band_freq"
echo "evaluation: $evaluation"
echo "scaling: $scaling"
echo "species: $species"
echo "parc: $parc"

# ========================
# Threading control
# ========================
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

# ========================
# Environment setup
# ========================
source /fs03/kg98/vbarnes/miniconda/bin/activate
conda activate HeteroModes_py39
cd /fs04/kg98/vbarnes/HeteroModes/scripts/model_rest

# ========================
# Run jobs
# ========================
if [ "$running_in_slurm" = true ]; then
    # ---------------- SLURM MODE ----------------
    HMAP_LABEL=${HMAP_LABELS[$SLURM_ARRAY_TASK_ID]}
    njobs=$SLURM_CPUS_PER_TASK

    echo "Running SLURM job for HMAP_LABEL: $HMAP_LABEL"
    python3 optimisation_group.py \
        --hmap_label "$HMAP_LABEL" \
        --id "$id" \
        --n_runs "$nruns" \
        --n_modes "$nmodes" \
        --n_splits "$nsplits" \
        --n_subjs "$nsubjs" \
        --n_jobs "$njobs" \
        --alpha $alpha \
        --beta $beta \
        --r $r \
        --gamma $gamma \
        --den "$den" \
        --metrics $metrics \
        --band_freq $band_freq \
        --evaluation "$evaluation" \
        --scaling "$scaling" \
        --species "$species" \
        --parc "$parc"

else
    # ---------------- TERMINAL MODE ----------------
    # export NUMEXPR_MAX_THREADS=10
    # export NUMEXPR_NUM_THREADS=1

    njobs=6  # You can change this default. Make sure you have enough memory as well.
    for HMAP_LABEL in "${HMAP_LABELS[@]}"; do
        echo "Running terminal job for HMAP_LABEL: $HMAP_LABEL"

        python3 optimisation_group.py \
            --hmap_label "$HMAP_LABEL" \
            --id "$id" \
            --n_runs "$nruns" \
            --n_modes "$nmodes" \
            --n_splits "$nsplits" \
            --n_subjs "$nsubjs" \
            --n_jobs "$njobs" \
            --alpha $alpha \
            --beta $beta \
            --r $r \
            --gamma $gamma \
            --den "$den" \
            --metrics $metrics \
            --band_freq $band_freq \
            --evaluation "$evaluation" \
            --scaling "$scaling" \
            --species "$species" \
            --parc "$parc"
    done
fi
