#!/bin/bash
#SBATCH --job-name=sharp_09_grid
#SBATCH --partition=shared
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=danielle.a.nelson@stonybrook.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# NOTE: partition "shared" is CPU-only -- training works but is slow (~30-90 min
# per Keras config). If LEAP2 has a GPU partition (`sinfo` to check), switch:
#   #SBATCH --partition=<gpu-partition>
#   #SBATCH --gres=gpu:1
#
# Hyperparameter grid over the SHARP models as a SLURM ARRAY: task N runs line N
# of slurm/grid_configs.txt (regenerate with generate_grid_configs.py).
# Submit with the array sized to the config file, throttled to be polite:
#   sbatch --array=1-$(wc -l < slurm/grid_configs.txt)%20 slurm/09_train_grid.sh
#
# Each task trains on SUBDIRS_TRAIN and evaluates cross-domain on S7a, inside its
# OWN working directory gridruns/<jobid>_<taskid>/ -- CSI_network.py names some
# outputs by model only, so concurrent tasks of the same model would otherwise
# overwrite each other. Collect everything afterwards with 10_collect_grid_results.sh.

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/config.sh"
activate_env

# Training-network dimensions (match run_models_dgx.sh on the DGX):
# feature_length=100 Doppler bins, sample_length=WINDOW_LENGTH=340 steps.
FEATURE_LENGTH=100
GRID_SUBDIRS_TEST="S7a"   # grid scored on S7a only (config.sh SUBDIRS_TEST = all 9 domains)

CONFIG_FILE="$PYTHON_CODE_DIR/slurm/grid_configs.txt"
LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CONFIG_FILE")"
[ -n "$LINE" ] || { echo "ERROR: no config line ${SLURM_ARRAY_TASK_ID} in $CONFIG_FILE" >&2; exit 1; }
IFS='|' read -r MODEL LR DO HP <<< "$LINE"
echo "task ${SLURM_ARRAY_TASK_ID}: model=$MODEL lr=$LR dropout=$DO hparams=$HP"

# Isolated per-task workdir; the package tree is found via PYTHONPATH.
WORKDIR="$PYTHON_CODE_DIR/gridruns/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"
echo "$LINE" > config.txt
export PYTHONPATH="$PYTHON_CODE_DIR"

TRAIN_ARGS=(--model "$MODEL" --bandwidth 80 --sub_band 1 --hparams "$HP")
[ "$LR" != "-" ] && TRAIN_ARGS+=(--learning_rate "$LR")
[ "$DO" != "-" ] && TRAIN_ARGS+=(--dropout "$DO")

python -m SHARP_Modified.Python_code.CSI_network \
    "$DOPPLER_DIR" "$SUBDIRS_TRAIN" "$FEATURE_LENGTH" "$WINDOW_LENGTH" 1 \
    32 "$N_TOT" grid "$ACTIVITIES" "${TRAIN_ARGS[@]}"

python -m SHARP_Modified.Python_code.CSI_network_test \
    "$DOPPLER_DIR" "$GRID_SUBDIRS_TEST" "$FEATURE_LENGTH" "$WINDOW_LENGTH" 1 \
    32 "$N_TOT" grid "$ACTIVITIES" --model "$MODEL" --bandwidth 80 --sub_band 1

# The tf.data cache is ~6 GB per task -- delete it as soon as the run succeeds
# so a 100+-task array doesn't chew through the GPFS quota.
rm -rf cache
echo "task ${SLURM_ARRAY_TASK_ID} done."
