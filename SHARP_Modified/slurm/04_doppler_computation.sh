#!/bin/bash
#SBATCH --job-name=sharp_04_doppler
#SBATCH --partition=shared
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=danielle.a.nelson@stonybrook.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# PER-SUBDIR JOB ARRAY: one task per capture folder in SUBDIRS_ALL.
#   Size --array to the subdir count -- easiest via submit_doppler.sh (auto-sizes).
#   Manual override, e.g. 3 subdirs:  sbatch --array=0-2 04_doppler_computation.sh
# (Doppler is FFT-light, so per-subdir granularity is plenty.)

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/config.sh"
activate_env

# Pick this task's subdir from SUBDIRS_ALL by array index.
IFS=',' read -ra DIRS <<< "$SUBDIRS_ALL"
idx="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$idx" -ge "${#DIRS[@]}" ]; then
    echo "No subdir for array index $idx (only ${#DIRS[@]} in SUBDIRS_ALL) -- nothing to do."
    exit 0
fi
sub="${DIRS[$idx]}"

# Args: dir  subdirs  dir_doppler  start  end  sample_length  sliding  noise_level
# (optional --bandwidth / --sub_band default to 80 MHz full band)
echo "=== Doppler computation $sub (array task $idx of ${#DIRS[@]}) ==="
python -u CSI_doppler_computation.py \
    "$PROCESSED_PHASE_DIR" "$sub" "$DOPPLER_DIR" \
    "$DOPPLER_START" "$DOPPLER_END" "$SAMPLE_LENGTH" "$SLIDING" "$NOISE_LEVEL"
echo "Done."
