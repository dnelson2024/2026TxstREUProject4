#!/bin/bash
#SBATCH --job-name=sharp_03_recon
#SBATCH --partition=shared
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=danielle.a.nelson@stonybrook.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=5-00:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# PER-FILE JOB ARRAY: one task per Tr_vector_*.txt (from stage 02), all at once.
#   Size --array to the Tr-file count -- easiest via submit_recon.sh (auto-sizes).
#   Manual override, e.g. 24 files:  sbatch --array=0-23 03_phase_reconstruction.sh
# mail-type=FAIL only: avoid one email per array task on END.

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/config.sh"
activate_env

# Build a sorted list of Tr-file base names (no .txt) in PHASE_PROCESSING_DIR,
# then pick this task's file by array index. (PHASE_PROCESSING_DIR ends with '/'.)
mapfile -t FILES < <(
    for f in "$PHASE_PROCESSING_DIR"Tr*.txt; do
        [ -e "$f" ] || continue
        basename "$f" .txt
    done | sort
)

idx="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$idx" -ge "${#FILES[@]}" ]; then
    echo "No file for array index $idx (only ${#FILES[@]} Tr file(s) found) -- nothing to do."
    exit 0
fi
name="${FILES[$idx]}"

# Args: dir  dir_save  nss  ncore  start_idx  end_idx  --name <single Tr file>
echo "=== reconstruction $name (array task $idx of ${#FILES[@]}) ==="
python -u CSI_phase_sanitization_signal_reconstruction.py \
    "$PHASE_PROCESSING_DIR" "$PROCESSED_PHASE_DIR" \
    "$NSS" "${N_RX:-4}" 0 -1 --name "$name"
echo "Done."
