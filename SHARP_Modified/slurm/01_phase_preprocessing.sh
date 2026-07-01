#!/bin/bash
#SBATCH --job-name=sharp_01_preproc
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
# PER-FILE JOB ARRAY: one task per .mat file across SUBDIRS_ALL, all run at once.
#   Size --array to the file count -- easiest via submit_preproc.sh (auto-sizes).
#   Manual override, e.g. 6 files:  sbatch --array=0-5 01_phase_preprocessing.sh
# mail-type=FAIL only: array jobs would otherwise email once PER TASK on END.

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/config.sh"
activate_env

# Build a deterministic, sorted list of "subdir|filebase" for every .mat file
# across the configured subdirs, then pick this task's file by array index.
IFS=',' read -ra DIRS <<< "$SUBDIRS_ALL"
mapfile -t FILES < <(
    for sub in "${DIRS[@]}"; do
        for f in "$INPUT_DIR/$sub/"*.mat; do
            [ -e "$f" ] || continue
            printf '%s|%s\n' "$sub" "$(basename "$f" .mat)"
        done
    done | sort
)

idx="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$idx" -ge "${#FILES[@]}" ]; then
    echo "No file for array index $idx (only ${#FILES[@]} .mat file(s) found) -- nothing to do."
    exit 0
fi
entry="${FILES[$idx]}"
sub="${entry%%|*}"
name="${entry##*|}"

# Args: dir  all_dir  name  nss  ncore(=#antennas)  start_idx
# all_dir=0 -> process ONLY this one file.
echo "=== preprocessing $sub/$name (array task $idx of ${#FILES[@]}) ==="
python -u CSI_phase_sanitization_signal_preprocessing.py \
    "$INPUT_DIR/$sub/" 0 "$name" "$NSS" "${N_RX:-4}" 0
echo "Done."
