#!/bin/bash
#SBATCH --job-name=sharp_02_hest
#SBATCH --partition=shared
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=danielle.a.nelson@stonybrook.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=5-00:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# This is the slowest stage (Lasso regression per packet) -> longer time limit.
# PER-FILE JOB ARRAY: one task per .mat file across SUBDIRS_ALL, all run at once.
#   The --array range must cover the number of .mat files. Easiest: submit via
#   submit_hest.sh, which counts the files and sizes --array automatically.
#   Manual override, e.g. for 6 files:  sbatch --array=0-5 02_phase_h_estimation.sh

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

# Args: dir  all_dir  name  nss  ncore(=#antennas)  start_r  end_r
# all_dir=0 -> process ONLY this one file (its 4 streams); end_r=-1 -> to the end.
echo "=== H-estimation $sub/$name (array task $idx of ${#FILES[@]}) ==="
# -u = unbuffered stdout so progress prints show up in the log live (not at the end).
python -u CSI_phase_sanitization_H_estimation.py \
    "$INPUT_DIR/$sub/" 0 "$name" "$NSS" "${N_RX:-4}" 0 -1
echo "Done."
