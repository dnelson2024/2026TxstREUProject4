#!/bin/bash
# Submit stage 02 (H-estimation) as a PER-FILE job array, sized automatically to
# the number of .mat files across SUBDIRS_ALL so every activity file runs at once.
#
# Usage:  bash submit_hest.sh        (run from the slurm/ folder)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
source ./config.sh

# Count .mat files across the configured subdirs (same enumeration the job uses).
IFS=',' read -ra DIRS <<< "$SUBDIRS_ALL"
n=0
for sub in "${DIRS[@]}"; do
    for f in "$INPUT_DIR/$sub/"*.mat; do
        [ -e "$f" ] && n=$((n+1))
    done
done

if [ "$n" -eq 0 ]; then
    echo "ERROR: no .mat files found under $INPUT_DIR for SUBDIRS_ALL=$SUBDIRS_ALL" >&2
    exit 1
fi

echo "Found $n .mat file(s) across [$SUBDIRS_ALL] -> submitting array 0-$((n-1))"
sbatch --array=0-$((n-1)) 02_phase_h_estimation.sh
echo "Watch with: squeue -u \$USER"
