#!/bin/bash
# Submit stage 01 (preprocessing) as a PER-FILE job array, sized to the number
# of .mat files across SUBDIRS_ALL. Run from the slurm/ folder:  bash submit_preproc.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
source ./config.sh

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
sbatch --array=0-$((n-1)) 01_phase_preprocessing.sh
echo "Watch with: squeue -u \$USER"
