#!/bin/bash
# Submit stage 04 (doppler) as a PER-SUBDIR job array, sized to the number of
# subdirs in SUBDIRS_ALL. Run from the slurm/ folder:  bash submit_doppler.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
source ./config.sh

IFS=',' read -ra DIRS <<< "$SUBDIRS_ALL"
n="${#DIRS[@]}"

if [ "$n" -eq 0 ]; then
    echo "ERROR: SUBDIRS_ALL is empty" >&2
    exit 1
fi
echo "Found $n subdir(s) [$SUBDIRS_ALL] -> submitting array 0-$((n-1))"
sbatch --array=0-$((n-1)) 04_doppler_computation.sh
echo "Watch with: squeue -u \$USER"
