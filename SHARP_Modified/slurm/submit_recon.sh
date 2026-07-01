#!/bin/bash
# Submit stage 03 (reconstruction) as a PER-FILE job array, sized to the number
# of Tr_vector_*.txt files in PHASE_PROCESSING_DIR (produced by stage 02).
# Run from the slurm/ folder:  bash submit_recon.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
source ./config.sh

n=0
for f in "$PHASE_PROCESSING_DIR"Tr*.txt; do
    [ -e "$f" ] && n=$((n+1))
done

if [ "$n" -eq 0 ]; then
    echo "ERROR: no Tr*.txt files in $PHASE_PROCESSING_DIR -- has stage 02 finished?" >&2
    exit 1
fi
echo "Found $n Tr file(s) in $PHASE_PROCESSING_DIR -> submitting array 0-$((n-1))"
sbatch --array=0-$((n-1)) 03_phase_reconstruction.sh
echo "Watch with: squeue -u \$USER"
