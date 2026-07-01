#!/bin/bash
# Submit the whole non-network pipeline, parallelized, in dependency order.
# Stages 01-04 are job arrays (sized automatically); 05/06 are single jobs that
# aggregate all data. Each stage waits for the previous to finish OK (afterok).
#
# Usage:  bash submit_all.sh        (run from the slurm/ folder)
# To run one stage alone, use its submit_*.sh helper instead.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
source ./config.sh

# --- Compute array sizes from inputs known up front -------------------------
IFS=',' read -ra DIRS <<< "$SUBDIRS_ALL"
n_sub="${#DIRS[@]}"
n_mat=0
for sub in "${DIRS[@]}"; do
    for f in "$INPUT_DIR/$sub/"*.mat; do
        [ -e "$f" ] && n_mat=$((n_mat+1))
    done
done
if [ "$n_mat" -eq 0 ]; then
    echo "ERROR: no .mat files under $INPUT_DIR for SUBDIRS_ALL=$SUBDIRS_ALL" >&2
    exit 1
fi
n_tr=$((4 * n_mat))    # stage 02 writes 4 Tr files (one per stream) per .mat

echo "Sizing: $n_mat .mat file(s), $n_sub subdir(s) -> stage03 array up to $n_tr tasks"

# --- Submit the chain -------------------------------------------------------
j1=$(sbatch --parsable --array=0-$((n_mat-1)) 01_phase_preprocessing.sh)
echo "01 preprocessing   -> $j1  (array 0-$((n_mat-1)))"
j2=$(sbatch --parsable --dependency=afterok:"$j1" --array=0-$((n_mat-1)) 02_phase_h_estimation.sh)
echo "02 H-estimation    -> $j2  (array 0-$((n_mat-1)))"
j3=$(sbatch --parsable --dependency=afterok:"$j2" --array=0-$((n_tr-1)) 03_phase_reconstruction.sh)
echo "03 reconstruction  -> $j3  (array 0-$((n_tr-1)), extra tasks no-op)"
j4=$(sbatch --parsable --dependency=afterok:"$j3" --array=0-$((n_sub-1)) 04_doppler_computation.sh)
echo "04 doppler         -> $j4  (array 0-$((n_sub-1)))"
j5=$(sbatch --parsable --dependency=afterok:"$j4" 05_create_dataset_train.sh)
echo "05 dataset (train) -> $j5"
j6=$(sbatch --parsable --dependency=afterok:"$j4" 06_create_dataset_test.sh)
echo "06 dataset (test)  -> $j6"
echo "Submitted. Watch with: squeue -u \$USER"
