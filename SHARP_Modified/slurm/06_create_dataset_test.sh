#!/bin/bash
#SBATCH --job-name=sharp_06_dstest
#SBATCH --partition=shared
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=danielle.a.nelson@stonybrook.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# NOTE: edit --partition above to match LEAP2_PARTITION in config.sh.

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/config.sh"
activate_env

# Args: dir  subdirs  sample_lengths  sliding  windows_length  stride  labels  n_tot
echo "=== create dataset (test) ==="
python CSI_doppler_create_dataset_test.py \
    "$DOPPLER_DIR" "$SUBDIRS_TEST" \
    "$SAMPLE_LENGTH" "$SLIDING" "$WINDOW_LENGTH" "$STRIDE" \
    "$ACTIVITIES" "$N_TOT"
echo "Done."
