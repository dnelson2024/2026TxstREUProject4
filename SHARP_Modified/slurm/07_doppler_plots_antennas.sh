#!/bin/bash
#SBATCH --job-name=sharp_07_plotant
#SBATCH --partition=shared
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=danielle.a.nelson@stonybrook.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# NOTE: edit --partition above to match LEAP2_PARTITION in config.sh.
# Headless plotting via MPLBACKEND=Agg (set in config.sh) -> saves .png, no display.

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/config.sh"
activate_env

# The script hard-codes "./plots/" for output -> make sure it exists.
mkdir -p ./plots "$IMAGE_DIR"

# Args: dir  sub_dir  feature_length  sliding  labels_activities  end_plt
# Plots all antennas per capture folder. By default loops over ALL of SUBDIRS_ALL;
# set PLOT_SUBDIR="S1a" (or "S1a,S2a") to restrict to specific folders.
if [ -n "${PLOT_SUBDIR:-}" ]; then
    IFS=',' read -ra DIRS <<< "$PLOT_SUBDIR"
else
    IFS=',' read -ra DIRS <<< "$SUBDIRS_ALL"
fi

for sub in "${DIRS[@]}"; do
    echo "=== doppler plots (antennas) for $sub ==="
    # '|| echo' so one bad subdir doesn't abort the whole loop under set -e.
    python -u CSI_doppler_plots_antennas.py \
        "$DOPPLER_DIR" "$sub" 100 "$SLIDING" "$ACTIVITIES" 20000 \
        || echo "WARNING: plotting failed for $sub -- skipping"
done

# Collect the generated images into the central IMAGE_DIR.
cp -v ./plots/*.png "$IMAGE_DIR"/ 2>/dev/null || echo "(no .png produced)"
echo "Images collected in: $IMAGE_DIR"
echo "Done."
