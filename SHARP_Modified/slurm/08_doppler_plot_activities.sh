#!/bin/bash
#SBATCH --job-name=sharp_08_plotact
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=5-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=danielle.a.nelson@stonybrook.edu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# IMPORTANT: SLURM only reads "#SBATCH" (no space after #). "# SBATCH" is treated
# as a plain comment and the directive is ignored -> that caused the
# "No partition specified" error. Keep these lines exactly as "#SBATCH ...".
# Headless plotting via MPLBACKEND=Agg (set in config.sh) -> saves .pdf, no display.

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/config.sh"
activate_env

# The script hard-codes "./plots/" for output -> make sure it exists.
mkdir -p ./plots "$IMAGE_DIR"

# Args: dir  sub_dir  feature_length  sliding  labels_activities  start_plt  end_plt
# Produces the COMBINED multi-activity PDF(s) per subdir. By default loops over ALL
# of SUBDIRS_ALL; set PLOT_SUBDIR="S1a" (or "S1a,S2a") to restrict to some folders.
# Expects the 5-class layout (E,L,W,R,J); the script skips its 5-class-only panels
# gracefully if fewer are present.
if [ -n "${PLOT_SUBDIR:-}" ]; then
    IFS=',' read -ra DIRS <<< "$PLOT_SUBDIR"
else
    IFS=',' read -ra DIRS <<< "$SUBDIRS_ALL"
fi

for sub in "${DIRS[@]}"; do
    echo "=== doppler plots (activities) for $sub ==="
    # '|| echo' so one bad subdir doesn't abort the whole loop under set -e.
    python -u CSI_doppler_plot_activities.py \
        "$DOPPLER_DIR" "$sub" 100 "$SLIDING" "$ACTIVITIES" 570 1070 \
        || echo "WARNING: plotting failed for $sub -- skipping"
done

# Collect ONLY the combined multi-activity PDFs (each is named per-subdir) into
# IMAGE_DIR. Skips the single-activity plot and any per-activity PNGs.
cp -v ./plots/csi_doppler_activities_*.pdf "$IMAGE_DIR"/ 2>/dev/null || echo "(no combined PDFs produced)"
echo "Images collected in: $IMAGE_DIR"
echo "Done."
