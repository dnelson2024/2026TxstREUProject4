#!/bin/bash
# Push the SHARP code from this local repo to LEAP2.
# Run from your LOCAL machine (NOT on LEAP2), in the repo root:
#     bash sync_to_leap2.sh              # do the sync
#     bash sync_to_leap2.sh --dry-run    # preview what would change, transfer nothing
#
# Mapping (matches LEAP2's layout, where the .py files live in the txstpr4 root):
#     Python_code/*.py  ->  $REMOTE_BASE/
#     slurm/*           ->  $REMOTE_BASE/slurm/
#
# SAFE BY DESIGN: no --delete, and it only sends Python_code/ and slurm/. Your
# LEAP2 data/output dirs (input_files, phase_processing, processed_phase,
# doppler_traces) and logs are never read or touched by this script.
set -euo pipefail

REMOTE="urq23@leap2.txstate.edu"
REMOTE_BASE="/mmfs1/home/urq23/txstpr4"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

DRY=()
if [ "${1:-}" = "--dry-run" ]; then
    DRY=(--dry-run)
    echo "=== DRY RUN: showing changes only, transferring nothing ==="
fi

# Common rsync flags. -a archive, -v verbose, -z compress, -i show item changes.
COMMON=(-avzi "${DRY[@]}" --exclude '__pycache__/' --exclude '*.pyc')

echo "==> Python_code/  ->  $REMOTE:$REMOTE_BASE/"
# Exclude the big pretrained .h5 models (only needed by the network stages you
# aren't running). Drop this --exclude if you ever need them on LEAP2.
rsync "${COMMON[@]}" --exclude '*.h5' \
    "$REPO_DIR/Python_code/" "$REMOTE:$REMOTE_BASE/"

echo "==> slurm/  ->  $REMOTE:$REMOTE_BASE/slurm/"
# NOTE: this includes config.sh -> it WILL overwrite the LEAP2 config.sh with the
# repo copy. That copy has the fixes (trailing-slash paths, N_RX, conda activate).
# If you keep LEAP2-only edits in config.sh, add:  --exclude 'config.sh'
rsync "${COMMON[@]}" --exclude 'logs/' \
    "$REPO_DIR/slurm/" "$REMOTE:$REMOTE_BASE/slurm/"

echo "Done."
