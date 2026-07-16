#!/bin/bash
# Sync the Widar3.0 raw-CSI pipeline with LEAP2. Run LOCALLY (on the DGX):
#   bash Widar3.0/sync_widar_to_leap2.sh push        # code only
#   bash Widar3.0/sync_widar_to_leap2.sh push-har    # raw HAR .dat tree (~95 GB)
#   bash Widar3.0/sync_widar_to_leap2.sh push-gait   # raw Gait .dat tree (~13 GB)
#   bash Widar3.0/sync_widar_to_leap2.sh fetch       # pull Doppler datasets + plots
#
# push-har alternative: the full HAR set can instead be downloaded DIRECTLY on
# a LEAP2 login node from the Tsinghua share (no DGX upload):
#   bash Widar3.0/download_widar_all.sh   # unzips into Widar_CSI/<date>/...
#   ln -s Widar_CSI Widar3.0-HAR          # then point the pipeline at it
# All pushes are idempotent -- a dropped connection just needs a re-run.
set -euo pipefail

REMOTE="urq23@leap2.txstate.edu"
REMOTE_BASE="/mmfs1/home/urq23/txstpr4"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="/home/danelson/proj4-txstreu/2026TxstREUProject4-data"

# one shared ssh connection for every rsync in this run
export RSYNC_RSH="ssh -o ControlMaster=auto -o ControlPath=$HOME/.ssh/leap2-sync-%r@%h-%p -o ControlPersist=15m"

case "${1:?usage: sync_widar_to_leap2.sh push|push-har|push-gait|fetch}" in
  push)
    rsync -az --relative \
          "$LOCAL_DIR/./csi_pipeline.py" "$LOCAL_DIR/./slurm" \
          "$LOCAL_DIR/./download_widar_all.sh" \
          "$REMOTE:$REMOTE_BASE/Widar3.0/"
    echo "PUSH DONE. Next: push-har / push-gait (or download HAR on LEAP2), then"
    echo "  bash Widar3.0/slurm/wr_submit.sh har"
    echo "  bash Widar3.0/slurm/wr_submit.sh gait"
    ;;
  push-har)
    rsync -a --info=progress2 --exclude='*.zip' \
          "$DATA_ROOT/Widar3.0-HAR/" "$REMOTE:$REMOTE_BASE/Widar3.0-HAR/"
    echo "PUSH-HAR DONE -> $REMOTE_BASE/Widar3.0-HAR/"
    ;;
  push-gait)
    rsync -a --info=progress2 --exclude='*.zip' \
          "$DATA_ROOT/Widar3.0-Gait/" "$REMOTE:$REMOTE_BASE/Widar3.0-Gait/"
    echo "PUSH-GAIT DONE -> $REMOTE_BASE/Widar3.0-Gait/"
    ;;
  fetch)
    for task in HAR Gait; do
        rsync -az --info=progress2 \
              "$REMOTE:$REMOTE_BASE/widar_doppler_data_$task/" \
              "$DATA_ROOT/widar_doppler_data_$task/" \
            || echo "(widar_doppler_data_$task not on LEAP2 yet -- skipped)"
    done
    rsync -az "$REMOTE:$REMOTE_BASE/widar_csi_work_har/plots/" \
          "$LOCAL_DIR/plots/doppler_har/" || true
    rsync -az "$REMOTE:$REMOTE_BASE/widar_csi_work_gait/plots/" \
          "$LOCAL_DIR/plots/doppler_gait/" || true
    echo "FETCH DONE -> $DATA_ROOT/widar_doppler_data_{HAR,Gait}/"
    ;;
  *)
    echo "usage: $0 push|push-har|push-gait|fetch" >&2; exit 1
    ;;
esac
