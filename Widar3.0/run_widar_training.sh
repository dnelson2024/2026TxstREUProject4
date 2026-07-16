#!/bin/bash
# Train the top-4 SHARP models (by S7a decision-fusion accuracy) on the
# Widar3.0 pilot Doppler dataset, sequentially, CPU-only (GPU unusable until
# the DGX reboot -- UVM zombies). Run detached from the repo root:
#   setsid bash Widar3.0/run_widar_training.sh > widar_train.log 2>&1 < /dev/null &
source /home/danelson/anaconda3/etc/profile.d/conda.sh
conda activate p4v3.11
set -uo pipefail
cd /home/danelson/proj4-txstreu/2026TxstREUProject4
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=""   # CPU only: GPU hangs until reboot

EPOCHS="${EPOCHS:-25}"           # env-overridable, e.g. EPOCHS=50 bash Widar3.0/run_widar_training.sh

for m in rcnn cnn_bilstm lstm bilstm; do
    echo "=================== [widar $m] $(date '+%F %T') ==================="
    python -u Widar3.0/train_widar.py "$m" --epochs "$EPOCHS" || echo "=== $m FAILED ==="
done
echo ""
cat Widar3.0/results_widar.csv 2>/dev/null || true
echo "WIDAR TRAINING COMPLETE"
