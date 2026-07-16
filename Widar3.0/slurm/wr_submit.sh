#!/bin/bash
# Submit the Widar3.0 raw-CSI -> SHARP Doppler pipeline on LEAP2 for ONE task:
#   bash Widar3.0/slurm/wr_submit.sh har     # gesture recognition dataset
#   bash Widar3.0/slurm/wr_submit.sh gait    # gait (user) identification dataset
# Prerequisite: the raw .dat tree at the path wr_env.sh maps for the task
# (Widar3.0-HAR/ or Widar3.0-Gait/ under ~/txstpr4), e.g. rsynced from the DGX
# with  bash Widar3.0/sync_widar_to_leap2.sh push-har|push-gait
# Builds the manifest on the login node, prints TWO coherence diagnostics
# (sanity: Intel 5300 phase must be coherent, unlike CSI-Bench), then submits
# the strided process array and the dataset+plots job (afterok dependency).
set -euo pipefail

BASE="/mmfs1/home/urq23/txstpr4"
cd "$BASE"
mkdir -p logs   # missing logs/ dir at submit CWD kills array tasks silently

source /mmfs1/home/urq23/anaconda3/etc/profile.d/conda.sh
conda activate p4
TASK="${1:?usage: wr_submit.sh har|gait}"
source "$BASE/Widar3.0/slurm/wr_env.sh" "$TASK"
echo "task: $WIDAR_TASK   representation: $WIDAR_REPR"
echo "csi:  $WIDAR_CSI"
echo "out:  $WIDAR_OUT"

python -u "$BASE/Widar3.0/csi_pipeline.py" manifest
WR_N=$(( $(wc -l < "$WIDAR_WORK/manifest.csv") - 1 ))
echo "--- coherence check (should say 'coherent'; CSI-Bench said RANDOMIZED) ---"
python -u "$BASE/Widar3.0/csi_pipeline.py" coherence --index 0
python -u "$BASE/Widar3.0/csi_pipeline.py" coherence --index $((WR_N / 2))

WR_TASKS=$(( WR_N < 1000 ? WR_N : 1000 ))
ARRAY_ID=$(sbatch --parsable -J "wr_${TASK}_proc" --array=0-$((WR_TASKS - 1))%150 \
           --export=ALL,WR_N="$WR_N",WR_TASKS="$WR_TASKS" \
           "$BASE/Widar3.0/slurm/wr_csi.sh" "$TASK" process)
echo "process array submitted: $ARRAY_ID ($WR_TASKS tasks over $WR_N files)"

DATASET_ID=$(sbatch --parsable -J "wr_${TASK}_data" --dependency=afterok:"$ARRAY_ID" \
             "$BASE/Widar3.0/slurm/wr_csi.sh" "$TASK" dataset)
echo "dataset+plots job submitted: $DATASET_ID (runs after the array)"
echo "watch with:  squeue -u urq23"
