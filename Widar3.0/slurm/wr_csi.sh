#!/bin/bash
#SBATCH --job-name=widar_csi
#SBATCH --partition=parallel
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=danielle.a.nelson@stonybrook.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# One job script for both datasets and both stages:
#   wr_csi.sh <har|gait> process   ARRAY task: SHARP steps 01-04 for .dat files
#            task_id, task_id+T, task_id+2T, ... (T = WR_TASKS, set by
#            wr_submit.sh so the array fits SLURM's MaxArraySize)
#   wr_csi.sh <har|gait> dataset   single task: steps 05+06 (windows + splits
#            -> .npy) then 07+08 (SHARP-style plots)
# Submit everything with:  bash Widar3.0/slurm/wr_submit.sh har|gait
set -uo pipefail   # no -e: one bad recording must not kill the whole task

source /mmfs1/home/urq23/anaconda3/etc/profile.d/conda.sh
conda activate p4
set -e
export MPLBACKEND=Agg

TASK="${1:?usage: wr_csi.sh har|gait process|dataset}"
STAGE="${2:?usage: wr_csi.sh har|gait process|dataset}"
source "/mmfs1/home/urq23/txstpr4/Widar3.0/slurm/wr_env.sh" "$TASK"
PIPE="$WR_BASE/Widar3.0/csi_pipeline.py"

case "$STAGE" in
  process)
    : "${WR_N:?set by wr_submit.sh}" "${WR_TASKS:?set by wr_submit.sh}"
    for ((i = ${SLURM_ARRAY_TASK_ID:?array task id missing}; i < WR_N; i += WR_TASKS)); do
        python -u "$PIPE" process --index "$i" || echo "INDEX $i FAILED"
    done
    ;;
  dataset)
    python -u "$PIPE" dataset
    python -u "$PIPE" plots
    ;;
  *)
    echo "unknown stage: $STAGE" >&2; exit 1
    ;;
esac
echo "Done."
