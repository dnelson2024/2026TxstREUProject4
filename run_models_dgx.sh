#!/bin/bash
# Train (and optionally test + print metrics) the SHARP models on the NVIDIA DGX.
# TensorFlow uses the GPU automatically -- no SLURM here, just plain bash.
#
# IMPORTANT: run from the REPO ROOT (this script's own folder). CSI_network.py
# uses package imports (SHARP_Modified.Python_code.*), so it must be launched as
# a module from the parent of SHARP_Modified/ -- which is where this script lives.
#
# Usage:
#   bash run_models_dgx.sh                    # all 7 models
#   bash run_models_dgx.sh cnn_bilstm vit     # only the listed models
set -euo pipefail
cd "$(dirname "$0")"        # repo root == parent of SHARP_Modified/

# ---------- EDIT for your DGX data / layout ----------
DATA_DIR="doppler_traces/"          # holds <subdir>/{train,val,test,complete}_antennas_*
SUBDIRS_TRAIN="S1a,S1b,S1c"         # CSI_network.py trains on these
SUBDIRS_TEST="S7a"                  # CSI_network_test.py evaluates on these
FEATURE_LENGTH=100
SAMPLE_LENGTH=340
CHANNELS=1
BATCH_SIZE=32
NUM_TOT=4
NAME_BASE="single_ant"
ACTIVITIES="E,L,W,R,J"
BANDWIDTH=80
SUB_BAND=1
RUN_TEST=1                          # 1 = also run test + metrics per model; 0 = train only
# -----------------------------------------------------

ALL_MODELS=(cnn cnn_bilstm vit bilstm lstm rcnn random_forest)
MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=("${ALL_MODELS[@]}")

echo "Models to run: ${MODELS[*]}"
for m in "${MODELS[@]}"; do
    echo "=================== [$m] TRAIN ==================="
    python -m SHARP_Modified.Python_code.CSI_network \
        "$DATA_DIR" "$SUBDIRS_TRAIN" "$FEATURE_LENGTH" "$SAMPLE_LENGTH" "$CHANNELS" \
        "$BATCH_SIZE" "$NUM_TOT" "$NAME_BASE" "$ACTIVITIES" \
        --model "$m" --bandwidth "$BANDWIDTH" --sub_band "$SUB_BAND"

    if [ "$RUN_TEST" = "1" ]; then
        echo "=================== [$m] TEST ==================="
        python -m SHARP_Modified.Python_code.CSI_network_test \
            "$DATA_DIR" "$SUBDIRS_TEST" "$FEATURE_LENGTH" "$SAMPLE_LENGTH" "$CHANNELS" \
            "$BATCH_SIZE" "$NUM_TOT" "$NAME_BASE" "$ACTIVITIES" \
            --model "$m" --bandwidth "$BANDWIDTH" --sub_band "$SUB_BAND"

        echo "=================== [$m] METRICS ==================="
        # Matches the filename CSI_network_test.py writes for this model.
        metrics_name="complete_different_${ACTIVITIES}_${SUBDIRS_TEST}_${m}_band_${BANDWIDTH}_subband_${SUB_BAND}"
        python -m SHARP_Modified.Python_code.CSI_network_metrics "$metrics_name" "$ACTIVITIES"
    fi
done
echo "All models done."
