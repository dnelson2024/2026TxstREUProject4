#!/bin/bash
# Train (and optionally test + print metrics) the SHARP models on the NVIDIA DGX.
# TensorFlow uses the GPU automatically -- no SLURM here, just plain bash.
#
# IMPORTANT: this script lives in SHARP_Modified/, but CSI_network.py uses package
# imports (SHARP_Modified.Python_code.*), so it must be launched as a module from
# the REPO ROOT (the parent of SHARP_Modified/) -- we cd up one level to get there.
# All output dirs (models/, outputs/, plots/, reports/) live at that repo root too.
#
# Usage (run from anywhere):
#   bash SHARP_Modified/run_models_dgx.sh                    # all 7 models
#   bash SHARP_Modified/run_models_dgx.sh cnn_bilstm vit     # only the listed models
set -euo pipefail
cd "$(dirname "$0")/.."     # repo root == parent of SHARP_Modified/, where this script now lives
export MPLBACKEND=Agg      # headless plotting -- no GUI window during batch runs

# ---------- EDIT for your DGX data / layout ----------
DATA_DIR="/home/shine-lab/proj4-txstreu/2026TxstREUProject4-data/doppler_traces/"   # holds <subdir>/{train,val,test,complete}_antennas_*
SUBDIRS_TRAIN="S1a,S1b,S1c"         # CSI_network.py trains on these
SUBDIRS_TEST="S7a"                  # CSI_network_test.py evaluates on these
FEATURE_LENGTH=100
SAMPLE_LENGTH=340
CHANNELS=1
BATCH_SIZE=32
NUM_TOT=4
NAME_BASE="models/results/single_ant"
ACTIVITIES="E,L,W,R,J"
BANDWIDTH=80
SUB_BAND=1
RUN_TEST=1                          # 1 = also run test + metrics per model; 0 = train only
# -----------------------------------------------------

ALL_MODELS=(cnn cnn_bilstm vit bilstm lstm rcnn widar3 random_forest svm knn gradient_boosting naive_bayes)
MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=("${ALL_MODELS[@]}")

# Optional hyperparameter overrides, e.g. to promote a grid-search winner:
#   LEARNING_RATE=0.0003 DROPOUT=0.3 bash SHARP_Modified/run_models_dgx.sh cnn_bilstm
# Unset -> CSI_network.py defaults (lr 0.0001; dropout 0.2 cnn / 0.3 others).
EXTRA_ARGS=()
[ -n "${LEARNING_RATE:-}" ] && EXTRA_ARGS+=(--learning_rate "$LEARNING_RATE")
[ -n "${DROPOUT:-}" ] && EXTRA_ARGS+=(--dropout "$DROPOUT")

# CSI_network.py/plt.savefig() don't create directories themselves.
mkdir -p models/results outputs/results/for_machine plots/results/pdfs plots/results/pngs reports/results

echo "Models to run: ${MODELS[*]}"
for m in "${MODELS[@]}"; do
    echo "=================== [$m] TRAIN ==================="
    python -m SHARP_Modified.Python_code.CSI_network \
        "$DATA_DIR" "$SUBDIRS_TRAIN" "$FEATURE_LENGTH" "$SAMPLE_LENGTH" "$CHANNELS" \
        "$BATCH_SIZE" "$NUM_TOT" "$NAME_BASE" "$ACTIVITIES" \
        --model "$m" --bandwidth "$BANDWIDTH" --sub_band "$SUB_BAND" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

    # CSI_network.py always writes loss plots and outputs/test_*.txt to the flat
    # top-level plots/pdfs, plots/pngs, outputs/ -- archive into production/ right away
    # so a later model or script (grid search) can't collide with these filenames.
    train_tag="${ACTIVITIES}_${SUBDIRS_TRAIN}_${m}_band_${BANDWIDTH}_subband_${SUB_BAND}"
    mv "plots/pdfs/loss_${train_tag}.pdf" "plots/results/pdfs/" 2>/dev/null || true
    mv "plots/pngs/loss_${train_tag}.png" "plots/results/pngs/" 2>/dev/null || true
    mv "outputs/test_${train_tag}.for_machine.pkl" "outputs/results/for_machine/" 2>/dev/null || true
    mv "outputs/change_number_antennas_test_${train_tag}.for_machine.pkl" "outputs/results/for_machine/" 2>/dev/null || true

    if [ "$RUN_TEST" = "1" ]; then
        echo "=================== [$m] TEST ==================="
        python -m SHARP_Modified.Python_code.CSI_network_test \
            "$DATA_DIR" "$SUBDIRS_TEST" "$FEATURE_LENGTH" "$SAMPLE_LENGTH" "$CHANNELS" \
            "$BATCH_SIZE" "$NUM_TOT" "$NAME_BASE" "$ACTIVITIES" \
            --model "$m" --bandwidth "$BANDWIDTH" --sub_band "$SUB_BAND"

        echo "=================== [$m] METRICS ==================="
        # Matches the filename CSI_network_test.py writes for this model.
        metrics_name="complete_different_${ACTIVITIES}_${SUBDIRS_TEST}_${m}_band_${BANDWIDTH}_subband_${SUB_BAND}"
        # tee: still shows in the live run log, and also saved for later reading.
        python -m SHARP_Modified.Python_code.CSI_network_metrics "$metrics_name" "$ACTIVITIES" \
            | tee "reports/results/${metrics_name}.txt"

        echo "=================== [$m] PLOTS ==================="
        # Writes plots/{pdfs,pngs}/cm_<metrics_name>*.{pdf,png} and roc_<metrics_name>.{pdf,png}
        python -m SHARP_Modified.Python_code.CSI_network_metrics_plot "$metrics_name" "$ACTIVITIES" --model "$m"
        mv "plots/pdfs/cm_${metrics_name}.pdf" "plots/pdfs/cm_${metrics_name}_max_merge.pdf" \
            "plots/pdfs/roc_${metrics_name}.pdf" plots/results/pdfs/ 2>/dev/null || true
        mv "plots/pngs/cm_${metrics_name}.png" "plots/pngs/cm_${metrics_name}_max_merge.png" \
            "plots/pngs/roc_${metrics_name}.png" plots/results/pngs/ 2>/dev/null || true
        mv "outputs/complete_different_${ACTIVITIES}_${SUBDIRS_TEST}_${m}_band_${BANDWIDTH}_subband_${SUB_BAND}.for_machine.pkl" \
           "outputs/change_number_antennas_complete_different_${ACTIVITIES}_${SUBDIRS_TEST}_${m}_band_${BANDWIDTH}_subband_${SUB_BAND}.for_machine.pkl" \
           outputs/results/for_machine/ 2>/dev/null || true

        # The pickles above are binary (for the metrics/plot scripts); regenerate the
        # human-readable per-model summary CSV from all of them after every model.
        python - "$ACTIVITIES" "$SUBDIRS_TEST" <<'EOF'
import csv, glob, pickle, re, sys
activities, subdirs_test = sys.argv[1], sys.argv[2]
rows = []
for f in sorted(glob.glob('outputs/results/for_machine/complete_different_%s_%s_*.for_machine.pkl' % (activities, subdirs_test))):
    model = re.search(r'_%s_(.+?)_band_' % re.escape(subdirs_test), f).group(1)
    with open(f, 'rb') as fp:
        d = pickle.load(fp)
    row = {'model': model,
           'accuracy_single': round(float(d['accuracy_single']), 4),
           'fscore_single': round(float(d['fscore_single'].mean()), 4),
           'accuracy_decision_fusion': round(float(d['accuracy_max_merge']), 4),
           'fscore_decision_fusion': round(float(d['fscore_max_merge'].mean()), 4)}
    for act, fs in zip(activities.split(','), d['fscore_max_merge']):
        row['fscore_' + act] = round(float(fs), 4)
    rows.append(row)
rows.sort(key=lambda r: r['accuracy_decision_fusion'], reverse=True)
out = 'outputs/results/summary_%s.csv' % subdirs_test
with open(out, 'w', newline='') as fp:
    w = csv.DictWriter(fp, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print('summary written: %s (%d models)' % (out, len(rows)))
EOF
    fi
done
echo "All models done."
