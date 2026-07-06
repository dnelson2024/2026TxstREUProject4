#!/bin/bash
# Curated sweep of the NEW hyperparameter dimensions (--hparams) for the non-cnn
# models: architecture knobs for the Keras models (units, num_layers, filters,
# merge_mode, vit patch/dim/depth) and constructor knobs for the classical models
# (C/kernel, n_neighbors/weights, n_estimators, var_smoothing, pca_components).
#
# Deliberately NOT a full grid: each model gets 2-3 hand-picked variations built
# on top of its best-known lr/dropout from run_hparam_grid_models.sh, so the
# whole sweep stays in the few-hours range (~22 runs x ~10 min).
# The cnn is excluded on purpose -- locked to the original SHARP architecture.
#
# Usage (run from anywhere):
#   bash SHARP_Modified/run_hparam_grid_dims.sh
set -euo pipefail
cd "$(dirname "$0")/.."     # repo root == parent of SHARP_Modified/
export MPLBACKEND=Agg

DATA_DIR="/home/shine-lab/proj4-txstreu/2026TxstREUProject4-data/doppler_traces/"
SUBDIRS_TRAIN="S1a,S1b,S1c"
SUBDIRS_TEST="S7a"
FEATURE_LENGTH=100
SAMPLE_LENGTH=340
CHANNELS=1
NUM_TOT=4
BATCH_SIZE=32
ACTIVITIES="E,L,W,R,J"
BANDWIDTH=80
SUB_BAND=1

# ---------- EDIT the configs here ----------
# One line per run:  model|learning_rate|dropout|hparams-JSON
# dropout "-" means: use the model's builtin default / not applicable (sklearn).
# Keras lr/dropout baselines = winners from run_hparam_grid_models.sh.
CONFIGS=(
  'cnn_bilstm|0.0003|0.3|{"units": 128}'
  'cnn_bilstm|0.0003|0.3|{"num_filters": [16, 32, 64]}'
  'rcnn|0.0003|0.2|{"units": 128}'
  'rcnn|0.0003|0.2|{"num_layers": 1}'
  'rcnn|0.0003|0.2|{"num_filters": [16, 32, 64]}'
  'bilstm|0.0001|0.4|{"units": 128}'
  'bilstm|0.0001|0.4|{"num_layers": 1}'
  'bilstm|0.0001|0.4|{"merge_mode": "sum"}'
  'lstm|0.0001|0.3|{"units": 128}'
  'lstm|0.0001|0.3|{"num_layers": 3}'
  'vit|0.0003|0.3|{"dim": 128, "heads": 8}'
  'vit|0.0003|0.3|{"depth": 6}'
  'vit|0.0003|0.3|{"patch": [20, 20]}'
  'svm|-|-|{"C": 10}'
  'svm|-|-|{"kernel": "linear"}'
  'svm|-|-|{"pca_components": 256}'
  'knn|-|-|{"n_neighbors": 11}'
  'knn|-|-|{"weights": "distance"}'
  'gradient_boosting|-|-|{"n_estimators": 500, "learning_rate": 0.05}'
  'gradient_boosting|-|-|{"pca_components": 256}'
  'naive_bayes|-|-|{"var_smoothing": 1e-6}'
  'naive_bayes|-|-|{"pca_components": 256}'
)
# --------------------------------------------

mkdir -p models/gridsearch_dims plots/gridsearch_dims/pdfs plots/gridsearch_dims/pngs \
         outputs/gridsearch_dims/for_machine

echo "Curated dims sweep: ${#CONFIGS[@]} runs"

results_file="outputs/gridsearch_dims/results_dims.csv"
if [ ! -f "$results_file" ]; then
    echo 'model,learning_rate,dropout,hparams,accuracy_single,fscore_single,accuracy_decision_fusion,fscore_decision_fusion' > "$results_file"
fi

run_idx=0
for cfg in "${CONFIGS[@]}"; do
    run_idx=$((run_idx + 1))
    IFS='|' read -r m lr do hp <<< "$cfg"
    tag="griddims_${m}_${run_idx}"
    name_base="models/gridsearch_dims/${tag}"

    TRAIN_ARGS=(--model "$m" --bandwidth "$BANDWIDTH" --sub_band "$SUB_BAND" --hparams "$hp")
    [ "$lr" != "-" ] && TRAIN_ARGS+=(--learning_rate "$lr")
    [ "$do" != "-" ] && TRAIN_ARGS+=(--dropout "$do")

    echo "=================== [$run_idx/${#CONFIGS[@]}] $m hparams=$hp TRAIN ==================="
    python -m SHARP_Modified.Python_code.CSI_network \
        "$DATA_DIR" "$SUBDIRS_TRAIN" "$FEATURE_LENGTH" "$SAMPLE_LENGTH" "$CHANNELS" \
        "$BATCH_SIZE" "$NUM_TOT" "$name_base" "$ACTIVITIES" \
        "${TRAIN_ARGS[@]}"

    train_tag="${ACTIVITIES}_${SUBDIRS_TRAIN}_${m}_band_${BANDWIDTH}_subband_${SUB_BAND}"
    # sklearn models produce no loss plot -- guard the mv
    mv "plots/pdfs/loss_${train_tag}.pdf" "plots/gridsearch_dims/pdfs/loss_${tag}.pdf" 2>/dev/null || true
    mv "plots/pngs/loss_${train_tag}.png" "plots/gridsearch_dims/pngs/loss_${tag}.png" 2>/dev/null || true
    rm -f "outputs/test_${train_tag}.for_machine.pkl" \
          "outputs/change_number_antennas_test_${train_tag}.for_machine.pkl"

    echo "=================== [$run_idx/${#CONFIGS[@]}] $m hparams=$hp S7a TEST ==================="
    python -m SHARP_Modified.Python_code.CSI_network_test \
        "$DATA_DIR" "$SUBDIRS_TEST" "$FEATURE_LENGTH" "$SAMPLE_LENGTH" "$CHANNELS" \
        "$BATCH_SIZE" "$NUM_TOT" "$name_base" "$ACTIVITIES" \
        --model "$m" --bandwidth "$BANDWIDTH" --sub_band "$SUB_BAND"

    src_metrics="outputs/complete_different_${ACTIVITIES}_${SUBDIRS_TEST}_${m}_band_${BANDWIDTH}_subband_${SUB_BAND}.for_machine.pkl"
    dst_metrics="outputs/gridsearch_dims/for_machine/test_${tag}.for_machine.pkl"
    mv "$src_metrics" "$dst_metrics"
    mv "outputs/change_number_antennas_complete_different_${ACTIVITIES}_${SUBDIRS_TEST}_${m}_band_${BANDWIDTH}_subband_${SUB_BAND}.for_machine.pkl" \
       "outputs/gridsearch_dims/for_machine/change_number_antennas_${tag}.for_machine.pkl"

    DST="$dst_metrics" MODEL="$m" LR="$lr" DO="$do" HP="$hp" RESULTS="$results_file" python - <<'EOF'
import csv, os, pickle
with open(os.environ['DST'], 'rb') as fp:
    d = pickle.load(fp)
with open(os.environ['RESULTS'], 'a', newline='') as f:
    csv.writer(f).writerow([os.environ['MODEL'], os.environ['LR'], os.environ['DO'], os.environ['HP'],
                            d['accuracy_single'], d['fscore_single'].mean(),
                            d['accuracy_max_merge'], d['fscore_max_merge'].mean()])
EOF
done

echo
echo "=================== RESULTS (per model, best first) ==================="
RESULTS="$results_file" python - <<'EOF'
import csv, os
rows = list(csv.DictReader(open(os.environ['RESULTS'])))
rows.sort(key=lambda r: (r['model'], -float(r['accuracy_decision_fusion'])))
print('%-18s %-42s %11s %11s' % ('model', 'hparams', 'acc_fusion', 'f1_fusion'))
for r in rows:
    print('%-18s %-42s %11.4f %11.4f' % (r['model'], r['hparams'][:42],
          float(r['accuracy_decision_fusion']), float(r['fscore_decision_fusion'])))
EOF
echo
echo "Full CSV: $results_file"
echo "=== DIMS SWEEP COMPLETE ==="
