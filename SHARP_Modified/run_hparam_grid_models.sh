#!/bin/bash
# Hyperparameter grid search for the non-cnn Keras models.
#
# Sweeps learning rate x dropout for each model in MODELS (batch_size fixed at 32).
# Each combo is trained on S1a/S1b/S1c and evaluated on the S7a domain-transfer set,
# same as run_models_dgx.sh / the cnn sweeps, so results are directly comparable.
# The cnn model is deliberately NOT in the list -- it was already swept by
# run_hparam_grid_arch.sh (best: lr=0.0001, dropout=0.4, filter_scale=1.0).
#
# Archives models/plots/pickles per combo into the gridsearch_models/ folders and
# appends every combo to one combined CSV (outputs/gridsearch_models/results_models.csv).
#
# Usage (run from anywhere):
#   bash SHARP_Modified/run_hparam_grid_models.sh                 # all 5 models
#   bash SHARP_Modified/run_hparam_grid_models.sh vit bilstm      # only the listed models
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

# ---------- EDIT the grid here ----------
ALL_MODELS=(cnn_bilstm vit bilstm lstm rcnn)
LEARNING_RATES=(0.0001 0.0003)
DROPOUTS=(0.2 0.3 0.4)
# -----------------------------------------

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=("${ALL_MODELS[@]}")

mkdir -p models/gridsearch_models plots/gridsearch_models/pdfs plots/gridsearch_models/pngs \
         outputs/gridsearch_models/for_machine

n_runs=$(( ${#MODELS[@]} * ${#LEARNING_RATES[@]} * ${#DROPOUTS[@]} ))
echo "Grid search: ${#MODELS[@]} models x ${#LEARNING_RATES[@]} lr x ${#DROPOUTS[@]} dropout = $n_runs runs"

results_file="outputs/gridsearch_models/results_models.csv"
if [ ! -f "$results_file" ]; then
    echo "model,learning_rate,dropout,accuracy_single,fscore_single,accuracy_decision_fusion,fscore_decision_fusion" > "$results_file"
fi

for m in "${MODELS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        for do in "${DROPOUTS[@]}"; do
            tag="gridmodels_${m}_lr${lr}_do${do}"
            name_base="models/gridsearch_models/${tag}"
            echo "=================== [$m] lr=$lr dropout=$do TRAIN ==================="
            python -m SHARP_Modified.Python_code.CSI_network \
                "$DATA_DIR" "$SUBDIRS_TRAIN" "$FEATURE_LENGTH" "$SAMPLE_LENGTH" "$CHANNELS" \
                "$BATCH_SIZE" "$NUM_TOT" "$name_base" "$ACTIVITIES" \
                --model "$m" --bandwidth "$BANDWIDTH" --sub_band "$SUB_BAND" \
                --learning_rate "$lr" --dropout "$do"

            train_tag="${ACTIVITIES}_${SUBDIRS_TRAIN}_${m}_band_${BANDWIDTH}_subband_${SUB_BAND}"
            mv "plots/pdfs/loss_${train_tag}.pdf" "plots/gridsearch_models/pdfs/loss_${tag}.pdf"
            mv "plots/pngs/loss_${train_tag}.png" "plots/gridsearch_models/pngs/loss_${tag}.png"
            rm -f "outputs/test_${train_tag}.for_machine.pkl" \
                  "outputs/change_number_antennas_test_${train_tag}.for_machine.pkl"

            echo "=================== [$m] lr=$lr dropout=$do S7a TEST ==================="
            python -m SHARP_Modified.Python_code.CSI_network_test \
                "$DATA_DIR" "$SUBDIRS_TEST" "$FEATURE_LENGTH" "$SAMPLE_LENGTH" "$CHANNELS" \
                "$BATCH_SIZE" "$NUM_TOT" "$name_base" "$ACTIVITIES" \
                --model "$m" --bandwidth "$BANDWIDTH" --sub_band "$SUB_BAND"

            src_metrics="outputs/complete_different_${ACTIVITIES}_${SUBDIRS_TEST}_${m}_band_${BANDWIDTH}_subband_${SUB_BAND}.for_machine.pkl"
            dst_metrics="outputs/gridsearch_models/for_machine/test_${tag}.for_machine.pkl"
            mv "$src_metrics" "$dst_metrics"

            src_ant_metrics="outputs/change_number_antennas_complete_different_${ACTIVITIES}_${SUBDIRS_TEST}_${m}_band_${BANDWIDTH}_subband_${SUB_BAND}.for_machine.pkl"
            mv "$src_ant_metrics" "outputs/gridsearch_models/for_machine/change_number_antennas_${tag}.for_machine.pkl"

            python -c "
import pickle, csv
with open('$dst_metrics', 'rb') as fp:
    d = pickle.load(fp)
with open('$results_file', 'a', newline='') as f:
    csv.writer(f).writerow(['$m', $lr, $do, d['accuracy_single'], d['fscore_single'].mean(),
                            d['accuracy_max_merge'], d['fscore_max_merge'].mean()])
"
        done
    done
done

echo
echo "=================== RESULTS (best combo per model) ==================="
python -c "
import csv
rows = list(csv.DictReader(open('$results_file')))
rows.sort(key=lambda r: (r['model'], -float(r['accuracy_decision_fusion'])))
print('%12s %8s %8s %11s %10s %11s %10s' % ('model', 'lr', 'dropout', 'acc_single', 'f1_single', 'acc_fusion', 'f1_fusion'))
seen = set()
for r in rows:
    star = '*' if r['model'] not in seen else ' '
    seen.add(r['model'])
    print('%s%11s %8s %8s %11.4f %10.4f %11.4f %10.4f' % (star, r['model'], r['learning_rate'], r['dropout'],
          float(r['accuracy_single']), float(r['fscore_single']),
          float(r['accuracy_decision_fusion']), float(r['fscore_decision_fusion'])))
print()
print('* = best combo for that model (by decision-fusion accuracy)')
"
echo
echo "Full CSV: $results_file"
