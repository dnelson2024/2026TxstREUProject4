#!/bin/bash
# Collect results from all finished 09_train_grid.sh array tasks into one CSV
# and print a best-per-model table. Run on the LEAP2 login node (no sbatch needed):
#   bash slurm/10_collect_grid_results.sh
# Writes: $PYTHON_CODE_DIR/gridruns/grid_results.csv
set -euo pipefail
source "$(dirname "$0")/config.sh"
activate_env

python - "$PYTHON_CODE_DIR/gridruns" "$ACTIVITIES" <<'EOF'
import csv, glob, os, pickle, sys

runs_dir, activities = sys.argv[1], sys.argv[2]
rows = []
for cfg_path in sorted(glob.glob(os.path.join(runs_dir, '*', 'config.txt'))):
    task_dir = os.path.dirname(cfg_path)
    with open(cfg_path) as fp:
        model, lr, do, hp = fp.read().strip().split('|', 3)
    pkls = glob.glob(os.path.join(task_dir, 'outputs',
                                  'complete_different_%s_S7a_%s_band_*.for_machine.pkl' % (activities, model)))
    if not pkls:
        print('skipping (no result -- task failed or still running): %s' % os.path.basename(task_dir))
        continue
    with open(pkls[0], 'rb') as fp:
        d = pickle.load(fp)
    rows.append({'task': os.path.basename(task_dir), 'model': model, 'learning_rate': lr,
                 'dropout': do, 'hparams': hp,
                 'accuracy_single': round(float(d['accuracy_single']), 4),
                 'fscore_single': round(float(d['fscore_single'].mean()), 4),
                 'accuracy_decision_fusion': round(float(d['accuracy_max_merge']), 4),
                 'fscore_decision_fusion': round(float(d['fscore_max_merge'].mean()), 4)})

if not rows:
    raise SystemExit('no finished tasks found under %s' % runs_dir)

out = os.path.join(runs_dir, 'grid_results.csv')
with open(out, 'w', newline='') as fp:
    w = csv.DictWriter(fp, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print('\n%d results -> %s\n' % (len(rows), out))

rows.sort(key=lambda r: (r['model'], -r['accuracy_decision_fusion']))
print('%-18s %-8s %-7s %-40s %11s %11s' % ('model', 'lr', 'drop', 'hparams', 'acc_fusion', 'f1_fusion'))
seen = set()
for r in rows:
    star = '*' if r['model'] not in seen else ' '
    seen.add(r['model'])
    print('%s%-17s %-8s %-7s %-40s %11.4f %11.4f' % (star, r['model'], r['learning_rate'],
          r['dropout'], r['hparams'][:40], r['accuracy_decision_fusion'], r['fscore_decision_fusion']))
print('\n* = best per model (decision-fusion accuracy on S7a)')
EOF
