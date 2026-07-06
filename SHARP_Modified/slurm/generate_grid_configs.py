"""Generate slurm/grid_configs.txt for the 09_train_grid.sh SLURM array job.

One line per run:  model|learning_rate|dropout|hparams-JSON
('-' for learning_rate/dropout means: use the model's builtin default; hparams
{} means the baseline architecture.)

Edit the lists below and re-run to regenerate:
    python generate_grid_configs.py
The cnn is excluded on purpose -- locked to the original SHARP architecture.
"""
import itertools
import json
import os

LEARNING_RATES = [0.0001, 0.0003]
DROPOUTS = [0.2, 0.3, 0.4]

# Architecture variants per Keras model ({} = baseline).
KERAS_VARIANTS = {
    'lstm': [{}, {'units': 128}, {'units': 128, 'num_layers': 3}, {'units': 32}],
    'bilstm': [{}, {'units': 128}, {'num_layers': 1}, {'merge_mode': 'sum'}],
    'rcnn': [{}, {'units': 128}, {'num_layers': 1}, {'num_filters': [16, 32, 64]}],
    'cnn_bilstm': [{}, {'units': 128}, {'num_filters': [16, 32, 64]}, {'dense_units': 64}],
    'vit': [{}, {'dim': 128, 'heads': 8}, {'depth': 6}, {'patch': [20, 20]}],
}
# widar3 keeps its original lr/dropout neighborhood (paper: lr 0.001, dropout 0.5).
WIDAR3_LRS = [0.001, 0.0003]
WIDAR3_DROPOUTS = [0.3, 0.5]
WIDAR3_VARIANTS = [{}, {'frame_len': 20}, {'gru_units': 64}, {'dense_units': 128}]

# Classical models: constructor grids (lr/dropout not applicable).
SKLEARN_GRIDS = {
    'svm': [{'C': c, 'kernel': k, 'pca_components': p}
            for c, k, p in itertools.product([1, 10], ['rbf', 'linear'], [128, 256])],
    'knn': [{'n_neighbors': k, 'weights': w}
            for k, w in itertools.product([5, 11, 25], ['uniform', 'distance'])],
    'gradient_boosting': [{'n_estimators': n, 'learning_rate': lr, 'pca_components': p}
                          for n, lr, p in itertools.product([100, 500], [0.1, 0.05], [128, 256])],
    'naive_bayes': [{'var_smoothing': v, 'pca_components': p}
                    for v, p in itertools.product([1e-9, 1e-6], [128, 256])],
    'random_forest': [{'max_depth': d, 'max_features': f}
                      for d, f in itertools.product([None, 20], ['sqrt', 0.1])],
}

lines = []
for model, variants in KERAS_VARIANTS.items():
    for lr, do, hp in itertools.product(LEARNING_RATES, DROPOUTS, variants):
        lines.append('%s|%s|%s|%s' % (model, lr, do, json.dumps(hp)))
for lr, do, hp in itertools.product(WIDAR3_LRS, WIDAR3_DROPOUTS, WIDAR3_VARIANTS):
    lines.append('widar3|%s|%s|%s' % (lr, do, json.dumps(hp)))
for model, grid in SKLEARN_GRIDS.items():
    for hp in grid:
        lines.append('%s|-|-|%s' % (model, json.dumps(hp)))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'grid_configs.txt')
with open(out, 'w') as fp:
    fp.write('\n'.join(lines) + '\n')
print('wrote %d configs to %s' % (len(lines), out))
per_model = {}
for ln in lines:
    per_model[ln.split('|')[0]] = per_model.get(ln.split('|')[0], 0) + 1
for m, n in sorted(per_model.items()):
    print('  %-18s %d' % (m, n))
