"""Train a SHARP-zoo model on the Widar3.0 raw-CSI Doppler dataset.

Data: widar_doppler_data_HAR/{train,val,test}_X.npy  (n, 3 antennas, 340, 100) f16
      built by csi_pipeline.py (REPRESENTATION=raw). Each antenna is trained
      as its own sample (SHARP antenna expansion); at eval the 3 antennas of a
      recording are merged SHARP-style (majority vote, ties -> summed scores)
      = decision fusion.

Record-only-if-better guard: the results row / model / plots for a model are
only overwritten when the new test decision-fusion accuracy beats the one
already recorded in results_widar.csv.

Usage (from the repo root, CPU is fine):
    python Widar3.0/train_widar.py rcnn [--lr 1e-4] [--epochs 25] [--batch 32]
                                        [--crop 130] [--gestures "Clap,Slide,..."]
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BASE)
sys.path.insert(0, REPO)

DATA = os.environ.get(
    'WIDAR_OUT', '/home/danelson/proj4-txstreu/2026TxstREUProject4-data/widar_doppler_data_HAR')
MODELS_DIR = os.path.join(REPO, 'models')
PLOTS_DIR = os.path.join(BASE, 'plots', 'results')


def load_split(split, crop=0, remap=None):
    X = np.load(os.path.join(DATA, split + '_X.npy'), mmap_mode='r')
    y = np.load(os.path.join(DATA, split + '_y.npy'))
    if remap is not None:                     # keep a gesture subset, relabel 0..k-1
        mask = np.isin(y, np.fromiter(remap.keys(), dtype=np.int64))
        X = X[mask]
        y = np.array([remap[v] for v in y[mask]], dtype=np.int64)
    n, n_ant = X.shape[0], X.shape[1]
    if crop:
        # Drop the symmetric noise-floor padding: >=60 constant trailing slices
        # drive a last-state LSTM readout to an input-independent fixed point
        # (all models froze at chance / loss ln(9) on the full 340).
        s = (X.shape[2] - crop) // 2
        X = X[:, :, s:s + crop]
    # (n, ant, T, 100) -> (n*ant, T, 100, 1); C-order keeps a recording's
    # antennas consecutive, which the fusion step below relies on.
    Xe = np.asarray(X, dtype=np.float32).reshape(n * n_ant, X.shape[2], X.shape[3], 1)
    ye = np.repeat(y, n_ant)
    return Xe, ye, y, n_ant


def fuse(pred, y_rec, n_ant):
    """SHARP max-merge: per-recording majority vote over the antennas' argmax,
    ties broken by the summed prediction scores."""
    lab_single = pred.argmax(axis=1)
    fused = np.zeros_like(y_rec)
    for i in range(len(y_rec)):
        votes = lab_single[i * n_ant:(i + 1) * n_ant]
        uniq, cnt = np.unique(votes, return_counts=True)
        best = uniq[cnt == cnt.max()]
        if len(best) == 1:
            fused[i] = best[0]
        else:
            fused[i] = pred[i * n_ant:(i + 1) * n_ant].sum(axis=0).argmax()
    return fused


def macro_f1(y_true, y_pred, n_classes):
    from sklearn.metrics import precision_recall_fscore_support
    _, _, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(n_classes), zero_division=0)
    return f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--crop', type=int, default=130,
                        help='center-crop length on the slice axis (0 = full 340; '
                             'n_bins is 2-199, median 127)')
    parser.add_argument('--gestures', default='',
                        help='comma-separated gesture names to keep (default all 9); '
                             'outputs get a _<k>g suffix and their own '
                             'record-only-if-better namespace')
    args = parser.parse_args()

    import tensorflow as tf
    from SHARP_Modified.Python_code.network_utility import build_model

    labels_all = json.load(open(os.path.join(DATA, 'labels.json')))
    remap, tag = None, ''
    labels = labels_all
    if args.gestures:
        want = [g.strip() for g in args.gestures.split(',') if g.strip()]
        bad = [g for g in want if g not in labels_all]
        if bad:
            sys.exit('unknown gestures %s -- available: %s' % (bad, labels_all))
        keep = sorted(labels_all.index(g) for g in set(want))
        labels = [labels_all[i] for i in keep]
        remap = {orig: new for new, orig in enumerate(keep)}
        tag = '_%dg' % len(labels)
        print('gesture subset: %s' % labels, flush=True)
    results_csv = os.path.join(BASE, 'results_widar%s.csv' % tag)
    n_classes = len(labels)
    Xtr, ytr, _, n_ant = load_split('train', args.crop, remap)
    Xva, yva, yva_rec, _ = load_split('val', args.crop, remap)
    Xte, yte, yte_rec, _ = load_split('test', args.crop, remap)
    print('train %s  val %s  test %s  classes %d' %
          (Xtr.shape, Xva.shape, Xte.shape, n_classes), flush=True)

    t0 = time.time()
    net = build_model(args.model, Xtr.shape[1:], n_classes)
    net.summary()
    net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3,
                                            restore_best_weights=True)
    hist = net.fit(Xtr, ytr, batch_size=args.batch, epochs=args.epochs,
                   validation_data=(Xva, yva), callbacks=[stop], verbose=2)
    epochs_trained = len(hist.history['loss'])   # < args.epochs when early stopping fires
    print('trained in %.1f min (%d epochs)' % ((time.time() - t0) / 60, epochs_trained), flush=True)

    pred_te = net.predict(Xte, batch_size=args.batch, verbose=0)
    # per-single-antenna test cross-entropy (same loss the training curve tracks)
    loss_single = float(net.evaluate(Xte, yte, batch_size=args.batch, verbose=0)[0])
    acc_single = float((pred_te.argmax(axis=1) == yte).mean())
    f_single = macro_f1(yte, pred_te.argmax(axis=1), n_classes)
    fused_te = fuse(pred_te, yte_rec, n_ant)
    acc_fusion = float((fused_te == yte_rec).mean())
    f_fusion = macro_f1(yte_rec, fused_te, n_classes)
    pred_va = net.predict(Xva, batch_size=args.batch, verbose=0)
    acc_val_fusion = float((fuse(pred_va, yva_rec, n_ant) == yva_rec).mean())
    print('[%s] single %.4f | decision-fusion %.4f (macro-F1 %.4f) | val fusion %.4f'
          % (args.model, acc_single, acc_fusion, float(f_fusion.mean()), acc_val_fusion),
          flush=True)

    # ---- record-only-if-better guard (keyed on test decision-fusion acc) ----
    rows = []
    if os.path.exists(results_csv):
        rows = list(csv.DictReader(open(results_csv)))
    old = next((r for r in rows if r['model'] == args.model), None)
    if old and float(old['accuracy_decision_fusion']) >= acc_fusion:
        print('existing %s row is better (%.4f >= %.4f) -- keeping existing results'
              % (args.model, float(old['accuracy_decision_fusion']), acc_fusion))
        return

    row = {'model': args.model, 'lr': args.lr, 'batch': args.batch,
           'accuracy_single': round(acc_single, 4),
           'loss_single': round(loss_single, 4),
           'fscore_single': round(float(f_single.mean()), 4),
           'accuracy_decision_fusion': round(acc_fusion, 4),
           'fscore_decision_fusion': round(float(f_fusion.mean()), 4),
           'accuracy_val_fusion': round(acc_val_fusion, 4)}
    if tag:
        row['gestures'] = '|'.join(labels)
    for name, fs in zip(labels, f_fusion):
        row['fscore_' + name] = round(float(fs), 4)
    rows = [r for r in rows if r['model'] != args.model] + [row]
    rows.sort(key=lambda r: -float(r['accuracy_decision_fusion']))
    with open(results_csv, 'w', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerows(rows)

    os.makedirs(MODELS_DIR, exist_ok=True)
    net.save(os.path.join(MODELS_DIR, 'widar_%s%s.keras' % (args.model, tag)))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ---- training curve (loss + accuracy vs epoch), pdf+png like the SHARP plots ----
    h = hist.history
    epochs_range = range(1, epochs_trained + 1)
    acc_key = 'accuracy' if 'accuracy' in h else \
        next((k for k in h if 'acc' in k and not k.startswith('val_')), None)
    has_acc = acc_key is not None
    figc, axes = plt.subplots(1, 2 if has_acc else 1, figsize=(10 if has_acc else 5.5, 4), squeeze=False)
    axes = axes[0]
    figc.suptitle('%s -- Widar3.0' % args.model, fontsize=14)
    axes[0].plot(epochs_range, h['loss'], label='train')
    axes[0].plot(epochs_range, h['val_loss'], label='validation')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].set_title('Loss')
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True)); axes[0].legend()
    if has_acc:
        axes[1].plot(epochs_range, h[acc_key], label='train')
        axes[1].plot(epochs_range, h['val_' + acc_key], label='validation')
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy'); axes[1].set_title('Accuracy')
        axes[1].xaxis.set_major_locator(MaxNLocator(integer=True)); axes[1].legend()
    figc.tight_layout()
    for ext in ('png', 'pdf'):
        figc.savefig(os.path.join(PLOTS_DIR, 'loss_widar_%s%s.%s' % (args.model, tag, ext)), dpi=150)
    plt.close(figc)

    # confusion matrix (decision fusion), pdf+png like the SHARP plots
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(yte_rec, fused_te, labels=np.arange(n_classes), normalize='true')
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(n_classes), labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_classes), labels, fontsize=8)
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, '%.2f' % cm[i, j], ha='center', va='center', fontsize=7,
                    color='white' if cm[i, j] > 0.5 else 'black')
    ax.set_xlabel('predicted'); ax.set_ylabel('true')
    ax.set_title('%s -- Widar3.0 decision fusion (test acc %.3f, %d epochs)'
                 % (args.model, acc_fusion, epochs_trained))
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(PLOTS_DIR, 'cm_widar_%s%s.%s' % (args.model, tag, ext)), dpi=150)
    plt.close(fig)
    print('recorded %s -> %s' % (args.model, results_csv))


if __name__ == '__main__':
    main()
