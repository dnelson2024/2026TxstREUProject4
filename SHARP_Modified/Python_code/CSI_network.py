
"""
    Copyright (C) 2022 Francesca Meneghello
    contact: meneghello@dei.unipd.it
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import json
import glob
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
import os
from SHARP_Modified.Python_code.dataset_utility import create_dataset_single, expand_antennas, load_data_single
from SHARP_Modified.Python_code.network_utility import *
from SHARP_Modified.Python_code.plots_utility import plt_loss_curve
from sklearn.metrics import precision_recall_fscore_support, accuracy_score



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('subdirs', help='Subdirs for training')
    parser.add_argument('feature_length', help='Length along the feature dimension (height)', type=int)
    parser.add_argument('sample_length', help='Length along the time dimension (width)', type=int)
    parser.add_argument('channels', help='Number of channels', type=int)
    parser.add_argument('batch_size', help='Number of samples in a batch', type=int)
    parser.add_argument('num_tot', help='Number of antenna * number of spatial streams', type=int)
    parser.add_argument('name_base', help='Name base for the files')
    parser.add_argument('activities', help='Activities to be considered')
    parser.add_argument('--grouping', choices=['none', 'presence', 'motion'], default='none',
                        help='train on grouped binary labels: presence = E vs occupied, '
                             'motion = stationary (E,L) vs dynamic (W,R,J)')
    parser.add_argument('--model', help='Model architecture to train (default cnn)', default='cnn',
                        choices=['cnn', 'cnn_bilstm', 'vit', 'bilstm', 'lstm', 'rcnn', 'widar3', 'cnn_tuned', 'widar3_tuned', 'random_forest', 'svm', 'knn', 'gradient_boosting', 'naive_bayes', 'resnet18'])
    parser.add_argument('--bandwidth', help='Bandwidth in [MHz] to select the subcarriers, can be 20, 40, 80 '
                                            '(default 80)', default=80, required=False, type=int)
    parser.add_argument('--sub_band', help='Sub_band idx in [1, 2, 3, 4] for 20 MHz, [1, 2] for 40 MHz '
                                           '(default 1)', default=1, required=False, type=int)
    parser.add_argument('--learning_rate', help='Adam learning rate for the Keras models (default 0.0001)',
                        default=0.0001, required=False, type=float)
    parser.add_argument('--epochs', help='Number of training epochs for the Keras models (default 25). '
                                         'Ignored by the sklearn models (random_forest, svm, knn, '
                                         'gradient_boosting, naive_bayes).',
                        default=25, required=False, type=int)
    parser.add_argument('--dropout', help='Dropout rate for the Keras models (default: each model\'s builtin '
                                          'default -- 0.2 for cnn, 0.3 for the others)',
                        default=None, required=False, type=float)
    parser.add_argument('--filter_scale', help='Scale factor applied to the cnn reduction block filter counts '
                                               '(default 1.0)', default=1.0, required=False, type=float)
    parser.add_argument('--hparams', help='JSON dict of extra model hyperparameters forwarded to the model '
                                          'builder, e.g. \'{"units": 128, "num_layers": 3}\' for lstm/bilstm or '
                                          '\'{"C": 10, "gamma": "scale"}\' for svm. Not supported for the cnn '
                                          '(locked to the original SHARP architecture).',
                        default='{}', required=False)
    args = parser.parse_args()
    extra_hparams = json.loads(args.hparams)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    # Default TF grabs a small fixed pool up front (fatal OOM on the Inception module's
    # parallel branches on unified-memory GPUs like GB10); grow allocation on demand instead.
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    bandwidth = args.bandwidth
    sub_band = args.sub_band

    csi_act = args.activities
    activities = []
    for lab_act in csi_act.split(','):
        activities.append(lab_act)
    activities = np.asarray(activities)
    # binary regrouping of the activity labels (files/caches get a distinct
    # '-<grouping>' tag so nothing collides with the 5-class results)
    GROUPINGS = {'presence': ((0, 1, 1, 1, 1), ('empty', 'occupied')),
                 'motion': ((0, 0, 1, 1, 1), ('stationary', 'dynamic'))}
    group_map, grouped_names = GROUPINGS.get(args.grouping, (None, None))
    csi_act_out = str(csi_act) + ('' if group_map is None else '-' + args.grouping)

    name_base = args.name_base
    cache_dir = 'cache'
    os.makedirs(cache_dir, exist_ok=True)
    # include the model name so different models (e.g. run back-to-back or overlapping
    # in run_models_dgx.sh) never share a cache file and race on the same tempstate
    cache_base = os.path.join(cache_dir, os.path.basename(name_base) + '_' + args.model)
    # glob-remove (not just the finished .data-*/.index) so a tempstate/lockfile left
    # behind by a previous interrupted run can't collide with this one
    for cache_stage in ('train', 'val', 'train_test', 'val_test', 'test', 'test_test'):
        for stale_file in glob.glob(cache_base + '_' + str(csi_act) + '_cache_' + cache_stage + '*'):
            os.remove(stale_file)

    subdirs_training = args.subdirs  # string
    labels_train = []
    all_files_train = []
    labels_val = []
    all_files_val = []
    labels_test = []
    all_files_test = []
    sample_length = args.sample_length
    feature_length = args.feature_length
    channels = args.channels
    num_antennas = args.num_tot
    input_shape = (num_antennas, sample_length, feature_length, channels)
    input_network = (sample_length, feature_length, channels)
    batch_size = args.batch_size
    output_shape = activities.shape[0]
    labels_considered = np.arange(output_shape)
    activities = activities[labels_considered]

    suffix = '.txt'

    for sdir in subdirs_training.split(','):
        exp_save_dir = args.dir + sdir + '/'
        dir_train = args.dir + sdir + '/train_antennas_' + str(csi_act) + '/'
        name_labels = args.dir + sdir + '/labels_train_antennas_' + str(csi_act) + suffix
        with open(name_labels, "rb") as fp:  # Unpickling
            labels_train.extend(pickle.load(fp))
        name_f = args.dir + sdir + '/files_train_antennas_' + str(csi_act) + suffix
        with open(name_f, "rb") as fp:  # Unpickling
            # stored paths are baked in from wherever the dataset was created (e.g. a
            # cluster); re-root them to dir_train so training works from any local copy
            all_files_train.extend([dir_train + os.path.basename(p) for p in pickle.load(fp)])

        dir_val = args.dir + sdir + '/val_antennas_' + str(csi_act) + '/'
        name_labels = args.dir + sdir + '/labels_val_antennas_' + str(csi_act) + suffix
        with open(name_labels, "rb") as fp:  # Unpickling
            labels_val.extend(pickle.load(fp))
        name_f = args.dir + sdir + '/files_val_antennas_' + str(csi_act) + suffix
        with open(name_f, "rb") as fp:  # Unpickling
            all_files_val.extend([dir_val + os.path.basename(p) for p in pickle.load(fp)])

        dir_test = args.dir + sdir + '/test_antennas_' + str(csi_act) + '/'
        name_labels = args.dir + sdir + '/labels_test_antennas_' + str(csi_act) + suffix
        with open(name_labels, "rb") as fp:  # Unpickling
            labels_test.extend(pickle.load(fp))
        name_f = args.dir + sdir + '/files_test_antennas_' + str(csi_act) + suffix
        with open(name_f, "rb") as fp:  # Unpickling
            all_files_test.extend([dir_test + os.path.basename(p) for p in pickle.load(fp)])

    file_train_selected = [all_files_train[idx] for idx in range(len(labels_train)) if labels_train[idx] in
                           labels_considered]
    labels_train_selected = [labels_train[idx] for idx in range(len(labels_train)) if labels_train[idx] in
                             labels_considered]
    if group_map:
        labels_train_selected = [group_map[l] for l in labels_train_selected]

    file_train_selected_expanded, labels_train_selected_expanded, stream_ant_train = \
        expand_antennas(file_train_selected, labels_train_selected, num_antennas)

    name_cache = cache_base + '_' + csi_act_out + '_cache_train'
    dataset_csi_train = create_dataset_single(file_train_selected_expanded, labels_train_selected_expanded,
                                              stream_ant_train, input_network, batch_size,
                                              shuffle=True, cache_file=name_cache)

    file_val_selected = [all_files_val[idx] for idx in range(len(labels_val)) if labels_val[idx] in
                         labels_considered]
    labels_val_selected = [labels_val[idx] for idx in range(len(labels_val)) if labels_val[idx] in
                           labels_considered]
    if group_map:
        labels_val_selected = [group_map[l] for l in labels_val_selected]

    file_val_selected_expanded, labels_val_selected_expanded, stream_ant_val = \
        expand_antennas(file_val_selected, labels_val_selected, num_antennas)

    name_cache_val = cache_base + '_' + csi_act_out + '_cache_val'
    dataset_csi_val = create_dataset_single(file_val_selected_expanded, labels_val_selected_expanded,
                                            stream_ant_val, input_network, batch_size,
                                            shuffle=False, cache_file=name_cache_val)

    file_test_selected = [all_files_test[idx] for idx in range(len(labels_test)) if labels_test[idx] in
                          labels_considered]
    labels_test_selected = [labels_test[idx] for idx in range(len(labels_test)) if labels_test[idx] in
                            labels_considered]
    if group_map:
        labels_test_selected = [group_map[l] for l in labels_test_selected]

    file_test_selected_expanded, labels_test_selected_expanded, stream_ant_test = \
        expand_antennas(file_test_selected, labels_test_selected, num_antennas)

    name_cache_test = cache_base + '_' + csi_act_out + '_cache_test'
    dataset_csi_test = create_dataset_single(file_test_selected_expanded, labels_test_selected_expanded,
                                             stream_ant_test, input_network, batch_size,
                                             shuffle=False, cache_file=name_cache_test)

    if group_map:
        output_shape = len(grouped_names)
        labels_considered = np.arange(output_shape)
        activities = np.asarray(grouped_names)

    num_samples_train = len(file_train_selected_expanded)
    num_samples_val = len(file_val_selected_expanded)
    num_samples_test = len(file_test_selected_expanded)
    train_steps_per_epoch = int(np.ceil(num_samples_train / batch_size))
    val_steps_per_epoch = int(np.ceil(num_samples_val / batch_size))
    test_steps_per_epoch = int(np.ceil(num_samples_test / batch_size))

    # ---- Build & train the chosen model; produce per-(single-antenna)-sample scores ----
    if args.model in SKLEARN_MODELS:
        # sklearn path: flatten each single-antenna Doppler sample into a vector.
        def _materialize(files_exp, streams):
            return np.stack([np.asarray(load_data_single(f, s)).ravel()
                             for f, s in zip(files_exp, streams)])

        x_train_skl = _materialize(file_train_selected_expanded, stream_ant_train)
        x_val_skl = _materialize(file_val_selected_expanded, stream_ant_val)
        x_test_skl = _materialize(file_test_selected_expanded, stream_ant_test)

        clf = build_sklearn_model(args.model, **extra_hparams)
        clf.fit(x_train_skl, np.array(labels_train_selected_expanded))

        train_prediction_list = sklearn_class_scores(clf, x_train_skl, output_shape)
        val_prediction_list = sklearn_class_scores(clf, x_val_skl, output_shape)
        test_prediction_list = sklearn_class_scores(clf, x_test_skl, output_shape)

        import joblib
        joblib.dump(clf, name_base + '_' + csi_act_out + '_' + args.model + '.joblib')
    else:
        csi_model = build_model(args.model, input_network, output_shape,
                                dropout=args.dropout, filter_scale=args.filter_scale, **extra_hparams)
        csi_model.summary()

        optimiz = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        csi_model.compile(optimizer=optimiz, loss=loss,
                          metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

        callback_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                                         min_delta=1e-3, restore_best_weights=True)
        name_model = name_base + '_' + csi_act_out + '_' + args.model + '_network.keras'
        # No ModelCheckpoint: restore_best_weights above already leaves the best epoch in
        # memory, and csi_model.save() below persists it -- a best-only checkpoint would just
        # be overwritten by that save, so it was redundant.

        # One clean line per epoch (with a completion bar) instead of Keras' per-batch
        # progress bar, which floods the run log with hundreds of step-count updates.
        def _epoch_line(epoch, logs):
            logs = logs or {}
            done, total = epoch + 1, args.epochs
            filled = int(round(20 * done / total))
            bar = '#' * filled + '-' * (20 - filled)
            msg = 'epoch %2d/%d [%s] loss %.4f val_loss %.4f' % (
                done, total, bar, logs.get('loss', float('nan')), logs.get('val_loss', float('nan')))
            acc = logs.get('sparse_categorical_accuracy')
            if acc is not None:
                msg += ' acc %.4f val_acc %.4f' % (acc, logs.get('val_sparse_categorical_accuracy', float('nan')))
            print(msg, flush=True)
        callback_epoch = tf.keras.callbacks.LambdaCallback(on_epoch_end=_epoch_line)

        history = csi_model.fit(dataset_csi_train, epochs=args.epochs, steps_per_epoch=train_steps_per_epoch,
                                validation_data=dataset_csi_val, validation_steps=val_steps_per_epoch,
                                callbacks=[callback_stop, callback_epoch], verbose=0)
        csi_model.save(name_model)
        csi_model = tf.keras.models.load_model(name_model)

        os.makedirs('./plots', exist_ok=True)
        plot_name = csi_act_out + '_' + subdirs_training + '_' + args.model + '_band_' + \
                   str(bandwidth) + '_subband_' + str(sub_band)
        plt_loss_curve(history.history, plot_name, model_name=args.model)

        # dataset_csi_val/dataset_csi_test were already read (and truncated mid-write, since
        # validation_steps stops short of a full pass) during fit() above; re-reading their
        # on-disk cache here races with that half-finished write and raises NotFoundError.
        # Build fresh, single-pass, non-repeating datasets for prediction instead -- same
        # pattern already used for dataset_csi_train_test below.
        name_cache_train_test = cache_base + '_' + str(csi_act) + '_cache_train_test'
        dataset_csi_train_test = create_dataset_single(file_train_selected_expanded, labels_train_selected_expanded,
                                                       stream_ant_train, input_network, batch_size,
                                                       shuffle=False, cache_file=name_cache_train_test, prefetch=False)
        name_cache_val_test = cache_base + '_' + str(csi_act) + '_cache_val_test'
        dataset_csi_val_test = create_dataset_single(file_val_selected_expanded, labels_val_selected_expanded,
                                                     stream_ant_val, input_network, batch_size,
                                                     shuffle=False, cache_file=name_cache_val_test, prefetch=False)
        name_cache_test_test = cache_base + '_' + str(csi_act) + '_cache_test_test'
        dataset_csi_test_test = create_dataset_single(file_test_selected_expanded, labels_test_selected_expanded,
                                                      stream_ant_test, input_network, batch_size,
                                                      shuffle=False, cache_file=name_cache_test_test, prefetch=False)
        train_prediction_list = csi_model.predict(dataset_csi_train_test,
                                                  steps=train_steps_per_epoch)[:num_samples_train]
        val_prediction_list = csi_model.predict(dataset_csi_val_test,
                                                steps=val_steps_per_epoch)[:num_samples_val]
        test_prediction_list = csi_model.predict(dataset_csi_test_test,
                                                 steps=test_steps_per_epoch)[:num_samples_test]

    # ---- Shared evaluation (identical for every model) ----
    train_labels_true = np.array(labels_train_selected_expanded)
    train_labels_pred = np.argmax(train_prediction_list, axis=1)
    conf_matrix_train = confusion_matrix(train_labels_true, train_labels_pred)

    val_labels_true = np.array(labels_val_selected_expanded)
    val_labels_pred = np.argmax(val_prediction_list, axis=1)
    conf_matrix_val = confusion_matrix(val_labels_true, val_labels_pred)

    test_labels_true = np.array(labels_test_selected_expanded)
    test_labels_pred = np.argmax(test_prediction_list, axis=1)
    conf_matrix = confusion_matrix(test_labels_true, test_labels_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(test_labels_true,
                                                                   test_labels_pred,
                                                                   labels=labels_considered)
    accuracy = accuracy_score(test_labels_true, test_labels_pred)

    # merge antennas test
    labels_true_merge = np.array(labels_test_selected)
    pred_max_merge = np.zeros_like(labels_test_selected)
    for i_lab in range(len(labels_test_selected)):
        pred_antennas = test_prediction_list[i_lab * num_antennas:(i_lab + 1) * num_antennas, :]
        lab_merge_max = np.argmax(np.sum(pred_antennas, axis=0))

        pred_max_antennas = test_labels_pred[i_lab * num_antennas:(i_lab + 1) * num_antennas]
        lab_unique, count = np.unique(pred_max_antennas, return_counts=True)
        lab_max_merge = -1
        if lab_unique.shape[0] > 1:
            count_argsort = np.flip(np.argsort(count))
            count_sort = count[count_argsort]
            lab_unique_sort = lab_unique[count_argsort]
            if count_sort[0] == count_sort[1] or lab_unique.shape[0] > 2:  # ex aequo between two labels
                lab_max_merge = lab_merge_max
            else:
                lab_max_merge = lab_unique_sort[0]
        else:
            lab_max_merge = lab_unique[0]
        pred_max_merge[i_lab] = lab_max_merge

    conf_matrix_max_merge = confusion_matrix(labels_true_merge, pred_max_merge, labels=labels_considered)
    precision_max_merge, recall_max_merge, fscore_max_merge, _ = \
        precision_recall_fscore_support(labels_true_merge, pred_max_merge, labels=labels_considered)
    accuracy_max_merge = accuracy_score(labels_true_merge, pred_max_merge)

    metrics_matrix_dict = {'conf_matrix': conf_matrix,
                           'accuracy_single': accuracy,
                           'precision_single': precision,
                           'recall_single': recall,
                           'fscore_single': fscore,
                           'conf_matrix_max_merge': conf_matrix_max_merge,
                           'accuracy_max_merge': accuracy_max_merge,
                           'precision_max_merge': precision_max_merge,
                           'recall_max_merge': recall_max_merge,
                           'fscore_max_merge': fscore_max_merge}

    os.makedirs('./outputs', exist_ok=True)
    # .for_machine.pkl: binary pickle for CSI_network_metrics(_plot).py, not human-readable
    name_file = './outputs/test_' + csi_act_out + '_' + subdirs_training + '_' + args.model + '_band_' + \
                str(bandwidth) + '_subband_' + str(sub_band) + '.for_machine.pkl'
    with open(name_file, "wb") as fp:  # Pickling
        pickle.dump(metrics_matrix_dict, fp)

    # impact of the number of antennas
    one_antenna = [[0], [1], [2], [3]]
    two_antennas = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    three_antennas = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    four_antennas = [[0, 1, 2, 3]]
    seq_ant_list = [one_antenna, two_antennas, three_antennas, four_antennas]
    average_accuracy_change_num_ant = np.zeros((num_antennas,))
    average_fscore_change_num_ant = np.zeros((num_antennas,))
    labels_true_merge = np.array(labels_test_selected)
    for ant_n in range(num_antennas):
        seq_ant = seq_ant_list[ant_n]
        num_seq = len(seq_ant)
        for seq_n in range(num_seq):
            pred_max_merge = np.zeros((len(labels_test_selected),))
            ants_selected = seq_ant[seq_n]
            for i_lab in range(len(labels_test_selected)):
                pred_antennas = test_prediction_list[i_lab * num_antennas:(i_lab + 1) * num_antennas, :]
                pred_antennas = pred_antennas[ants_selected, :]

                lab_merge_max = np.argmax(np.sum(pred_antennas, axis=0))

                pred_max_antennas = test_labels_pred[i_lab * num_antennas:(i_lab + 1) * num_antennas]
                pred_max_antennas = pred_max_antennas[ants_selected]
                lab_unique, count = np.unique(pred_max_antennas, return_counts=True)
                lab_max_merge = -1
                if lab_unique.shape[0] > 1:
                    count_argsort = np.flip(np.argsort(count))
                    count_sort = count[count_argsort]
                    lab_unique_sort = lab_unique[count_argsort]
                    if count_sort[0] == count_sort[1] or lab_unique.shape[0] > ant_n - 1:  # ex aequo between two labels
                        lab_max_merge = lab_merge_max
                    else:
                        lab_max_merge = lab_unique_sort[0]
                else:
                    lab_max_merge = lab_unique[0]
                pred_max_merge[i_lab] = lab_max_merge

            _, _, fscore_max_merge, _ = precision_recall_fscore_support(labels_true_merge, pred_max_merge,
                                                                        labels=list(labels_considered))
            accuracy_max_merge = accuracy_score(labels_true_merge, pred_max_merge)

            average_accuracy_change_num_ant[ant_n] += accuracy_max_merge
            average_fscore_change_num_ant[ant_n] += np.mean(fscore_max_merge)

        average_accuracy_change_num_ant[ant_n] = average_accuracy_change_num_ant[ant_n] / num_seq
        average_fscore_change_num_ant[ant_n] = average_fscore_change_num_ant[ant_n] / num_seq

    metrics_matrix_dict = {'average_accuracy_change_num_ant': average_accuracy_change_num_ant,
                           'average_fscore_change_num_ant': average_fscore_change_num_ant}

    name_file = './outputs/change_number_antennas_test_' + csi_act_out + '_' + subdirs_training + '_' + \
                args.model + '_band_' + str(bandwidth) + '_subband_' + str(sub_band) + '.for_machine.pkl'
    with open(name_file, "wb") as fp:  # Pickling
        pickle.dump(metrics_matrix_dict, fp)
