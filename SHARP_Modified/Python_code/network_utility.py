
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

import tensorflow as tf


def conv2d_bn(x_in, filters, kernel_size, strides=(1, 1), padding='same', activation='relu', bn=False, name=None):
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, name=name)(x_in)
    if bn:
        bn_name = None if name is None else name + '_bn'
        x = tf.keras.layers.BatchNormalization(axis=3, name=bn_name)(x)
    if activation is not None:
        x = tf.keras.layers.Activation(activation)(x)
    return x


def reduction_a_block_small(x_in, base_name, filter_scale=1.0):
    x1 = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='valid')(x_in)

    f2 = max(1, round(5 * filter_scale))
    x2 = conv2d_bn(x_in, f2, (2, 2), strides=(2, 2), padding='valid', name=base_name + 'conv2_1_res_a')

    f3a, f3b, f3c = (max(1, round(n * filter_scale)) for n in (3, 6, 9))
    x3 = conv2d_bn(x_in, f3a, (1, 1), name=base_name + 'conv3_1_res_a')
    x3 = conv2d_bn(x3, f3b, (2, 2), name=base_name + 'conv3_2_res_a')
    x3 = conv2d_bn(x3, f3c, (4, 4), strides=(2, 2), padding='same', name=base_name + 'conv3_3_res_a')

    x4 = tf.keras.layers.Concatenate()([x1, x2, x3])
    return x4


def csi_network_inc_res(input_sh, output_sh, dropout=0.2, filter_scale=1.0):
    x_input = tf.keras.Input(input_sh)

    x2 = reduction_a_block_small(x_input, base_name='1st', filter_scale=filter_scale)

    x3 = conv2d_bn(x2, 3, (1, 1), name='conv4')

    x = tf.keras.layers.Flatten()(x3)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(output_sh, activation=None, name='dense2')(x)
    model = tf.keras.Model(inputs=x_input, outputs=x, name='csi_model')
    return model


# ============================================================================
# Additional architectures. Every builder takes:
#     input_sh  = (sample_length, feature_length, channels)  # time x velocity x C
#     output_sh = number of activity classes
# and returns a Keras model whose LAST layer outputs raw LOGITS (no softmax), to
# stay compatible with SparseCategoricalCrossentropy(from_logits=True) and the
# argmax-based evaluation in CSI_network.py.
# ============================================================================


@tf.keras.utils.register_keras_serializable(package='SHARP')
class _AddPositionEmbedding(tf.keras.layers.Layer):
    """Learned positional embedding added to a (batch, n_patches, dim) tensor."""
    def __init__(self, n_patches, dim, **kwargs):
        super().__init__(**kwargs)
        self.n_patches = n_patches
        self.dim = dim
        self.pos = tf.keras.layers.Embedding(input_dim=n_patches, output_dim=dim)

    def call(self, x):
        positions = tf.range(start=0, limit=self.n_patches, delta=1)
        return x + self.pos(positions)

    def get_config(self):
        config = super().get_config()
        config.update({'n_patches': self.n_patches, 'dim': self.dim})
        return config


def _to_time_sequence(x_in):
    """(T, F, C) -> (T, F*C): one feature vector per time step for RNNs."""
    t = x_in.shape[1]
    return tf.keras.layers.Reshape((t, -1))(x_in)


def build_cnn(input_sh, output_sh, dropout=0.2, filter_scale=1.0):
    # The original SHARP Inception-ResNet CNN (kept as the default 'cnn').
    return csi_network_inc_res(input_sh, output_sh, dropout=dropout, filter_scale=filter_scale)


def build_lstm(input_sh, output_sh, dropout=0.3, units=64, num_layers=2, recurrent_dropout=0.0):
    # NOTE: recurrent_dropout > 0 disables the cuDNN fast path -> much slower on GPU.
    x_in = tf.keras.Input(input_sh)
    x = _to_time_sequence(x_in)
    for i in range(num_layers):
        x = tf.keras.layers.LSTM(units, return_sequences=(i < num_layers - 1),
                                 recurrent_dropout=recurrent_dropout)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(output_sh, activation=None)(x)
    return tf.keras.Model(x_in, x, name='lstm')


def build_bilstm(input_sh, output_sh, dropout=0.3, units=64, num_layers=2, recurrent_dropout=0.0,
                 merge_mode='concat'):
    x_in = tf.keras.Input(input_sh)
    x = _to_time_sequence(x_in)
    for i in range(num_layers):
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units, return_sequences=(i < num_layers - 1),
                                 recurrent_dropout=recurrent_dropout),
            merge_mode=merge_mode)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(output_sh, activation=None)(x)
    return tf.keras.Model(x_in, x, name='bilstm')


def _conv_front_end(x_in, num_filters=(16, 32), kernel_size=(3, 3), pool_size=2):
    """Conv feature extractor that POOLS ONLY the feature axis, keeping the time
    axis intact so it can feed a recurrent layer as a per-timestep sequence."""
    x = x_in
    for filters in num_filters:
        x = tf.keras.layers.Conv2D(filters, tuple(kernel_size), padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPool2D((1, pool_size))(x)
    # (T, F', C') -> (T, F'*C')
    t, f, c = x.shape[1], x.shape[2], x.shape[3]
    x = tf.keras.layers.Reshape((t, f * c))(x)
    return x


def build_cnn_bilstm(input_sh, output_sh, dropout=0.3, num_filters=(16, 32), kernel_size=(3, 3),
                     pool_size=2, units=64, num_layers=2, merge_mode='concat', dense_units=0):
    x_in = tf.keras.Input(input_sh)
    x = _conv_front_end(x_in, num_filters=num_filters, kernel_size=kernel_size, pool_size=pool_size)
    for i in range(num_layers):
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units, return_sequences=(i < num_layers - 1)),
            merge_mode=merge_mode)(x)
    if dense_units:
        x = tf.keras.layers.Dense(dense_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(output_sh, activation=None)(x)
    return tf.keras.Model(x_in, x, name='cnn_bilstm')


def build_rcnn(input_sh, output_sh, dropout=0.3, num_filters=(16, 32), kernel_size=(3, 3),
               pool_size=2, units=64, num_layers=2):
    # Recurrent-CNN (CRNN): conv feature extractor + UNIdirectional LSTM.
    x_in = tf.keras.Input(input_sh)
    x = _conv_front_end(x_in, num_filters=num_filters, kernel_size=kernel_size, pool_size=pool_size)
    for i in range(num_layers):
        x = tf.keras.layers.LSTM(units, return_sequences=(i < num_layers - 1))(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(output_sh, activation=None)(x)
    return tf.keras.Model(x_in, x, name='rcnn')


def build_vit(input_sh, output_sh, patch=(10, 10), dim=64, depth=4, heads=4, mlp_dim=None,
              mlp_ratio=2.0, dropout=0.3, attention_dropout=0.0):
    # mlp_dim=None derives the feedforward width from mlp_ratio (original default:
    # dim=64 * 2.0 = 128, identical to the previous hardcoded mlp_dim=128).
    if mlp_dim is None:
        mlp_dim = int(dim * mlp_ratio)
    x_in = tf.keras.Input(input_sh)
    # Patch embedding via a strided conv (valid padding floors non-divisible dims).
    ph = min(patch[0], input_sh[0])
    pw = min(patch[1], input_sh[1])
    x = tf.keras.layers.Conv2D(dim, (ph, pw), strides=(ph, pw), padding='valid')(x_in)
    n_patches = x.shape[1] * x.shape[2]
    x = tf.keras.layers.Reshape((n_patches, dim))(x)
    x = _AddPositionEmbedding(n_patches, dim)(x)
    for _ in range(depth):
        y = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        y = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=dim // heads,
                                               dropout=attention_dropout)(y, y)
        x = tf.keras.layers.Add()([x, y])
        y = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        y = tf.keras.layers.Dense(mlp_dim, activation='gelu')(y)
        y = tf.keras.layers.Dense(dim)(y)
        x = tf.keras.layers.Add()([x, y])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(output_sh, activation=None)(x)
    return tf.keras.Model(x_in, x, name='vit')


def build_widar3(input_sh, output_sh, frame_len=10, conv_filters=16, kernel_size=(5, 5),
                 pool_size=(2, 2), dense_units=64, gru_units=128, dropout=0.5):
    """Port of the Widar3.0 recognition DNN (Widar3.0/DNN_Model/widar3_keras.py):
    a per-frame CNN encoder (Conv2D -> MaxPool -> Flatten -> Dense -> Dropout -> Dense)
    followed by a GRU over the frame sequence. Widar3.0 consumes [T,20,20,1] BVP
    frames; SHARP provides one [340,100,1] Doppler spectrogram per antenna, so
    consecutive frame_len-step slices of the spectrogram serve as the "frames":
    (340,100,1) -> (34, 10, 100, 1) with the default frame_len=10.
    Defaults follow the original: 16 conv filters @5x5, pool 2x2, dense 64,
    GRU 128, dropout 0.5. (Trainer differs from the original script: Adam +
    logits/SparseCategoricalCrossentropy, the shared setup of this pipeline,
    instead of RMSprop + softmax/categorical_crossentropy.)"""
    t, f, c = input_sh
    if t % frame_len:
        raise ValueError('frame_len %d must divide the %d time steps' % (frame_len, t))
    n_frames = t // frame_len
    td = tf.keras.layers.TimeDistributed
    x_in = tf.keras.Input(input_sh)
    x = tf.keras.layers.Reshape((n_frames, frame_len, f, c))(x_in)
    x = td(tf.keras.layers.Conv2D(conv_filters, tuple(kernel_size), activation='relu'))(x)
    x = td(tf.keras.layers.MaxPooling2D(tuple(pool_size)))(x)
    x = td(tf.keras.layers.Flatten())(x)
    x = td(tf.keras.layers.Dense(dense_units, activation='relu'))(x)
    x = td(tf.keras.layers.Dropout(dropout))(x)
    x = td(tf.keras.layers.Dense(dense_units, activation='relu'))(x)
    x = tf.keras.layers.GRU(gru_units, return_sequences=False)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(output_sh, activation=None)(x)
    return tf.keras.Model(x_in, x, name='widar3')


def _res_block(x, filters, stride=1):
    shortcut = x
    y = tf.keras.layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False)(y)
    y = tf.keras.layers.BatchNormalization()(y)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, use_bias=False)(x)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    return tf.keras.layers.ReLU()(tf.keras.layers.Add()([y, shortcut]))


def build_resnet18(input_sh, output_sh, dropout=0.3, base_filters=64):
    # Standard ResNet-18 (4 stages x 2 basic blocks) on the Doppler spectrogram.
    x_in = tf.keras.Input(input_sh)
    x = tf.keras.layers.Conv2D(base_filters, 7, strides=2, padding='same', use_bias=False)(x_in)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)
    for stage, filters in enumerate([base_filters, base_filters * 2,
                                     base_filters * 4, base_filters * 8]):
        for block in range(2):
            x = _res_block(x, filters, stride=2 if (stage > 0 and block == 0) else 1)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(output_sh, activation=None)(x)
    return tf.keras.Model(x_in, x, name='resnet18')


# Keras builders keyed by --model value. 'random_forest' is handled separately
# in CSI_network.py (sklearn, not a Keras model).
# 'cnn_tuned'/'widar3_tuned' are the SAME architectures as 'cnn'/'widar3' under a
# separate name: 'cnn' stays locked to the original SHARP hyperparameters, while
# cnn_tuned accepts --hparams (dropout/filter_scale) freely; the *_tuned names also
# keep their trained checkpoints/outputs separate from the baseline versions.
KERAS_MODEL_BUILDERS = {
    'cnn': build_cnn,
    'cnn_tuned': build_cnn,
    'cnn_bilstm': build_cnn_bilstm,
    'vit': build_vit,
    'bilstm': build_bilstm,
    'lstm': build_lstm,
    'rcnn': build_rcnn,
    'widar3': build_widar3,
    'widar3_tuned': build_widar3,
    'resnet18': build_resnet18,
}


def build_model(model_name, input_sh, output_sh, dropout=None, filter_scale=1.0, **hparams):
    if model_name not in KERAS_MODEL_BUILDERS:
        raise ValueError('Unknown Keras model %r. Options: %s'
                         % (model_name, list(KERAS_MODEL_BUILDERS)))
    # The cnn is the original SHARP architecture and stays EXACTLY as published --
    # no extra hyperparameters beyond dropout/filter_scale are accepted for it.
    if model_name == 'cnn' and hparams:
        raise ValueError('The cnn architecture is locked to the original SHARP design; '
                         '--hparams is not supported for it (got %r)' % sorted(hparams))
    # widar3 is likewise locked to the original Widar3.0 settings (frame_len 10,
    # 16@5x5 conv, dense 64, GRU 128, dropout 0.5); use widar3_tuned to vary them.
    if model_name == 'widar3' and (hparams or dropout is not None):
        raise ValueError('widar3 is locked to the original Widar3.0 configuration; '
                         'use --model widar3_tuned to change dropout/hparams')
    # dropout=None keeps each builder's own default (0.2 for cnn, 0.3 for the rest);
    # filter_scale only exists for the cnn architecture.
    kwargs = dict(hparams)
    if dropout is not None:
        kwargs['dropout'] = dropout
    if model_name in ('cnn', 'cnn_tuned'):
        kwargs['filter_scale'] = filter_scale
    return KERAS_MODEL_BUILDERS[model_name](input_sh, output_sh, **kwargs)


# ============================================================================
# Classical (scikit-learn) models. These operate on the flattened single-antenna
# Doppler window (340*100 = 34,000 values). random_forest handles that raw
# dimensionality fine; the others get a StandardScaler+PCA(128) front end inside
# the pipeline (fit on train only) or they would be intractably slow / poor in
# 34k dimensions.
# ============================================================================

SKLEARN_MODELS = ('random_forest', 'svm', 'knn', 'gradient_boosting', 'naive_bayes')


def build_sklearn_model(model_name, **hparams):
    """hparams are forwarded to the classifier's constructor (sklearn's own names:
    random_forest n_estimators/max_depth/min_samples_split/min_samples_leaf/max_features,
    svm C/kernel/gamma/degree, knn n_neighbors/weights/metric/p,
    gradient_boosting n_estimators/learning_rate/max_depth/l2_regularization,
    naive_bayes var_smoothing). 'pca_components' tunes the PCA front end
    (svm/knn/gradient_boosting/naive_bayes only)."""
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    pca_components = hparams.pop('pca_components', None)
    if model_name == 'random_forest':
        if pca_components is not None:
            raise ValueError('random_forest runs on the raw features (no PCA front end)')
        params = dict(n_estimators=200, n_jobs=-1, random_state=42)
        params.update(hparams)
        return RandomForestClassifier(**params)
    if pca_components is None:
        pca_components = 128
    if model_name == 'svm':
        # no probability=True (5x slower); decision_function margins get
        # softmaxed in sklearn_class_scores instead.
        params = dict(kernel='rbf', random_state=42)
        params.update(hparams)
        return make_pipeline(StandardScaler(), PCA(n_components=pca_components, random_state=42),
                             SVC(**params))
    if model_name == 'knn':
        params = dict(n_neighbors=5, n_jobs=-1)
        params.update(hparams)
        return make_pipeline(StandardScaler(), PCA(n_components=pca_components, random_state=42),
                             KNeighborsClassifier(**params))
    if model_name == 'gradient_boosting':
        if 'n_estimators' in hparams:  # sklearn's HistGB calls it max_iter
            hparams['max_iter'] = hparams.pop('n_estimators')
        params = dict(random_state=42)
        params.update(hparams)
        return make_pipeline(PCA(n_components=pca_components, random_state=42),
                             HistGradientBoostingClassifier(**params))
    if model_name == 'naive_bayes':
        return make_pipeline(PCA(n_components=pca_components, random_state=42), GaussianNB(**hparams))
    raise ValueError('Unknown sklearn model %r. Options: %s' % (model_name, list(SKLEARN_MODELS)))


def sklearn_class_scores(clf, x_mat, n_classes):
    """[N, n_classes] score matrix for the antenna-fusion/argmax evaluation:
    predict_proba when the model has it, else softmaxed decision_function margins.
    Columns are mapped through clf.classes_ so missing classes stay zero."""
    import numpy as np
    if hasattr(clf, 'predict_proba'):
        p = clf.predict_proba(x_mat)
    else:
        d = clf.decision_function(x_mat)
        if d.ndim == 1:
            d = np.stack([-d, d], axis=1)
        d = d - d.max(axis=1, keepdims=True)
        p = np.exp(d)
        p /= p.sum(axis=1, keepdims=True)
    full = np.zeros((x_mat.shape[0], n_classes), dtype=float)
    full[:, clf.classes_.astype(int)] = p
    return full
