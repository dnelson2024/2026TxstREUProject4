
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


def reduction_a_block_small(x_in, base_name):
    x1 = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='valid')(x_in)

    x2 = conv2d_bn(x_in, 5, (2, 2), strides=(2, 2), padding='valid', name=base_name + 'conv2_1_res_a')

    x3 = conv2d_bn(x_in, 3, (1, 1), name=base_name + 'conv3_1_res_a')
    x3 = conv2d_bn(x3, 6, (2, 2), name=base_name + 'conv3_2_res_a')
    x3 = conv2d_bn(x3, 9, (4, 4), strides=(2, 2), padding='same', name=base_name + 'conv3_3_res_a')

    x4 = tf.keras.layers.Concatenate()([x1, x2, x3])
    return x4


def csi_network_inc_res(input_sh, output_sh):
    x_input = tf.keras.Input(input_sh)

    x2 = reduction_a_block_small(x_input, base_name='1st')

    x3 = conv2d_bn(x2, 3, (1, 1), name='conv4')

    x = tf.keras.layers.Flatten()(x3)
    x = tf.keras.layers.Dropout(0.2)(x)
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


class _AddPositionEmbedding(tf.keras.layers.Layer):
    """Learned positional embedding added to a (batch, n_patches, dim) tensor."""
    def __init__(self, n_patches, dim, **kwargs):
        super().__init__(**kwargs)
        self.n_patches = n_patches
        self.pos = tf.keras.layers.Embedding(input_dim=n_patches, output_dim=dim)

    def call(self, x):
        positions = tf.range(start=0, limit=self.n_patches, delta=1)
        return x + self.pos(positions)


def _to_time_sequence(x_in):
    """(T, F, C) -> (T, F*C): one feature vector per time step for RNNs."""
    t = x_in.shape[1]
    return tf.keras.layers.Reshape((t, -1))(x_in)


def build_cnn(input_sh, output_sh):
    # The original SHARP Inception-ResNet CNN (kept as the default 'cnn').
    return csi_network_inc_res(input_sh, output_sh)


def build_lstm(input_sh, output_sh):
    x_in = tf.keras.Input(input_sh)
    x = _to_time_sequence(x_in)
    x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(output_sh, activation=None)(x)
    return tf.keras.Model(x_in, x, name='lstm')


def build_bilstm(input_sh, output_sh):
    x_in = tf.keras.Input(input_sh)
    x = _to_time_sequence(x_in)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(output_sh, activation=None)(x)
    return tf.keras.Model(x_in, x, name='bilstm')


def _conv_front_end(x_in):
    """Conv feature extractor that POOLS ONLY the feature axis, keeping the time
    axis intact so it can feed a recurrent layer as a per-timestep sequence."""
    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x_in)
    x = tf.keras.layers.MaxPool2D((1, 2))(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPool2D((1, 2))(x)
    # (T, F', 32) -> (T, F'*32)
    t, f, c = x.shape[1], x.shape[2], x.shape[3]
    x = tf.keras.layers.Reshape((t, f * c))(x)
    return x


def build_cnn_bilstm(input_sh, output_sh):
    x_in = tf.keras.Input(input_sh)
    x = _conv_front_end(x_in)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(output_sh, activation=None)(x)
    return tf.keras.Model(x_in, x, name='cnn_bilstm')


def build_rcnn(input_sh, output_sh):
    # Recurrent-CNN (CRNN): conv feature extractor + UNIdirectional LSTM.
    x_in = tf.keras.Input(input_sh)
    x = _conv_front_end(x_in)
    x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(output_sh, activation=None)(x)
    return tf.keras.Model(x_in, x, name='rcnn')


def build_vit(input_sh, output_sh, patch=(10, 10), dim=64, depth=4, heads=4, mlp_dim=128):
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
        y = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=dim // heads)(y, y)
        x = tf.keras.layers.Add()([x, y])
        y = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        y = tf.keras.layers.Dense(mlp_dim, activation='gelu')(y)
        y = tf.keras.layers.Dense(dim)(y)
        x = tf.keras.layers.Add()([x, y])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(output_sh, activation=None)(x)
    return tf.keras.Model(x_in, x, name='vit')


# Keras builders keyed by --model value. 'random_forest' is handled separately
# in CSI_network.py (sklearn, not a Keras model).
KERAS_MODEL_BUILDERS = {
    'cnn': build_cnn,
    'cnn_bilstm': build_cnn_bilstm,
    'vit': build_vit,
    'bilstm': build_bilstm,
    'lstm': build_lstm,
    'rcnn': build_rcnn,
}


def build_model(model_name, input_sh, output_sh):
    if model_name not in KERAS_MODEL_BUILDERS:
        raise ValueError('Unknown Keras model %r. Options: %s'
                         % (model_name, list(KERAS_MODEL_BUILDERS)))
    return KERAS_MODEL_BUILDERS[model_name](input_sh, output_sh)
