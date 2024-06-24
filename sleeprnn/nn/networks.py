"""networks.py: Module that defines neural network models functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

PATH_THIS_DIR = os.path.dirname(__file__)
PATH_RESOURCES = os.path.join(PATH_THIS_DIR, "..", "..", "resources")

import numpy as np
import tensorflow as tf

from sleeprnn.nn import layers
from sleeprnn.nn import spectrum
from sleeprnn.common import constants
from sleeprnn.common import checks
from sleeprnn.common import pkeys


def dummy_net(inputs, params, training, name="model_dummy"):
    """Dummy network used for debugging purposes."""
    print("Using model DUMMY")
    with tf.variable_scope(name):
        cwt_prebn = None
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        inputs = inputs[:, border_crop:-border_crop]
        # Simulates downsampling by 8
        inputs = inputs[:, ::8]
        # Simulates shape [batch, time, feats]
        inputs = tf.expand_dims(inputs, axis=2)

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            inputs,
            2,
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            name="logits",
        )
        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
    return logits, probabilities, cwt_prebn


def debug_net(inputs, params, training, name="model_debug"):
    """Dummy network used for debugging purposes."""
    print("Using DEBUG net")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        # Shape is [batch, time, n_scales, n_channels]
        outputs, cwt_prebn = layers.cmorlet_layer(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=64,  # params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[pkeys.USE_LOG],
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        if params[pkeys.USE_RELU]:
            outputs = tf.nn.relu(outputs)

        outputs = tf.layers.average_pooling2d(
            inputs=outputs, pool_size=(1, 2), strides=(1, 2)
        )

        # Convolutional stage (standard feed-forward)
        init_filters = params[pkeys.INITIAL_CONV_FILTERS]
        outputs = layers.conv2d_prebn_block(
            outputs,
            16,  # init_filters,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1",
        )

        outputs = layers.conv2d_prebn_block(
            outputs,
            32,  # init_filters,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_2",
        )

        # outputs = layers.conv2d_prebn_block(
        #     outputs,
        #     2,  # init_filters,
        #     training,
        #     batchnorm=params[pkeys.TYPE_BATCHNORM],
        #     downsampling=params[pkeys.CONV_DOWNSAMPLING],
        #     kernel_init=tf.initializers.he_normal(),
        #     name='convblock_3')

        # Flattening for dense part, shape [batch, time, feats]
        outputs = layers.sequence_flatten(outputs, "flatten")

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            128,  # params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=1,
            num_dirs=constants.UNIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            name="logits",
        )
        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
    return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v1(inputs, params, training, name="model_v1"):
    """Wavelet transform and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms. After this, the
    outputs is flatten and is passed to a 2-layers BLSTM.
    The final classification is made with a FC layer with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model') A name for the network.
    """
    print("Using model V1")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=1,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[pkeys.USE_LOG],
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        if params[pkeys.USE_RELU]:
            outputs = tf.nn.relu(outputs)

        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, "flatten")

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            batchnorm_first_lstm=params[pkeys.TYPE_BATCHNORM],
            dropout_first_lstm=None,
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)

        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v4(inputs, params, training, name="model_v4"):
    """Wavelet transform, conv, and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a standard convolutional stage
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v4') A name for the network.
    """
    print("Using model V4")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=1,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[pkeys.USE_LOG],
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        if params[pkeys.USE_RELU]:
            outputs = tf.nn.relu(outputs)

        # Convolutional stage (standard feed-forward)
        init_filters = params[pkeys.INITIAL_CONV_FILTERS]
        outputs = layers.conv2d_prebn_block(
            outputs,
            init_filters,
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1",
        )
        outputs = layers.conv2d_prebn_block(
            outputs,
            init_filters * 2,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_2",
        )
        outputs = layers.conv2d_prebn_block(
            outputs,
            init_filters * 4,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_3",
        )

        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, "flatten")

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v5(inputs, params, training, name="model_v5"):
    """Wavelet transform, conv, and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a standard convolutional stage
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v5') A name for the network.
    """
    print("Using model V5")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=1,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[pkeys.USE_LOG],
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        if params[pkeys.USE_RELU]:
            outputs = tf.nn.relu(outputs)

        # Convolutional stage (standard feed-forward)
        init_filters = params[pkeys.INITIAL_CONV_FILTERS]
        outputs = layers.conv2d_prebn_block(
            outputs,
            init_filters,
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=None,
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1",
        )

        # Maxpool in frequencies, then avg in time
        # shape is [batch, time, freqs, channels]
        outputs = tf.layers.max_pooling2d(
            inputs=outputs, pool_size=(1, 2), strides=(1, 2)
        )
        outputs = tf.layers.average_pooling2d(
            inputs=outputs, pool_size=(2, 1), strides=(2, 1)
        )

        outputs = layers.conv2d_prebn_block(
            outputs,
            init_filters * 2,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=None,
            kernel_init=tf.initializers.he_normal(),
            name="convblock_2",
        )

        # Maxpool in frequencies, then avg in time
        # shape is [batch, time, freqs, channels]
        outputs = tf.layers.max_pooling2d(
            inputs=outputs, pool_size=(1, 2), strides=(1, 2)
        )
        outputs = tf.layers.average_pooling2d(
            inputs=outputs, pool_size=(2, 1), strides=(2, 1)
        )

        outputs = layers.conv2d_prebn_block(
            outputs,
            init_filters * 4,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=None,
            kernel_init=tf.initializers.he_normal(),
            name="convblock_3",
        )

        # Maxpool in frequencies, then avg in time
        # shape is [batch, time, freqs, channels]
        outputs = tf.layers.max_pooling2d(
            inputs=outputs, pool_size=(1, 2), strides=(1, 2)
        )
        outputs = tf.layers.average_pooling2d(
            inputs=outputs, pool_size=(2, 1), strides=(2, 1)
        )

        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, "flatten")

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v6(inputs, params, training, name="model_v6"):
    """Wavelet transform, conv, and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a standard convolutional stage
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v6') A name for the network.
    """
    print("Using model V6")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=1,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[pkeys.USE_LOG],
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        if params[pkeys.USE_RELU]:
            outputs = tf.nn.relu(outputs)

        # Convolutional stage (standard feed-forward)
        init_filters = params[pkeys.INITIAL_CONV_FILTERS]
        outputs = layers.conv2d_prebn_block(
            outputs,
            init_filters,
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=None,
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1",
        )

        # avg in time, then Maxpool in frequencies
        # shape is [batch, time, freqs, channels]
        outputs = tf.layers.average_pooling2d(
            inputs=outputs, pool_size=(2, 1), strides=(2, 1)
        )
        outputs = tf.layers.max_pooling2d(
            inputs=outputs, pool_size=(1, 2), strides=(1, 2)
        )

        outputs = layers.conv2d_prebn_block(
            outputs,
            init_filters * 2,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=None,
            kernel_init=tf.initializers.he_normal(),
            name="convblock_2",
        )

        # avg in time, then Maxpool in frequencies
        # shape is [batch, time, freqs, channels]
        outputs = tf.layers.average_pooling2d(
            inputs=outputs, pool_size=(2, 1), strides=(2, 1)
        )
        outputs = tf.layers.max_pooling2d(
            inputs=outputs, pool_size=(1, 2), strides=(1, 2)
        )

        outputs = layers.conv2d_prebn_block(
            outputs,
            init_filters * 4,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=None,
            kernel_init=tf.initializers.he_normal(),
            name="convblock_3",
        )

        # avg in time, then Maxpool in frequencies
        # shape is [batch, time, freqs, channels]
        outputs = tf.layers.average_pooling2d(
            inputs=outputs, pool_size=(2, 1), strides=(2, 1)
        )
        outputs = tf.layers.max_pooling2d(
            inputs=outputs, pool_size=(1, 2), strides=(1, 2)
        )

        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, "flatten")

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v7(inputs, params, training, name="model_v7"):
    """Wavelet transform, conv, and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a standard convolutional stage
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v7') A name for the network.
    """
    print("Using model V7")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=1,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[pkeys.USE_LOG],
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        if params[pkeys.USE_RELU]:
            outputs = tf.nn.relu(outputs)

        # Convolutional stage (standard feed-forward)
        init_filters = params[pkeys.INITIAL_CONV_FILTERS]
        outputs = layers.conv2d_prebn_block(
            outputs,
            init_filters,
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1",
        )
        outputs = layers.conv2d_prebn_block(
            outputs,
            init_filters,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_2",
        )
        outputs = layers.conv2d_prebn_block(
            outputs,
            init_filters,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_3",
        )

        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, "flatten")

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v7_lite(inputs, params, training, name="model_v7_lite"):
    """Wavelet transform, conv, and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a standard convolutional stage
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v7_lite') A name for the network.
    """
    print("Using model V7 LITE")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=32,  # params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[pkeys.USE_LOG],
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        # outputs = tf.layers.average_pooling2d(
        #     inputs=outputs, pool_size=(1, 2), strides=(1, 2))

        if params[pkeys.USE_RELU]:
            outputs = tf.nn.relu(outputs)

        # Convolutional stage (standard feed-forward)
        init_filters = params[pkeys.INITIAL_CONV_FILTERS]
        outputs = layers.conv2d_prebn_block(
            outputs,
            16,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1",
        )

        outputs = layers.conv2d_prebn_block(
            outputs,
            32,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_2",
        )

        # outputs = layers.conv2d_prebn_block(
        #     outputs,
        #     init_filters,
        #     training,
        #     batchnorm=params[pkeys.TYPE_BATCHNORM],
        #     downsampling=params[pkeys.CONV_DOWNSAMPLING],
        #     kernel_init=tf.initializers.he_normal(),
        #     name='convblock_3')

        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, "flatten")

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v7_litebig(inputs, params, training, name="model_v7_litebig"):
    """Wavelet transform, conv, and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a standard convolutional stage
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v7_litebig') A name for the network.
    """
    print("Using model V7 LITE BIG")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=32,  # params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[pkeys.USE_LOG],
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        # outputs = tf.layers.average_pooling2d(
        #     inputs=outputs, pool_size=(1, 2), strides=(1, 2))

        if params[pkeys.USE_RELU]:
            outputs = tf.nn.relu(outputs)

        # Convolutional stage (standard feed-forward)
        init_filters = params[pkeys.INITIAL_CONV_FILTERS]
        outputs = layers.conv2d_prebn_block(
            outputs,
            32,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1",
        )

        outputs = layers.conv2d_prebn_block(
            outputs,
            32,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_2",
        )

        # outputs = layers.conv2d_prebn_block(
        #     outputs,
        #     init_filters,
        #     training,
        #     batchnorm=params[pkeys.TYPE_BATCHNORM],
        #     downsampling=params[pkeys.CONV_DOWNSAMPLING],
        #     kernel_init=tf.initializers.he_normal(),
        #     name='convblock_3')

        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, "flatten")

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v8(inputs, params, training, name="model_v8"):
    """Wavelet transform, conv, and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a standard convolutional stage
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v8') A name for the network.
    """
    print("Using model V8")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=32,  # params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[pkeys.USE_LOG],
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        if params[pkeys.USE_RELU]:
            outputs = tf.nn.relu(outputs)

        # Split magnitude and phase
        outputs_magnitude = outputs[..., 0:1]
        outputs_phase = outputs[..., 1:]

        # MAGNITUDE PATH

        # Convolutional stage (standard feed-forward)
        init_filters = params[pkeys.INITIAL_CONV_FILTERS]
        outputs_magnitude = layers.conv2d_prebn_block(
            outputs_magnitude,
            16,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1_mag",
        )
        outputs_magnitude = layers.conv2d_prebn_block(
            outputs_magnitude,
            32,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_2_mag",
        )

        # PHASE PATH

        # Convolutional stage (standard feed-forward)
        init_filters = params[pkeys.INITIAL_CONV_FILTERS]
        outputs_phase = layers.conv2d_prebn_block(
            outputs_phase,
            16,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1_pha",
        )
        outputs_phase = layers.conv2d_prebn_block(
            outputs_phase,
            32,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_2_pha",
        )

        # Now concatenate magnitude and phase paths
        outputs = tf.concat([outputs_magnitude, outputs_phase], axis=-1)

        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, "flatten")

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v9(inputs, params, training, name="model_v9"):
    """Wavelet transform, conv, and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a standard convolutional stage
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v9') A name for the network.
    """
    print("Using model V9")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=32,  # params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[pkeys.USE_LOG],
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        if params[pkeys.USE_RELU]:
            outputs = tf.nn.relu(outputs)

        # Split magnitude and phase
        outputs_magnitude = outputs[..., 0:1]
        outputs_phase = outputs[..., 1:]

        # MAGNITUDE PATH

        # Convolutional stage (standard feed-forward)
        init_filters = params[pkeys.INITIAL_CONV_FILTERS]
        outputs_magnitude = layers.conv2d_prebn_block(
            outputs_magnitude,
            16,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1_mag",
        )

        # PHASE PATH

        # Convolutional stage (standard feed-forward)
        init_filters = params[pkeys.INITIAL_CONV_FILTERS]
        outputs_phase = layers.conv2d_prebn_block(
            outputs_phase,
            16,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1_pha",
        )

        # Now concatenate magnitude and phase paths
        outputs = tf.concat([outputs_magnitude, outputs_phase], axis=-1)

        outputs = layers.conv2d_prebn_block(
            outputs,
            32,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_2",
        )

        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, "flatten")

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v10(inputs, params, training, name="model_v10"):
    """Wavelet transform, conv, and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a standard convolutional stage
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v10') A name for the network.
    """
    print("Using model V10")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[pkeys.USE_LOG],
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        if params[pkeys.USE_RELU]:
            outputs = tf.nn.relu(outputs)

        # Convolutional stage (standard feed-forward)
        init_filters = params[pkeys.INITIAL_CONV_FILTERS]
        outputs = layers.conv2d_prebn_block(
            outputs,
            init_filters,
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1",
        )

        outputs = layers.conv2d_prebn_block(
            outputs,
            init_filters,
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_2",
        )

        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, "flatten")

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v11(inputs, params, training, name="model_v11"):
    """conv 1D and BLSTM to make a prediction.

    This models has a standard convolutional stage on time-domain
    (pre-activation BN). After this, the outputs is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v11') A name for the network.
    """
    print("Using model V11 (Time-Domain)")
    with tf.variable_scope(name):

        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        other_outputs_dict = {"last_hidden": outputs}
        return logits, probabilities, other_outputs_dict


def wavelet_blstm_net_v12(inputs, params, training, name="model_v12"):
    """Wavelet transform, conv, and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a standard convolutional stage
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v12') A name for the network.
    """
    print("Using model V12 (cwt)")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[pkeys.USE_LOG],
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        if params[pkeys.USE_RELU]:
            outputs = tf.nn.relu(outputs)

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1",
        )

        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_2",
        )

        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, "flatten")

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v13(inputs, params, training, name="model_v13"):
    """Wavelet transform, conv, and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a standard convolutional stage
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v13') A name for the network.
    """
    print("Using model V13 (cwt using freqs as channels)")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[pkeys.USE_LOG],
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        if params[pkeys.USE_RELU]:
            outputs = tf.nn.relu(outputs)

        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, "flatten")

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_2",
        )

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v14(inputs, params, training, name="model_v14"):
    """Wavelet transform, conv, and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a standard convolutional stage
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v14') A name for the network.
    """
    print("Using model V14 (cwt using freqs as channels)")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=1,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[pkeys.USE_LOG],
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        if params[pkeys.USE_RELU]:
            outputs = tf.nn.relu(outputs)

        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, "flatten")

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_2",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_3",
        )

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v15(inputs, params, training, name="model_v15"):
    """conv1D in time, conv2D in cwt, and BLSTM to make a prediction.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v15') A name for the network.
    """
    print("Using model V15 (Time_11 + CWT_12)")
    with tf.variable_scope(name):

        # ------ TIME PATH

        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs_time = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs_time = tf.expand_dims(inputs_time, axis=2)

        # BN at input
        outputs_time = layers.batchnorm_layer(
            inputs_time,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs_time = layers.conv1d_prebn_block(
            outputs_time,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_time1d_1",
        )

        outputs_time = layers.conv1d_prebn_block(
            outputs_time,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_time1d_2",
        )

        outputs_time = layers.conv1d_prebn_block(
            outputs_time,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_time1d_3",
        )

        # ----- CWT PATH

        outputs_cwt, cwt_prebn = layers.cmorlet_layer(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[pkeys.USE_LOG],
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        if params[pkeys.USE_RELU]:
            outputs_cwt = tf.nn.relu(outputs_cwt)

        # Convolutional stage (standard feed-forward)
        outputs_cwt = layers.conv2d_prebn_block(
            outputs_cwt,
            params[pkeys.CWT_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_cwt2d_1",
        )

        outputs_cwt = layers.conv2d_prebn_block(
            outputs_cwt,
            params[pkeys.CWT_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_cwt2d_2",
        )

        # Flattening for dense part
        outputs_cwt = layers.sequence_flatten(outputs_cwt, "flatten")

        # Concatenate both paths
        outputs = tf.concat([outputs_time, outputs_cwt], axis=-1)

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v16(inputs, params, training, name="model_v16"):
    """conv1D in time, conv1D in cwt, and BLSTM to make a prediction.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v16') A name for the network.
    """
    print("Using model V16 (Time_11 + CWT_13)")
    with tf.variable_scope(name):

        # ------ TIME PATH

        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs_time = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs_time = tf.expand_dims(inputs_time, axis=2)

        # BN at input
        outputs_time = layers.batchnorm_layer(
            inputs_time,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs_time = layers.conv1d_prebn_block(
            outputs_time,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_time1d_1",
        )

        outputs_time = layers.conv1d_prebn_block(
            outputs_time,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_time1d_2",
        )

        outputs_time = layers.conv1d_prebn_block(
            outputs_time,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_time1d_3",
        )

        # ----- CWT PATH

        outputs_cwt, cwt_prebn = layers.cmorlet_layer(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[pkeys.USE_LOG],
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        if params[pkeys.USE_RELU]:
            outputs_cwt = tf.nn.relu(outputs_cwt)

        # Flattening for dense part
        outputs_cwt = layers.sequence_flatten(outputs_cwt, "flatten")

        # Convolutional stage (standard feed-forward)
        outputs_cwt = layers.conv1d_prebn_block(
            outputs_cwt,
            params[pkeys.CWT_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_cwt1d_1",
        )

        outputs_cwt = layers.conv1d_prebn_block(
            outputs_cwt,
            params[pkeys.CWT_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_cwt1d_2",
        )

        # Concatenate both paths
        outputs = tf.concat([outputs_time, outputs_cwt], axis=-1)

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v17(inputs, params, training, name="model_v17"):
    """Wavelet transform, conv, and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a standard convolutional stage
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v17') A name for the network.
    """
    print("Using model V17 (cwt with real and imaginary parts directly)")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer_rectangular(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[pkeys.USE_LOG],
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        if params[pkeys.USE_RELU]:
            outputs = tf.nn.relu(outputs)

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1",
        )

        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_2",
        )

        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, "flatten")

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v18(inputs, params, training, name="model_v18"):
    """conv1D in time, conv2D in cwt, and BLSTM to make a prediction.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v18') A name for the network.
    """
    print("Using model V18 (Time_11 + CWT_17)")
    with tf.variable_scope(name):

        # ------ TIME PATH

        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs_time = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs_time = tf.expand_dims(inputs_time, axis=2)

        # BN at input
        outputs_time = layers.batchnorm_layer(
            inputs_time,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs_time = layers.conv1d_prebn_block(
            outputs_time,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_time1d_1",
        )

        outputs_time = layers.conv1d_prebn_block(
            outputs_time,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_time1d_2",
        )

        outputs_time = layers.conv1d_prebn_block(
            outputs_time,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_time1d_3",
        )

        # ----- CWT PATH

        outputs_cwt, cwt_prebn = layers.cmorlet_layer_rectangular(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            use_log=params[pkeys.USE_LOG],
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        if params[pkeys.USE_RELU]:
            outputs_cwt = tf.nn.relu(outputs_cwt)

        # Convolutional stage (standard feed-forward)
        outputs_cwt = layers.conv2d_prebn_block(
            outputs_cwt,
            params[pkeys.CWT_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_cwt2d_1",
        )

        outputs_cwt = layers.conv2d_prebn_block(
            outputs_cwt,
            params[pkeys.CWT_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_cwt2d_2",
        )

        # Flattening for dense part
        outputs_cwt = layers.sequence_flatten(outputs_cwt, "flatten")

        # Concatenate both paths
        outputs = tf.concat([outputs_time, outputs_cwt], axis=-1)

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v19(inputs, params, training, name="model_v19"):
    """Wavelet transform, conv, and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a standard convolutional stage
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v19') A name for the network.
    """
    print("Using model V19 (general cwt)")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer_general(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            return_real_part=params[pkeys.CWT_RETURN_REAL_PART],
            return_imag_part=params[pkeys.CWT_RETURN_IMAG_PART],
            return_magnitude=params[pkeys.CWT_RETURN_MAGNITUDE],
            return_phase=params[pkeys.CWT_RETURN_PHASE],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1",
        )

        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_2",
        )

        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, "flatten")

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)

        other_outputs_dict = {"last_hidden": outputs}

        return logits, probabilities, other_outputs_dict


def wavelet_blstm_net_v20_concat(inputs, params, training, name="model_v20_concat"):
    """conv 1D and BLSTM to make a prediction.

    This models has a standard convolutional stage on time-domain
    (pre-activation BN). After this, the outputs is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v20_concat') A name for the network.
    """
    print("Using model V20_CONCAT (Time-Domain + sigma band)")
    with tf.variable_scope(name):
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        # compute sigma band
        inputs_sigma = spectrum.compute_sigma_band(
            inputs,
            fs=params[pkeys.FS],
            ntaps=params[pkeys.SIGMA_FILTER_NTAPS],
            border_crop=border_crop,
        )

        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # ----- CONCATENATE INPUT AND SIGMA ACROSS CHANNELS
        inputs = tf.concat([inputs, inputs_sigma], axis=2)
        # -----

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v20_indep(inputs, params, training, name="model_v20_indep"):
    """conv 1D and BLSTM to make a prediction.

    This models has a standard convolutional stage on time-domain
    (pre-activation BN). After this, the outputs is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v20_indep') A name for the network.
    """
    print("Using model V20_INDEP (Time-Domain + sigma band)")
    with tf.variable_scope(name):
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        # compute sigma band
        inputs_sigma = spectrum.compute_sigma_band(
            inputs,
            fs=params[pkeys.FS],
            ntaps=params[pkeys.SIGMA_FILTER_NTAPS],
            border_crop=border_crop,
        )

        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs_original = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs_original = tf.expand_dims(inputs_original, axis=2)

        # Each channel is processed independently

        with tf.variable_scope("tower_original"):
            # BN at input
            outputs_original = layers.batchnorm_layer(
                inputs_original,
                "bn_input",
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                training=training,
            )

            # 1D convolutions expect shape [batch, time_len, n_feats]

            # Convolutional stage (standard feed-forward)
            outputs_original = layers.conv1d_prebn_block(
                outputs_original,
                params[pkeys.TIME_CONV_FILTERS_1],
                training,
                kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                kernel_init=tf.initializers.he_normal(),
                name="convblock_1d_1",
            )

            outputs_original = layers.conv1d_prebn_block(
                outputs_original,
                params[pkeys.TIME_CONV_FILTERS_2],
                training,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                kernel_init=tf.initializers.he_normal(),
                name="convblock_1d_2",
            )

            outputs_original = layers.conv1d_prebn_block(
                outputs_original,
                params[pkeys.TIME_CONV_FILTERS_3],
                training,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                kernel_init=tf.initializers.he_normal(),
                name="convblock_1d_3",
            )

        with tf.variable_scope("tower_sigma"):
            # BN at input
            outputs_sigma = layers.batchnorm_layer(
                inputs_sigma,
                "bn_input",
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                training=training,
            )

            # 1D convolutions expect shape [batch, time_len, n_feats]

            # Convolutional stage (standard feed-forward)
            outputs_sigma = layers.conv1d_prebn_block(
                outputs_sigma,
                params[pkeys.TIME_CONV_FILTERS_1],
                training,
                kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                kernel_init=tf.initializers.he_normal(),
                name="convblock_1d_1",
            )

            outputs_sigma = layers.conv1d_prebn_block(
                outputs_sigma,
                params[pkeys.TIME_CONV_FILTERS_2],
                training,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                kernel_init=tf.initializers.he_normal(),
                name="convblock_1d_2",
            )

            outputs_sigma = layers.conv1d_prebn_block(
                outputs_sigma,
                params[pkeys.TIME_CONV_FILTERS_3],
                training,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                kernel_init=tf.initializers.he_normal(),
                name="convblock_1d_3",
            )

        # Concatenate both paths
        outputs = tf.concat([outputs_original, outputs_sigma], axis=-1)

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v21(inputs, params, training, name="model_v21"):
    """Wavelet transform, conv, and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a standard convolutional stage
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v21') A name for the network.
    """
    print("Using model V21 (timev11 + general cwt v19)")
    with tf.variable_scope(name):

        # ------ TIME PATH

        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs_time = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs_time = tf.expand_dims(inputs_time, axis=2)

        # BN at input
        outputs_time = layers.batchnorm_layer(
            inputs_time,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs_time = layers.conv1d_prebn_block(
            outputs_time,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_time1d_1",
        )

        outputs_time = layers.conv1d_prebn_block(
            outputs_time,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_time1d_2",
        )

        outputs_time = layers.conv1d_prebn_block(
            outputs_time,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_time1d_3",
        )

        # ----- CWT PATH

        outputs_cwt, cwt_prebn = layers.cmorlet_layer_general(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            return_real_part=params[pkeys.CWT_RETURN_REAL_PART],
            return_imag_part=params[pkeys.CWT_RETURN_IMAG_PART],
            return_magnitude=params[pkeys.CWT_RETURN_MAGNITUDE],
            return_phase=params[pkeys.CWT_RETURN_PHASE],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        # Convolutional stage (standard feed-forward)
        outputs_cwt = layers.conv2d_prebn_block(
            outputs_cwt,
            params[pkeys.CWT_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_cwt2d_1",
        )

        outputs_cwt = layers.conv2d_prebn_block(
            outputs_cwt,
            params[pkeys.CWT_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_cwt2d_2",
        )

        # Flattening for dense part
        outputs_cwt = layers.sequence_flatten(outputs_cwt, "flatten")

        # Dropout before concatenation
        outputs_time = layers.dropout_layer(
            outputs_time,
            "drop_time",
            drop_rate=params[pkeys.DROP_RATE_BEFORE_LSTM],
            dropout=params[pkeys.TYPE_DROPOUT],
            training=training,
        )
        outputs_cwt = layers.dropout_layer(
            outputs_cwt,
            "drop_cwt",
            drop_rate=params[pkeys.DROP_RATE_BEFORE_LSTM],
            dropout=params[pkeys.TYPE_DROPOUT],
            training=training,
        )

        # Concatenate both paths
        outputs = tf.concat([outputs_time, outputs_cwt], axis=-1)

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=None,
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=None,
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v22(inputs, params, training, name="model_v22"):
    """Wavelet transform, conv, and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a standard convolutional stage
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v22') A name for the network.
    """
    print("Using model V22 (general cwt with 8 indep branches)")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer_general(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            return_real_part=params[pkeys.CWT_RETURN_REAL_PART],
            return_imag_part=params[pkeys.CWT_RETURN_IMAG_PART],
            return_magnitude=params[pkeys.CWT_RETURN_MAGNITUDE],
            return_phase=params[pkeys.CWT_RETURN_PHASE],
            lower_freq=1,
            upper_freq=20,
            n_scales=8,
            stride=1,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        # Output sequence has shape [batch_size, time_len, n_scales, channels]
        # Unstack scales
        outputs_unstack = tf.unstack(outputs, axis=2)
        # Now we have shape [batch size, time len, channels]

        outputs_after_conv = []
        for i, single_output in enumerate(outputs_unstack):
            with tf.variable_scope("scale_%d" % i):
                # Convolutional stage (standard feed-forward)
                single_output = layers.conv1d_prebn_block(
                    single_output,
                    params[pkeys.CWT_CONV_FILTERS_1],
                    training,
                    kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
                    batchnorm=params[pkeys.TYPE_BATCHNORM],
                    downsampling=params[pkeys.CONV_DOWNSAMPLING],
                    kernel_init=tf.initializers.he_normal(),
                    name="convblock_1",
                )

                single_output = layers.conv1d_prebn_block(
                    single_output,
                    params[pkeys.CWT_CONV_FILTERS_2],
                    training,
                    batchnorm=params[pkeys.TYPE_BATCHNORM],
                    downsampling=params[pkeys.CONV_DOWNSAMPLING],
                    kernel_init=tf.initializers.he_normal(),
                    name="convblock_2",
                )

                single_output = layers.conv1d_prebn_block(
                    single_output,
                    params[pkeys.CWT_CONV_FILTERS_3],
                    training,
                    batchnorm=params[pkeys.TYPE_BATCHNORM],
                    downsampling=params[pkeys.CONV_DOWNSAMPLING],
                    kernel_init=tf.initializers.he_normal(),
                    name="convblock_3",
                )

                # Dropout before concatenation  (ensures same drop across scales)
                single_output = layers.dropout_layer(
                    single_output,
                    "drop",
                    drop_rate=params[pkeys.DROP_RATE_BEFORE_LSTM],
                    dropout=params[pkeys.TYPE_DROPOUT],
                    training=training,
                )

            outputs_after_conv.append(single_output)

        # Concatenate all paths
        outputs = tf.concat(outputs_after_conv, axis=-1)

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v23(inputs, params, training, name="model_v23"):
    """conv 1D and BLSTM to make a prediction.

    This models has a standard convolutional stage on time-domain
    (pre-activation BN). After this, the outputs is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v23') A name for the network.
    """
    print("Using model V23 (Time-Domain)")
    with tf.variable_scope(name):

        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.OUTPUT_LSTM_UNITS] > 0:
            # Smaller LSTM layer for decoding predictions
            outputs = layers.lstm_layer(
                outputs,
                num_units=params[pkeys.OUTPUT_LSTM_UNITS],
                num_dirs=constants.BIDIRECTIONAL,
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                name="lstm_out",
            )

        # Final FC classification layer
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v24(inputs, params, training, name="model_v24"):
    """conv 1D and BLSTM to make a prediction.

    This models has a standard convolutional stage on time-domain
    (pre-activation BN). After this, the outputs is passed to
    a 2-layers BLSTM. Then to upconv layers.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v24') A name for the network.
    """
    print("Using model V24 (Time-Domain with UpConv output)")
    with tf.variable_scope(name):

        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        outputs = layers.dropout_layer(
            outputs,
            "drop_after_lstm",
            drop_rate=params[pkeys.DROP_RATE_HIDDEN],
            dropout=params[pkeys.TYPE_DROPOUT],
            training=training,
        )

        # Upconvolutions to recover resolution
        outputs = layers.upconv1d_prebn(
            outputs,
            params[pkeys.LAST_OUTPUT_CONV_FILTERS] * 4,
            kernel_size=5,
            training=training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            kernel_init=tf.initializers.he_normal(),
            name="upconv_1d_1",
        )

        outputs = layers.upconv1d_prebn(
            outputs,
            params[pkeys.LAST_OUTPUT_CONV_FILTERS] * 2,
            kernel_size=5,
            training=training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            kernel_init=tf.initializers.he_normal(),
            name="upconv_1d_2",
        )

        outputs = layers.upconv1d_prebn(
            outputs,
            params[pkeys.LAST_OUTPUT_CONV_FILTERS],
            kernel_size=5,
            training=training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            kernel_init=tf.initializers.he_normal(),
            name="upconv_1d_3",
        )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            # dropout=params[pkeys.TYPE_DROPOUT],
            # drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v25(inputs, params, training, name="model_v25"):
    """BLSTM with UNET conv1d architecture to make a prediction.

    This models has a standard convolutional stage on time-domain
    (pre-activation BN). After this, the outputs is passed to
    a 2-layers BLSTM. Then to upconv layers with upconv skip connection
    (unet architecture).
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v25') A name for the network.
    """
    print("Using model V25 (Time-Domain Unet-LSTM)")
    with tf.variable_scope(name):

        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )
        print(outputs)

        # 1D convolutions expect shape [batch, time_len, n_feats]

        n_down = params[pkeys.UNET_TIME_N_DOWN]
        output_down = params[pkeys.TOTAL_DOWNSAMPLING_FACTOR]
        n_up = n_down - int(np.log(output_down) / np.log(2))

        outputs_from_down_list = []
        filters_from_down_list = []
        for i in range(n_down):
            this_factor = 2**i
            filters = params[pkeys.UNET_TIME_INITIAL_CONV_FILTERS] * this_factor
            filters_from_down_list.append(filters)
            outputs, outputs_prepool = layers.conv1d_prebn_block_unet_down(
                outputs,
                filters,
                training,
                n_layers=params[pkeys.UNET_TIME_N_CONV_DOWN],
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                kernel_init=tf.initializers.he_normal(),
                name="down_conv1d_%d" % i,
            )
            outputs_from_down_list.append(outputs_prepool)
            print("output before pool", outputs_prepool)
            print("output after pool", outputs)

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.UNET_TIME_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )
        print("outputs_lstm", outputs)

        for i in range(n_up):
            outputs_skip_prepool = outputs_from_down_list[-(i + 1)]
            filters = filters_from_down_list[-(i + 1)]

            print("outputs_skip_prepool", outputs_skip_prepool)

            outputs = layers.conv1d_prebn_block_unet_up(
                outputs,
                outputs_skip_prepool,
                filters,
                training,
                n_layers=params[pkeys.UNET_TIME_N_CONV_UP],
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                kernel_init=tf.initializers.he_normal(),
                name="up_conv1d_%d" % i,
            )
            print("output up", outputs)

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            # dropout=params[pkeys.TYPE_DROPOUT],
            # drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v11_skip(inputs, params, training, name="model_v11_skip"):
    """conv 1D and BLSTM to make a prediction.

    This models has a standard convolutional stage on time-domain
    (pre-activation BN). After this, the outputs is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v11_skip') A name for the network.
    """
    print("Using model V11 Skip (Time-Domain + skip connection)")
    with tf.variable_scope(name):

        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs_conv = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # Multilayer BLSTM (2 layers)
        outputs_lstm = layers.multilayer_lstm_block(
            outputs_conv,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        # Skip connection
        outputs = tf.concat([outputs_lstm, outputs_conv], axis=-1)

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v19_skip(inputs, params, training, name="model_v19_skip"):
    """Wavelet transform, conv, and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a standard convolutional stage
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v19_skip') A name for the network.
    """
    print("Using model V19 Skip (general cwt + skip connection)")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer_general(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            return_real_part=params[pkeys.CWT_RETURN_REAL_PART],
            return_imag_part=params[pkeys.CWT_RETURN_IMAG_PART],
            return_magnitude=params[pkeys.CWT_RETURN_MAGNITUDE],
            return_phase=params[pkeys.CWT_RETURN_PHASE],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1",
        )

        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_2",
        )

        # Flattening for dense part
        outputs_conv = layers.sequence_flatten(outputs, "flatten")

        # Multilayer BLSTM (2 layers)
        outputs_lstm = layers.multilayer_lstm_block(
            outputs_conv,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        # Skip connection
        outputs = tf.concat([outputs_lstm, outputs_conv], axis=-1)

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)

        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v19_skip2(inputs, params, training, name="model_v19_skip2"):
    """Wavelet transform, conv, and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a standard convolutional stage
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v19_skip2') A name for the network.
    """
    print("Using model V19 Skip2 (general cwt + skip connection v2)")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer_general(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            return_real_part=params[pkeys.CWT_RETURN_REAL_PART],
            return_imag_part=params[pkeys.CWT_RETURN_IMAG_PART],
            return_magnitude=params[pkeys.CWT_RETURN_MAGNITUDE],
            return_phase=params[pkeys.CWT_RETURN_PHASE],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        cwt_skip, _ = layers.cmorlet_layer_general(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            return_real_part=False,
            return_imag_part=False,
            return_magnitude=True,
            return_phase=False,
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=64,
            stride=1,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum_skip",
        )
        cwt_skip = tf.layers.average_pooling2d(
            inputs=cwt_skip, pool_size=(8, 1), strides=(8, 1)
        )
        cwt_skip = layers.sequence_flatten(cwt_skip, "cwt_skip_flatten")

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1",
        )

        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_2",
        )

        # Flattening for dense part
        outputs_conv = layers.sequence_flatten(outputs, "flatten")

        # Multilayer BLSTM (2 layers)
        outputs_lstm = layers.multilayer_lstm_block(
            outputs_conv,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        # Skip connection
        outputs = tf.concat([outputs_lstm, outputs_conv, cwt_skip], axis=-1)

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)

        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v19_skip3(inputs, params, training, name="model_v19_skip3"):
    """Wavelet transform, conv, and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a standard convolutional stage
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v19_skip3') A name for the network.
    """
    print("Using model V19 Skip3 (general cwt + skip connection v3)")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer_general(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            return_real_part=params[pkeys.CWT_RETURN_REAL_PART],
            return_imag_part=params[pkeys.CWT_RETURN_IMAG_PART],
            return_magnitude=params[pkeys.CWT_RETURN_MAGNITUDE],
            return_phase=params[pkeys.CWT_RETURN_PHASE],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        cwt_skip, _ = layers.cmorlet_layer_general(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            return_real_part=False,
            return_imag_part=False,
            return_magnitude=True,
            return_phase=False,
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=64,
            stride=1,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum_skip",
        )
        cwt_skip = tf.layers.average_pooling2d(
            inputs=cwt_skip, pool_size=(8, 1), strides=(8, 1)
        )
        cwt_skip = layers.sequence_flatten(cwt_skip, "cwt_skip_flatten")

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1",
        )

        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_2",
        )

        # Flattening for dense part
        outputs_conv = layers.sequence_flatten(outputs, "flatten")

        # Skip connection at the input of lstm (direct access to spectrogram)
        outputs_conv = tf.concat([outputs_conv, cwt_skip], axis=-1)

        # Multilayer BLSTM (2 layers)
        outputs_lstm = layers.multilayer_lstm_block(
            outputs_conv,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        # Skip connection
        outputs = tf.concat([outputs_lstm, outputs_conv, cwt_skip], axis=-1)

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)

        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v26(inputs, params, training, name="model_v26"):
    """Experimental

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v26') A name for the network.
    """
    print("Using model V26 (experimental)")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer_general(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            return_real_part=True,
            return_imag_part=True,
            return_magnitude=True,
            return_phase=False,
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        cwt_skip = outputs[..., 2:3]
        cwt_skip = tf.layers.average_pooling2d(inputs=cwt_skip, pool_size=4, strides=4)
        cwt_skip = layers.sequence_flatten(cwt_skip, "cwt_skip_flatten")

        outputs = outputs[..., :2]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1",
        )

        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_2",
        )

        # Flattening for dense part
        outputs_conv = layers.sequence_flatten(outputs, "flatten")

        # Multilayer BLSTM (2 layers)
        # Skip connection at the input of lstm (direct access to magnitude)
        outputs_lstm = layers.multilayer_lstm_block(
            tf.concat([outputs_conv, cwt_skip], axis=-1),
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        # Two branches
        outputs_lstm = layers.sequence_fc_layer(
            tf.concat([outputs_lstm, cwt_skip], axis=-1),
            params[pkeys.FC_UNITS],
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            activation=tf.nn.relu,
            name="fc_1_lstm",
        )
        outputs_conv = layers.sequence_fc_layer(
            tf.concat([outputs_conv, cwt_skip], axis=-1),
            params[pkeys.FC_UNITS],
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            activation=tf.nn.relu,
            name="fc_1_conv",
        )

        # Skip connection
        outputs = tf.concat([outputs_lstm, outputs_conv], axis=-1)

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)

        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v27(inputs, params, training, name="model_v27"):
    """Experimental

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v27') A name for the network.
    """
    print("Using model V27 (experimental)")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        # CWt skip
        outputs, cwt_prebn = layers.cmorlet_layer_general(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            return_real_part=False,
            return_imag_part=False,
            return_magnitude=True,
            return_phase=False,
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )
        cwt_skip = tf.layers.average_pooling2d(inputs=outputs, pool_size=4, strides=4)
        cwt_skip = layers.sequence_flatten(cwt_skip, "cwt_skip_flatten")

        # Convolutional stage (standard feed-forward)
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs_conv = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # Multilayer BLSTM (2 layers)
        # Skip connection at the input of lstm (direct access to magnitude)
        outputs_lstm = layers.multilayer_lstm_block(
            tf.concat([outputs_conv, cwt_skip], axis=-1),
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        # Two branches
        outputs_lstm = layers.sequence_fc_layer(
            tf.concat([outputs_lstm, cwt_skip], axis=-1),
            params[pkeys.FC_UNITS],
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            activation=tf.nn.relu,
            name="fc_1_lstm",
        )
        outputs_conv = layers.sequence_fc_layer(
            tf.concat([outputs_conv, cwt_skip], axis=-1),
            params[pkeys.FC_UNITS],
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            activation=tf.nn.relu,
            name="fc_1_conv",
        )

        # Skip connection
        outputs = tf.concat([outputs_lstm, outputs_conv], axis=-1)

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)

        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v28(inputs, params, training, name="model_v28"):
    """Experimental

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v28') A name for the network.
    """
    print("Using model V28 (experimental)")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        # CWt skip
        outputs_cwt, cwt_prebn = layers.cmorlet_layer_general(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            return_real_part=False,
            return_imag_part=False,
            return_magnitude=True,
            return_phase=False,
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        cwt_skip = tf.layers.average_pooling2d(
            inputs=outputs_cwt, pool_size=4, strides=4
        )
        cwt_skip = layers.sequence_flatten(cwt_skip, "cwt_skip_flatten")

        outputs_cwt = tf.layers.average_pooling2d(
            inputs=outputs_cwt, pool_size=(4, 1), strides=(4, 1)
        )
        outputs_cwt = layers.sequence_flatten(outputs_cwt, "outputs_cwt_flatten")

        # Convolutional stage (standard feed-forward)
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs_conv = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # Multilayer BLSTM (2 layers)
        # Skip connection at the input of lstm (direct access to magnitude)
        outputs_lstm = layers.multilayer_lstm_block(
            tf.concat([outputs_conv, cwt_skip], axis=-1),
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        # Three branches
        outputs_lstm = layers.sequence_fc_layer(
            tf.concat([outputs_lstm, cwt_skip], axis=-1),
            params[pkeys.FC_UNITS],
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            activation=tf.nn.relu,
            name="fc_1_lstm",
        )
        outputs_conv = layers.sequence_fc_layer(
            tf.concat([outputs_conv, cwt_skip], axis=-1),
            params[pkeys.FC_UNITS],
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            activation=tf.nn.relu,
            name="fc_1_conv",
        )
        outputs_cwt = layers.sequence_fc_layer(
            outputs_cwt,
            params[pkeys.FC_UNITS],
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            activation=tf.nn.relu,
            name="fc_1_cwt",
        )

        # Fusion
        outputs = tf.concat([outputs_lstm, outputs_conv, outputs_cwt], axis=-1)

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)

        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v29(inputs, params, training, name="model_v29"):
    """Experimental

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v29') A name for the network.
    """
    print("Using model V29 (experimental)")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        # CWt skip
        outputs_cwt, cwt_prebn = layers.cmorlet_layer_general(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            return_real_part=False,
            return_imag_part=False,
            return_magnitude=True,
            return_phase=False,
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        cwt_skip = tf.layers.average_pooling2d(
            inputs=outputs_cwt, pool_size=4, strides=4
        )
        cwt_skip = layers.sequence_flatten(cwt_skip, "cwt_skip_flatten")

        outputs_cwt = tf.layers.average_pooling2d(
            inputs=outputs_cwt, pool_size=(4, 1), strides=(4, 1)
        )
        outputs_cwt = layers.sequence_flatten(outputs_cwt, "outputs_cwt_flatten")

        # Convolutional stage (standard feed-forward)
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs_conv = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # Multilayer BLSTM (2 layers)
        # Skip connection at the input of lstm (direct access to magnitude)
        outputs_lstm = layers.multilayer_lstm_block(
            tf.concat([outputs_conv, cwt_skip], axis=-1),
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        # Three branches
        outputs_lstm = layers.sequence_fc_layer(
            tf.concat([outputs_lstm, cwt_skip], axis=-1),
            params[pkeys.FC_UNITS],
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            activation=tf.nn.relu,
            name="fc_1_lstm",
        )
        outputs_conv = layers.sequence_fc_layer(
            tf.concat([outputs_conv, cwt_skip], axis=-1),
            params[pkeys.FC_UNITS],
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            activation=tf.nn.relu,
            name="fc_1_conv",
        )
        outputs_cwt = layers.sequence_fc_layer(
            outputs_cwt,
            params[pkeys.FC_UNITS],
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            activation=tf.nn.relu,
            name="fc_1_cwt",
        )

        # Fusion
        outputs = outputs_lstm + outputs_conv + outputs_cwt

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)

        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_30(inputs, params, training, name="model_v30"):
    """Experimental

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v30') A name for the network.
    """
    print("Using model V30 (experimental)")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        # CWt skip
        outputs_cwt, cwt_prebn = layers.cmorlet_layer_general(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            return_real_part=False,
            return_imag_part=False,
            return_magnitude=True,
            return_phase=False,
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        cwt_skip = tf.layers.average_pooling2d(
            inputs=outputs_cwt, pool_size=4, strides=4
        )
        cwt_skip = layers.sequence_flatten(cwt_skip, "cwt_skip_flatten")

        outputs_cwt = tf.layers.average_pooling2d(
            inputs=outputs_cwt, pool_size=(4, 1), strides=(4, 1)
        )
        outputs_cwt = layers.sequence_flatten(outputs_cwt, "outputs_cwt_flatten")

        # Convolutional stage (standard feed-forward)
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs_conv = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # Multilayer BLSTM (2 layers)
        # Skip connection at the input of lstm (direct access to magnitude)
        outputs_lstm = layers.multilayer_lstm_block(
            tf.concat([outputs_conv, cwt_skip], axis=-1),
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        # Three branches
        outputs_fusion = tf.concat([outputs_lstm, outputs_conv, outputs_cwt], axis=-1)

        outputs = layers.sequence_fc_layer(
            outputs_fusion,
            params[pkeys.FC_UNITS],
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            activation=tf.nn.relu,
            name="fc_1",
        )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)

        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v115(inputs, params, training, name="model_v115"):
    """conv 1D and BLSTM to make a prediction.

    This models has a standard convolutional stage on time-domain
    (pre-activation BN). After this, the outputs is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v115') A name for the network.
    """
    print("Using model V115 (Time-Domain + Kernel 5)")
    with tf.variable_scope(name):

        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=5,
            kernel_size_2=5,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            kernel_size_1=5,
            kernel_size_2=5,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            kernel_size_1=5,
            kernel_size_2=5,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v195(inputs, params, training, name="model_v195"):
    """Wavelet transform, conv, and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a standard convolutional stage
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v195') A name for the network.
    """
    print("Using model V195 (general cwt + kernel 5)")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer_general(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            return_real_part=params[pkeys.CWT_RETURN_REAL_PART],
            return_imag_part=params[pkeys.CWT_RETURN_IMAG_PART],
            return_magnitude=params[pkeys.CWT_RETURN_MAGNITUDE],
            return_phase=params[pkeys.CWT_RETURN_PHASE],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_1],
            training,
            kernel_size_1=5,
            kernel_size_2=5,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1",
        )

        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_2],
            training,
            kernel_size_1=5,
            kernel_size_2=5,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_2",
        )

        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, "flatten")

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)

        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v11g(inputs, params, training, name="model_v11g"):
    """conv 1D and BiGRU to make a prediction.

    This models has a standard convolutional stage on time-domain
    (pre-activation BN). After this, the outputs is passed to
    a 2-layers BiGRU.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v11g') A name for the network.
    """
    print("Using model V11g (Time-Domain + GRU)")
    with tf.variable_scope(name):

        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # Multilayer BiGRU (2 layers)
        outputs = layers.multilayer_gru_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_gru=params[pkeys.TYPE_DROPOUT],
            dropout_rest_gru=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_gru=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_gru=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_bigru",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v19g(inputs, params, training, name="model_v19g"):
    """Wavelet transform, conv, and BiGRU to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a standard convolutional stage
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers BiGRU.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v19g') A name for the network.
    """
    print("Using model V19g (general cwt + GRU)")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer_general(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            return_real_part=params[pkeys.CWT_RETURN_REAL_PART],
            return_imag_part=params[pkeys.CWT_RETURN_IMAG_PART],
            return_magnitude=params[pkeys.CWT_RETURN_MAGNITUDE],
            return_phase=params[pkeys.CWT_RETURN_PHASE],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1",
        )

        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_2",
        )

        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, "flatten")

        # Multilayer BiGRU (2 layers)
        outputs = layers.multilayer_gru_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_gru=params[pkeys.TYPE_DROPOUT],
            dropout_rest_gru=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_gru=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_gru=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_bigru",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)

        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v31(inputs, params, training, name="model_v31"):
    """Wavelet transform, conv, and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a standard convolutional stage
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v31') A name for the network.
    """
    print("Using model V31 (general cwt with indep branches, 2 convs)")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer_general(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            return_real_part=params[pkeys.CWT_RETURN_REAL_PART],
            return_imag_part=params[pkeys.CWT_RETURN_IMAG_PART],
            return_magnitude=params[pkeys.CWT_RETURN_MAGNITUDE],
            return_phase=params[pkeys.CWT_RETURN_PHASE],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=1,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        # Pooling
        n_bands = 8
        pool_size = params[pkeys.N_SCALES] // n_bands
        # Output sequence has shape [batch_size, time_len, n_scales, channels]
        outputs = tf.layers.average_pooling2d(
            inputs=outputs, pool_size=(2, pool_size), strides=(2, pool_size)
        )

        # Unstack bands
        outputs_unstack = tf.unstack(outputs, axis=2)
        # Now we have shape [batch size, time len, channels]

        outputs_after_conv = []
        for i, single_output in enumerate(outputs_unstack):
            with tf.variable_scope("band_%d" % i):
                # Convolutional stage (standard feed-forward)
                single_output = layers.conv1d_prebn_block(
                    single_output,
                    params[pkeys.CWT_CONV_FILTERS_1],
                    training,
                    kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
                    batchnorm=params[pkeys.TYPE_BATCHNORM],
                    downsampling=params[pkeys.CONV_DOWNSAMPLING],
                    kernel_init=tf.initializers.he_normal(),
                    name="convblock_1",
                )
                single_output = layers.conv1d_prebn_block(
                    single_output,
                    params[pkeys.CWT_CONV_FILTERS_2],
                    training,
                    batchnorm=params[pkeys.TYPE_BATCHNORM],
                    downsampling=params[pkeys.CONV_DOWNSAMPLING],
                    kernel_init=tf.initializers.he_normal(),
                    name="convblock_2",
                )
            outputs_after_conv.append(single_output)

        # Concatenate all paths
        outputs = tf.concat(outputs_after_conv, axis=-1)
        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, "flatten")

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v32(inputs, params, training, name="model_v32"):
    """Wavelet transform, conv, and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a standard convolutional stage
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v32') A name for the network.
    """
    print("Using model V32 (general cwt with indep branches, 3 convs)")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer_general(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            return_real_part=params[pkeys.CWT_RETURN_REAL_PART],
            return_imag_part=params[pkeys.CWT_RETURN_IMAG_PART],
            return_magnitude=params[pkeys.CWT_RETURN_MAGNITUDE],
            return_phase=params[pkeys.CWT_RETURN_PHASE],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=1,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        # Pooling
        n_bands = 8
        pool_size = params[pkeys.N_SCALES] // n_bands
        # Output sequence has shape [batch_size, time_len, n_scales, channels]
        outputs = tf.layers.average_pooling2d(
            inputs=outputs, pool_size=(1, pool_size), strides=(1, pool_size)
        )

        # Unstack bands
        outputs_unstack = tf.unstack(outputs, axis=2)
        # Now we have shape [batch size, time len, channels]

        outputs_after_conv = []
        for i, single_output in enumerate(outputs_unstack):
            with tf.variable_scope("band_%d" % i):
                # Convolutional stage (standard feed-forward)
                single_output = layers.conv1d_prebn_block(
                    single_output,
                    params[pkeys.CWT_CONV_FILTERS_1],
                    training,
                    kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
                    batchnorm=params[pkeys.TYPE_BATCHNORM],
                    downsampling=params[pkeys.CONV_DOWNSAMPLING],
                    kernel_init=tf.initializers.he_normal(),
                    name="convblock_1",
                )
                single_output = layers.conv1d_prebn_block(
                    single_output,
                    params[pkeys.CWT_CONV_FILTERS_2],
                    training,
                    batchnorm=params[pkeys.TYPE_BATCHNORM],
                    downsampling=params[pkeys.CONV_DOWNSAMPLING],
                    kernel_init=tf.initializers.he_normal(),
                    name="convblock_2",
                )
                single_output = layers.conv1d_prebn_block(
                    single_output,
                    params[pkeys.CWT_CONV_FILTERS_3],
                    training,
                    batchnorm=params[pkeys.TYPE_BATCHNORM],
                    downsampling=params[pkeys.CONV_DOWNSAMPLING],
                    kernel_init=tf.initializers.he_normal(),
                    name="convblock_3",
                )
            outputs_after_conv.append(single_output)

        # Concatenate all paths
        outputs = tf.concat(outputs_after_conv, axis=-1)
        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, "flatten")

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)

        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v19p(inputs, params, training, name="model_v19p"):
    """Wavelet transform, conv, and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a standard convolutional stage
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v19p') A name for the network.
    """
    print("Using model V19p (general cwt + conv1x1 before lstm)")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer_general(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            return_real_part=params[pkeys.CWT_RETURN_REAL_PART],
            return_imag_part=params[pkeys.CWT_RETURN_IMAG_PART],
            return_magnitude=params[pkeys.CWT_RETURN_MAGNITUDE],
            return_phase=params[pkeys.CWT_RETURN_PHASE],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1",
        )

        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_2",
        )

        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, "flatten")

        # conv 1x1
        n_filters = 256
        with tf.variable_scope("conv1x1_neck"):
            outputs = tf.expand_dims(outputs, axis=2)
            outputs = tf.layers.conv2d(
                inputs=outputs,
                filters=n_filters,
                kernel_size=(1, 1),
                padding=constants.PAD_SAME,
                strides=1,
                name="conv1",
                kernel_initializer=tf.initializers.he_normal(),
                use_bias=False,
            )
            outputs = layers.batchnorm_layer(
                outputs,
                "bn",
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                training=training,
                scale=False,
            )
            outputs = tf.nn.relu(outputs)
            outputs = tf.squeeze(outputs, axis=2, name="squeeze")

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)

        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v33(inputs, params, training, name="model_v33"):
    """Wavelet transform, conv, and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a standard convolutional stage
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v33') A name for the network.
    """
    print("Using model V33 (general cwt with many LSTM)")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer_general(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            return_real_part=params[pkeys.CWT_RETURN_REAL_PART],
            return_imag_part=params[pkeys.CWT_RETURN_IMAG_PART],
            return_magnitude=params[pkeys.CWT_RETURN_MAGNITUDE],
            return_phase=params[pkeys.CWT_RETURN_PHASE],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1",
        )

        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_2",
        )

        # Unstack bands
        outputs_unstack = tf.unstack(outputs, axis=2)
        # Now we have shape [batch size, time len, channels]

        outputs_after_first_lstm = []
        n_units_first_lstm = int(
            params[pkeys.INITIAL_LSTM_UNITS] / (params[pkeys.N_SCALES] / 4)
        )
        for i, single_output in enumerate(outputs_unstack):
            #  First layer LSTM
            single_output = layers.lstm_layer(
                single_output,
                num_units=n_units_first_lstm,
                num_dirs=constants.BIDIRECTIONAL,
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_BEFORE_LSTM],
                training=training,
                name="lstm_1_b%d" % (i + 1),
            )
            outputs_after_first_lstm.append(single_output)

        # Concatenate all paths
        outputs = tf.concat(outputs_after_first_lstm, axis=-1)

        # Second layer LSTM
        outputs = layers.lstm_layer(
            outputs,
            num_units=params[pkeys.INITIAL_LSTM_UNITS],
            num_dirs=constants.BIDIRECTIONAL,
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="lstm_2",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)

        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v34(inputs, params, training, name="model_v34"):
    """Wavelet transform, conv, and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a standard convolutional stage
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v34') A name for the network.
    """
    print("Using model V34 (general cwt)")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer_general(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            return_real_part=params[pkeys.CWT_RETURN_REAL_PART],
            return_imag_part=params[pkeys.CWT_RETURN_IMAG_PART],
            return_magnitude=params[pkeys.CWT_RETURN_MAGNITUDE],
            return_phase=params[pkeys.CWT_RETURN_PHASE],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=1,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, "flatten")

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)

        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_att01(inputs, params, training, name="model_att01"):
    print("Using model ATT01 (Time-Domain)")
    with tf.variable_scope(name):

        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # --------------------------
        # Attention layer
        # input shape: [batch, time_len, n_feats]
        # --------------------------
        original_length = params[pkeys.PAGE_DURATION] * params[pkeys.FS]
        seq_len = int(original_length / params[pkeys.TOTAL_DOWNSAMPLING_FACTOR])

        with tf.variable_scope("attention"):

            # Prepare input
            pos_enc = layers.get_positional_encoding(
                seq_len=seq_len,
                dims=params[pkeys.ATT_DIM],
                pe_factor=params[pkeys.ATT_PE_FACTOR],
                name="pos_enc",
            )
            pos_enc = tf.expand_dims(pos_enc, axis=0)  # Add batch axis
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.ATT_DIM],
                kernel_init=tf.initializers.he_normal(),
                training=training,
                name="fc_embed",
            )

            outputs = outputs + pos_enc
            outputs = layers.dropout_layer(
                outputs,
                "drop_embed",
                drop_rate=params[pkeys.ATT_DROP_RATE],
                dropout=params[pkeys.TYPE_DROPOUT],
                training=training,
            )

            # Prepare queries, keys, and values
            queries = layers.sequence_fc_layer(
                outputs,
                params[pkeys.ATT_DIM],
                kernel_init=tf.initializers.he_normal(),
                training=training,
                name="queries",
            )
            keys = layers.sequence_fc_layer(
                outputs,
                params[pkeys.ATT_DIM],
                kernel_init=tf.initializers.he_normal(),
                training=training,
                name="keys",
            )
            values = layers.sequence_fc_layer(
                outputs,
                params[pkeys.ATT_DIM],
                kernel_init=tf.initializers.he_normal(),
                training=training,
                name="values",
            )

            outputs = layers.naive_multihead_attention_layer(
                queries, keys, values, params[pkeys.ATT_N_HEADS], name="multi_head_att"
            )

            # FFN
            outputs = layers.sequence_fc_layer(
                outputs,
                4 * params[pkeys.ATT_DIM],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.ATT_DROP_RATE],
                training=training,
                activation=tf.nn.relu,
                name="ffn_1",
            )

            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.ATT_DIM],
                kernel_init=tf.initializers.he_normal(),
                training=training,
                activation=tf.nn.relu,
                name="ffn_2",
            )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_att02(inputs, params, training, name="model_att02"):
    print("Using model ATT02 (Time-Domain)")
    with tf.variable_scope(name):

        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # --------------------------
        # Attention layer
        # input shape: [batch, time_len, n_feats]
        # --------------------------
        original_length = params[pkeys.PAGE_DURATION] * params[pkeys.FS]
        seq_len = int(original_length / params[pkeys.TOTAL_DOWNSAMPLING_FACTOR])

        with tf.variable_scope("attention"):

            # Prepare input
            pos_enc = layers.get_positional_encoding(
                seq_len=seq_len,
                dims=params[pkeys.ATT_DIM],
                pe_factor=params[pkeys.ATT_PE_FACTOR],
                name="pos_enc",
            )
            pos_enc = tf.expand_dims(pos_enc, axis=0)  # Add batch axis
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.ATT_DIM],
                kernel_init=tf.initializers.he_normal(),
                training=training,
                name="fc_embed",
            )

            outputs = outputs + pos_enc
            outputs = layers.dropout_layer(
                outputs,
                "drop_embed",
                drop_rate=params[pkeys.ATT_DROP_RATE],
                dropout=params[pkeys.TYPE_DROPOUT],
                training=training,
            )

            n_heads = params[pkeys.ATT_N_HEADS]
            dim_per_head = int(params[pkeys.ATT_DIM] / n_heads)
            dim_per_lstm = int(dim_per_head / 2)
            with tf.variable_scope("multi_head_att"):
                outputs_head = []
                for idx_head in range(n_heads):
                    with tf.variable_scope("head_att_%d" % idx_head):
                        # scores have shape [batch, q_time_len, k_time_len]
                        # outputs have shape [batch, q_time_len, v_dims]
                        # Prepare queries, keys, and values
                        head_q = layers.lstm_layer(
                            outputs,
                            num_units=dim_per_lstm,
                            num_dirs=constants.BIDIRECTIONAL,
                            training=training,
                            name="q_lstm_%d" % idx_head,
                        )
                        head_k = layers.lstm_layer(
                            outputs,
                            num_units=dim_per_lstm,
                            num_dirs=constants.BIDIRECTIONAL,
                            training=training,
                            name="k_lstm_%d" % idx_head,
                        )
                        # Values must be isolated
                        head_v = layers.sequence_fc_layer(
                            outputs,
                            dim_per_head,
                            kernel_init=tf.initializers.he_normal(),
                            training=training,
                            name="v_%d" % idx_head,
                        )

                        head_o, _ = layers.attention_layer(
                            head_q, head_k, head_v, name="head_%d" % idx_head
                        )
                        outputs_head.append(head_o)

                # Concatenate heads
                outputs = tf.concat(outputs_head, axis=-1)

            # FFN
            outputs = layers.sequence_fc_layer(
                outputs,
                4 * params[pkeys.ATT_DIM],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.ATT_DROP_RATE],
                training=training,
                activation=tf.nn.relu,
                name="ffn_1",
            )

            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.ATT_DIM],
                kernel_init=tf.initializers.he_normal(),
                training=training,
                activation=tf.nn.relu,
                name="ffn_2",
            )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_att03(inputs, params, training, name="model_att03"):
    print("Using model ATT03 (Time-Domain)")
    with tf.variable_scope(name):

        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # --------------------------
        # Attention layer
        # input shape: [batch, time_len, n_feats]
        # --------------------------
        original_length = params[pkeys.PAGE_DURATION] * params[pkeys.FS]
        seq_len = int(original_length / params[pkeys.TOTAL_DOWNSAMPLING_FACTOR])
        att_dim = params[pkeys.ATT_DIM]
        n_heads = params[pkeys.ATT_N_HEADS]

        with tf.variable_scope("attention"):

            after_lstm_outputs = layers.lstm_layer(
                outputs,
                num_units=params[pkeys.ATT_LSTM_DIM],
                num_dirs=constants.BIDIRECTIONAL,
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_BEFORE_LSTM],
                training=training,
                name="blstm",
            )

            # Prepare input for values
            pos_enc = layers.get_positional_encoding(
                seq_len=seq_len,
                dims=att_dim,
                pe_factor=params[pkeys.ATT_PE_FACTOR],
                name="pos_enc",
            )
            pos_enc = tf.expand_dims(pos_enc, axis=0)  # Add batch axis

            v_outputs = layers.sequence_fc_layer(
                outputs,
                att_dim,
                kernel_init=tf.initializers.he_normal(),
                training=training,
                name="fc_embed_v",
            )
            v_outputs = v_outputs + pos_enc
            v_outputs = layers.dropout_layer(
                v_outputs,
                "drop_embed_v",
                drop_rate=params[pkeys.ATT_DROP_RATE],
                dropout=params[pkeys.TYPE_DROPOUT],
                training=training,
            )

            # Prepare input for queries and keys
            qk_outputs = layers.sequence_fc_layer(
                after_lstm_outputs,
                att_dim,
                kernel_init=tf.initializers.he_normal(),
                training=training,
                name="fc_embed_qk",
            )
            qk_outputs = qk_outputs + pos_enc
            qk_outputs = layers.dropout_layer(
                qk_outputs,
                "drop_embed_qk",
                drop_rate=params[pkeys.ATT_DROP_RATE],
                dropout=params[pkeys.TYPE_DROPOUT],
                training=training,
            )

            # Prepare queries, keys, and values
            queries = layers.sequence_fc_layer(
                qk_outputs,
                att_dim,
                kernel_init=tf.initializers.he_normal(),
                training=training,
                name="queries",
            )
            keys = layers.sequence_fc_layer(
                qk_outputs,
                att_dim,
                kernel_init=tf.initializers.he_normal(),
                training=training,
                name="keys",
            )
            values = layers.sequence_fc_layer(
                v_outputs,
                att_dim,
                kernel_init=tf.initializers.he_normal(),
                training=training,
                name="values",
            )

            outputs = layers.naive_multihead_attention_layer(
                queries, keys, values, n_heads, name="multi_head_att"
            )

            # FFN
            outputs = layers.sequence_fc_layer(
                outputs,
                4 * params[pkeys.ATT_DIM],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.ATT_DROP_RATE],
                training=training,
                activation=tf.nn.relu,
                name="ffn_1",
            )

            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.ATT_DIM],
                kernel_init=tf.initializers.he_normal(),
                training=training,
                activation=tf.nn.relu,
                name="ffn_2",
            )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_att04(inputs, params, training, name="model_att04"):
    print("Using model ATT04 (Time-Domain)")
    with tf.variable_scope(name):

        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # --------------------------
        # Attention layer
        # input shape: [batch, time_len, n_feats]
        # --------------------------
        original_length = params[pkeys.PAGE_DURATION] * params[pkeys.FS]
        seq_len = int(original_length / params[pkeys.TOTAL_DOWNSAMPLING_FACTOR])
        att_dim = params[pkeys.ATT_DIM]
        n_heads = params[pkeys.ATT_N_HEADS]

        with tf.variable_scope("attention"):

            # Multilayer BLSTM (2 layers)
            after_lstm_outputs = layers.multilayer_lstm_block(
                outputs,
                params[pkeys.ATT_LSTM_DIM],
                n_layers=2,
                num_dirs=constants.BIDIRECTIONAL,
                dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
                dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
                drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
                drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                name="multi_layer_blstm",
            )

            # Prepare input for values
            pos_enc = layers.get_positional_encoding(
                seq_len=seq_len,
                dims=att_dim,
                pe_factor=params[pkeys.ATT_PE_FACTOR],
                name="pos_enc",
            )
            pos_enc = tf.expand_dims(pos_enc, axis=0)  # Add batch axis

            v_outputs = layers.sequence_fc_layer(
                outputs,
                att_dim,
                kernel_init=tf.initializers.he_normal(),
                training=training,
                name="fc_embed_v",
            )
            v_outputs = v_outputs + pos_enc
            v_outputs = layers.dropout_layer(
                v_outputs,
                "drop_embed_v",
                drop_rate=params[pkeys.ATT_DROP_RATE],
                dropout=params[pkeys.TYPE_DROPOUT],
                training=training,
            )

            # Prepare input for queries and keys
            qk_outputs = layers.sequence_fc_layer(
                after_lstm_outputs,
                att_dim,
                kernel_init=tf.initializers.he_normal(),
                training=training,
                name="fc_embed_qk",
            )
            qk_outputs = qk_outputs + pos_enc
            qk_outputs = layers.dropout_layer(
                qk_outputs,
                "drop_embed_qk",
                drop_rate=params[pkeys.ATT_DROP_RATE],
                dropout=params[pkeys.TYPE_DROPOUT],
                training=training,
            )

            # Prepare queries, keys, and values
            queries = layers.sequence_fc_layer(
                qk_outputs,
                att_dim,
                kernel_init=tf.initializers.he_normal(),
                training=training,
                name="queries",
            )
            keys = layers.sequence_fc_layer(
                qk_outputs,
                att_dim,
                kernel_init=tf.initializers.he_normal(),
                training=training,
                name="keys",
            )
            values = layers.sequence_fc_layer(
                v_outputs,
                att_dim,
                kernel_init=tf.initializers.he_normal(),
                training=training,
                name="values",
            )

            outputs = layers.naive_multihead_attention_layer(
                queries, keys, values, n_heads, name="multi_head_att"
            )

            # FFN
            outputs = layers.sequence_fc_layer(
                outputs,
                4 * params[pkeys.ATT_DIM],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.ATT_DROP_RATE],
                training=training,
                activation=tf.nn.relu,
                name="ffn_1",
            )

            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.ATT_DIM],
                kernel_init=tf.initializers.he_normal(),
                training=training,
                activation=tf.nn.relu,
                name="ffn_2",
            )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_att04c(inputs, params, training, name="model_att04c"):
    print("Using model ATT04C (Time-Domain)")
    with tf.variable_scope(name):

        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # --------------------------
        # Attention layer
        # input shape: [batch, time_len, n_feats]
        # --------------------------
        original_length = params[pkeys.PAGE_DURATION] * params[pkeys.FS]
        seq_len = int(original_length / params[pkeys.TOTAL_DOWNSAMPLING_FACTOR])
        att_dim = params[pkeys.ATT_DIM]
        n_heads = params[pkeys.ATT_N_HEADS]
        att_pe_dim = params[pkeys.ATT_PE_CONCAT_DIM]

        with tf.variable_scope("attention"):

            # Multilayer BLSTM (2 layers)
            after_lstm_outputs = layers.multilayer_lstm_block(
                outputs,
                params[pkeys.ATT_LSTM_DIM],
                n_layers=2,
                num_dirs=constants.BIDIRECTIONAL,
                dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
                dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
                drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
                drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                name="multi_layer_blstm",
            )

            # Prepare input for values
            pos_enc = layers.get_positional_encoding(
                seq_len=seq_len,
                dims=att_pe_dim,
                pe_factor=params[pkeys.ATT_PE_FACTOR],
                name="pos_enc",
            )
            pos_enc = tf.expand_dims(pos_enc, axis=0)  # Add batch axis
            # Get the number of rows in the fed value at run-time.
            batch_size = tf.shape(outputs)[0]
            pos_enc = tf.tile(pos_enc, tf.stack([batch_size, 1, 1]))

            v_outputs = tf.concat([outputs, pos_enc], axis=-1)
            v_outputs = layers.dropout_layer(
                v_outputs,
                "drop_embed_v",
                drop_rate=params[pkeys.ATT_DROP_RATE],
                dropout=params[pkeys.TYPE_DROPOUT],
                training=training,
            )

            # Prepare input for queries and keys
            qk_outputs = tf.concat([after_lstm_outputs, pos_enc], axis=-1)
            qk_outputs = layers.dropout_layer(
                qk_outputs,
                "drop_embed_qk",
                drop_rate=params[pkeys.ATT_DROP_RATE],
                dropout=params[pkeys.TYPE_DROPOUT],
                training=training,
            )

            # Prepare queries, keys, and values
            queries = layers.sequence_fc_layer(
                qk_outputs,
                att_dim,
                kernel_init=tf.initializers.he_normal(),
                training=training,
                name="queries",
            )
            keys = layers.sequence_fc_layer(
                qk_outputs,
                att_dim,
                kernel_init=tf.initializers.he_normal(),
                training=training,
                name="keys",
            )
            values = layers.sequence_fc_layer(
                v_outputs,
                att_dim,
                kernel_init=tf.initializers.he_normal(),
                training=training,
                name="values",
            )

            outputs = layers.naive_multihead_attention_layer(
                queries, keys, values, n_heads, name="multi_head_att"
            )

            # FFN
            outputs = layers.sequence_fc_layer(
                outputs,
                4 * params[pkeys.ATT_DIM],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.ATT_DROP_RATE],
                training=training,
                activation=tf.nn.relu,
                name="ffn_1",
            )

            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.ATT_DIM],
                kernel_init=tf.initializers.he_normal(),
                training=training,
                activation=tf.nn.relu,
                name="ffn_2",
            )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v35(inputs, params, training, name="model_v35"):
    """Wavelet transform, conv, and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a standard convolutional stage
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v19') A name for the network.
    """
    print("Using model V35 (general cwt + time at last FC)")
    border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
    start_crop = border_crop
    if border_crop <= 0:
        end_crop = None
    else:
        end_crop = -border_crop

    with tf.variable_scope(name):

        # ---------------
        # RED-Time branch
        # ---------------
        time_outputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        time_outputs = tf.expand_dims(time_outputs, axis=2)
        # BN at input
        time_outputs = layers.batchnorm_layer(
            time_outputs,
            "time_bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )
        # 1D convolutions expect shape [batch, time_len, n_feats]
        # Convolutional stage (standard feed-forward)
        time_outputs = layers.conv1d_prebn_block(
            time_outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="time_convblock_1",
        )
        time_outputs = layers.conv1d_prebn_block(
            time_outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="time_convblock_2",
        )
        time_outputs = layers.conv1d_prebn_block(
            time_outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="time_convblock_3",
        )
        # Multilayer BLSTM (2 layers)
        time_outputs = layers.multilayer_lstm_block(
            time_outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="time_multi_layer_blstm",
        )
        # Additional FC layer to increase model flexibility
        time_outputs = layers.sequence_fc_layer(
            time_outputs,
            params[pkeys.FC_UNITS],
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            activation=tf.nn.relu,
            name="time_fc_1",
        )

        # ---------------
        # RED-CWT Branch
        # --------------
        cwt_outputs, cwt_prebn = layers.cmorlet_layer_general(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            return_real_part=params[pkeys.CWT_RETURN_REAL_PART],
            return_imag_part=params[pkeys.CWT_RETURN_IMAG_PART],
            return_magnitude=params[pkeys.CWT_RETURN_MAGNITUDE],
            return_phase=params[pkeys.CWT_RETURN_PHASE],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )
        # Convolutional stage (standard feed-forward)
        cwt_outputs = layers.conv2d_prebn_block(
            cwt_outputs,
            params[pkeys.CWT_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="cwt_convblock_1",
        )
        cwt_outputs = layers.conv2d_prebn_block(
            cwt_outputs,
            params[pkeys.CWT_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="cwt_convblock_2",
        )
        # Flattening for dense part
        cwt_outputs = layers.sequence_flatten(cwt_outputs, "cwt_flatten")
        # Multilayer BLSTM (2 layers)
        cwt_outputs = layers.multilayer_lstm_block(
            cwt_outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="cwt_multi_layer_blstm",
        )
        # Additional FC layer to increase model flexibility
        cwt_outputs = layers.sequence_fc_layer(
            cwt_outputs,
            params[pkeys.FC_UNITS],
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            activation=tf.nn.relu,
            name="cwt_fc_1",
        )

        # ---------------
        # Mixing
        # --------------
        outputs = tf.concat([cwt_outputs, time_outputs], axis=-1)

        # Additional FC layer to increase model flexibility
        outputs = layers.sequence_fc_layer(
            outputs,
            params[pkeys.FC_UNITS],
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            activation=tf.nn.relu,
            name="fc_mix",
        )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)

        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v11_ablation(inputs, params, training, name="model_v11_ablation"):
    """conv 1D and BLSTM to make a prediction.

    This models has a standard convolutional stage on time-domain
    (pre-activation BN). After this, the outputs is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v11') A name for the network.
    """
    print("Using model V11 Ablation (Time-Domain)")
    with tf.variable_scope(name):

        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.ABLATION_TYPE_BATCHNORM_INPUT],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.ABLATION_TYPE_BATCHNORM_CONV],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.ABLATION_TYPE_BATCHNORM_CONV],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.ABLATION_TYPE_BATCHNORM_CONV],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.ABLATION_DROP_RATE],
            drop_rate_rest_lstm=params[pkeys.ABLATION_DROP_RATE],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.ABLATION_DROP_RATE],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v11_ablation_scaled(
    inputs, params, training, name="model_v11_ablation_scaled"
):
    """conv 1D and BLSTM to make a prediction.

    This models has a standard convolutional stage on time-domain
    (pre-activation BN). After this, the outputs is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v11') A name for the network.
    """
    print("Using model V11 Ablation Scaled (Time-Domain)")
    with tf.variable_scope(name):

        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # Scale inputs
        print("Scaling input by 1/10")
        inputs = inputs / 10.0

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.ABLATION_TYPE_BATCHNORM_INPUT],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.ABLATION_TYPE_BATCHNORM_CONV],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.ABLATION_TYPE_BATCHNORM_CONV],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.ABLATION_TYPE_BATCHNORM_CONV],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.ABLATION_DROP_RATE],
            drop_rate_rest_lstm=params[pkeys.ABLATION_DROP_RATE],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.ABLATION_DROP_RATE],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v11_d6k5(inputs, params, training, name="model_v11_d6k5"):
    """conv 1D and BLSTM to make a prediction.

    This models has a standard convolutional stage on time-domain
    (pre-activation BN). After this, the outputs is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v11') A name for the network.
    """
    print("Using model V11 D6-K5 (Time-Domain)")
    with tf.variable_scope(name):

        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=5,
            kernel_size_2=5,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            kernel_size_1=5,
            kernel_size_2=5,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            kernel_size_1=5,
            kernel_size_2=5,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v11_d8k5(inputs, params, training, name="model_v11_d8k5"):
    """conv 1D and BLSTM to make a prediction.

    This models has a standard convolutional stage on time-domain
    (pre-activation BN). After this, the outputs is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v11') A name for the network.
    """
    print("Using model V11 D8-K5 (Time-Domain)")
    with tf.variable_scope(name):

        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=5,
            kernel_size_2=5,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            kernel_size_1=5,
            kernel_size_2=5,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            kernel_size_1=5,
            kernel_size_2=5,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=None,
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3a",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            kernel_size_1=5,
            kernel_size_2=5,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3b",
        )

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v11_d8k3(inputs, params, training, name="model_v11_d8k3"):
    """conv 1D and BLSTM to make a prediction.

    This models has a standard convolutional stage on time-domain
    (pre-activation BN). After this, the outputs is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v11') A name for the network.
    """
    print("Using model V11 D8-K3 (Time-Domain)")
    with tf.variable_scope(name):

        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=3,
            kernel_size_2=3,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            kernel_size_1=3,
            kernel_size_2=3,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            kernel_size_1=3,
            kernel_size_2=3,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=None,
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3a",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            kernel_size_1=3,
            kernel_size_2=3,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3b",
        )

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v11_outres(inputs, params, training, name="model_v11_outres"):
    """conv 1D and BLSTM to make a prediction.

    This models has a standard convolutional stage on time-domain
    (pre-activation BN). After this, the outputs is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v11') A name for the network.
    """
    print("Using model V11 Output Residual (Time-Domain)")
    with tf.variable_scope(name):

        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=3,
            kernel_size_2=3,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            kernel_size_1=3,
            kernel_size_2=3,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            kernel_size_1=3,
            kernel_size_2=3,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        # output residual
        # [batch_size, time_len, n_feats] -> [batch_size, time_len, 1, feats]
        outputs = tf.expand_dims(outputs, axis=2)

        with tf.variable_scope("res_0"):
            outputs = tf.layers.conv2d(
                inputs=outputs,
                filters=params[pkeys.OUTPUT_RESIDUAL_FC_SIZE],
                kernel_size=1,
                activation=None,
                padding=constants.PAD_SAME,
                kernel_initializer=tf.initializers.he_normal(),
                use_bias=False,
                name="res_fc_0",
            )
            outputs = layers.batchnorm_layer(
                outputs,
                "res_bn_0",
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                training=training,
                scale=False,
            )
            outputs = tf.nn.relu(outputs)

        with tf.variable_scope("res_1"):
            shortcut = outputs
            outputs = tf.layers.conv2d(
                inputs=outputs,
                filters=params[pkeys.OUTPUT_RESIDUAL_FC_SIZE],
                kernel_size=1,
                activation=None,
                padding=constants.PAD_SAME,
                kernel_initializer=tf.initializers.he_normal(),
                use_bias=False,
                name="res_fc_1a",
            )
            outputs = layers.batchnorm_layer(
                outputs,
                "res_bn_1",
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                training=training,
                scale=False,
            )
            outputs = tf.nn.relu(outputs)
            outputs = tf.layers.conv2d(
                inputs=outputs,
                filters=params[pkeys.OUTPUT_RESIDUAL_FC_SIZE],
                kernel_size=1,
                activation=None,
                padding=constants.PAD_SAME,
                kernel_initializer=tf.initializers.he_normal(),
                use_bias=False,
                name="res_fc_1b",
            )
            outputs = outputs + shortcut

        with tf.variable_scope("res_2"):
            shortcut = outputs
            outputs = layers.batchnorm_layer(
                outputs,
                "res_bn_2a",
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                training=training,
                scale=False,
            )
            outputs = tf.nn.relu(outputs)
            outputs = tf.layers.conv2d(
                inputs=outputs,
                filters=params[pkeys.OUTPUT_RESIDUAL_FC_SIZE],
                kernel_size=1,
                activation=None,
                padding=constants.PAD_SAME,
                kernel_initializer=tf.initializers.he_normal(),
                use_bias=False,
                name="res_fc_2a",
            )
            outputs = layers.batchnorm_layer(
                outputs,
                "res_bn_2b",
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                training=training,
                scale=False,
            )
            outputs = tf.nn.relu(outputs)
            outputs = tf.layers.conv2d(
                inputs=outputs,
                filters=params[pkeys.OUTPUT_RESIDUAL_FC_SIZE],
                kernel_size=1,
                activation=None,
                padding=constants.PAD_SAME,
                kernel_initializer=tf.initializers.he_normal(),
                use_bias=False,
                name="res_fc_2b",
            )
            outputs = outputs + shortcut

        with tf.variable_scope("res_3"):
            shortcut = outputs
            outputs = layers.batchnorm_layer(
                outputs,
                "res_bn_3a",
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                training=training,
                scale=False,
            )
            outputs = tf.nn.relu(outputs)
            outputs = tf.layers.conv2d(
                inputs=outputs,
                filters=params[pkeys.OUTPUT_RESIDUAL_FC_SIZE],
                kernel_size=1,
                activation=None,
                padding=constants.PAD_SAME,
                kernel_initializer=tf.initializers.he_normal(),
                use_bias=False,
                name="res_fc_3a",
            )
            outputs = layers.batchnorm_layer(
                outputs,
                "res_bn_3b",
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                training=training,
                scale=False,
            )
            outputs = tf.nn.relu(outputs)
            outputs = tf.layers.conv2d(
                inputs=outputs,
                filters=params[pkeys.OUTPUT_RESIDUAL_FC_SIZE],
                kernel_size=1,
                activation=None,
                padding=constants.PAD_SAME,
                kernel_initializer=tf.initializers.he_normal(),
                use_bias=False,
                name="res_fc_3b",
            )
            outputs = outputs + shortcut

        with tf.variable_scope("res_end"):
            outputs = layers.batchnorm_layer(
                outputs,
                "res_bn_4",
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                training=training,
                scale=False,
            )
            outputs = tf.nn.relu(outputs)
            outputs = tf.squeeze(outputs, axis=2, name="res_squeeze")

        # Additional FC layer to increase model flexibility
        if params[pkeys.FC_UNITS] > 0:
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v11_outplus(inputs, params, training, name="model_v11_outplus"):
    """conv 1D and BLSTM to make a prediction.

    This models has a standard convolutional stage on time-domain
    (pre-activation BN). After this, the outputs is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v11') A name for the network.
    """
    print("Using model V11 Output Plus (Time-Domain)")
    with tf.variable_scope(name):

        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=3,
            kernel_size_2=3,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            kernel_size_1=3,
            kernel_size_2=3,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            kernel_size_1=3,
            kernel_size_2=3,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        bn_at_fc = params[pkeys.OUTPUT_USE_BN]
        drop_at_fc = params[pkeys.OUTPUT_USE_DROP]

        with tf.variable_scope("fc_1"):
            if drop_at_fc:
                outputs = layers.dropout_layer(
                    outputs,
                    "drop",
                    drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                    dropout=params[pkeys.TYPE_DROPOUT],
                    training=training,
                )
            outputs = tf.expand_dims(outputs, axis=2)
            use_bias = not bn_at_fc
            outputs = tf.layers.conv2d(
                inputs=outputs,
                filters=params[pkeys.FC_UNITS_1],
                kernel_size=1,
                activation=None,
                padding=constants.PAD_SAME,
                kernel_initializer=tf.initializers.he_normal(),
                use_bias=use_bias,
                name="conv1",
            )
            if bn_at_fc:
                outputs = layers.batchnorm_layer(
                    outputs,
                    "bn",
                    batchnorm=params[pkeys.TYPE_BATCHNORM],
                    training=training,
                    scale=False,
                )
            outputs = tf.nn.relu(outputs)
            outputs = tf.squeeze(outputs, axis=2, name="squeeze")

        with tf.variable_scope("fc_2"):
            if drop_at_fc:
                outputs = layers.dropout_layer(
                    outputs,
                    "drop",
                    drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                    dropout=params[pkeys.TYPE_DROPOUT],
                    training=training,
                )
            outputs = tf.expand_dims(outputs, axis=2)
            use_bias = not bn_at_fc
            outputs = tf.layers.conv2d(
                inputs=outputs,
                filters=params[pkeys.FC_UNITS_2],
                kernel_size=1,
                activation=None,
                padding=constants.PAD_SAME,
                kernel_initializer=tf.initializers.he_normal(),
                use_bias=use_bias,
                name="conv1",
            )
            if bn_at_fc:
                outputs = layers.batchnorm_layer(
                    outputs,
                    "bn",
                    batchnorm=params[pkeys.TYPE_BATCHNORM],
                    training=training,
                    scale=False,
                )
            outputs = tf.nn.relu(outputs)
            outputs = tf.squeeze(outputs, axis=2, name="squeeze")

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v11_shield(inputs, params, training, name="model_v11_shield"):
    """conv 1D and BLSTM to make a prediction.

    This models has a standard convolutional stage on time-domain
    (pre-activation BN). After this, the outputs is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v11') A name for the network.
    """
    print("Using model V11 LSTM Shielding (Time-Domain)")
    with tf.variable_scope(name):

        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=3,
            kernel_size_2=3,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            kernel_size_1=3,
            kernel_size_2=3,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            kernel_size_1=3,
            kernel_size_2=3,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        shortcut = outputs
        lstm_down_factor = params[pkeys.SHIELD_LSTM_DOWN_FACTOR]

        if lstm_down_factor > 1:
            outputs = layers.downsampling_1d(
                outputs,
                "shield_down",
                lstm_down_factor,
                params[pkeys.SHIELD_LSTM_TYPE_POOL],
            )

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if lstm_down_factor > 1:
            outputs = layers.upsampling_1d_linear(
                outputs, "shield_up", lstm_down_factor
            )

        outputs = tf.concat([outputs, shortcut], axis=2)

        with tf.variable_scope("fc_1"):
            outputs = layers.dropout_layer(
                outputs,
                "drop",
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                dropout=params[pkeys.TYPE_DROPOUT],
                training=training,
            )
            outputs = tf.expand_dims(outputs, axis=2)
            outputs = tf.layers.conv2d(
                inputs=outputs,
                filters=params[pkeys.FC_UNITS_1],
                kernel_size=1,
                activation=None,
                padding=constants.PAD_SAME,
                kernel_initializer=tf.initializers.he_normal(),
                use_bias=True,
                name="conv1",
            )
            outputs = tf.nn.relu(outputs)
            outputs = tf.squeeze(outputs, axis=2, name="squeeze")

        with tf.variable_scope("fc_2"):
            outputs = layers.dropout_layer(
                outputs,
                "drop",
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                dropout=params[pkeys.TYPE_DROPOUT],
                training=training,
            )
            outputs = tf.expand_dims(outputs, axis=2)
            outputs = tf.layers.conv2d(
                inputs=outputs,
                filters=params[pkeys.FC_UNITS_2],
                kernel_size=1,
                activation=None,
                padding=constants.PAD_SAME,
                kernel_initializer=tf.initializers.he_normal(),
                use_bias=True,
                name="conv1",
            )
            outputs = tf.nn.relu(outputs)
            outputs = tf.squeeze(outputs, axis=2, name="squeeze")

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v11_lite(inputs, params, training, name="model_v11_lite"):
    """conv 1D and BLSTM to make a prediction.

    This models has a standard convolutional stage on time-domain
    (pre-activation BN). After this, the outputs is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v11') A name for the network.
    """
    print("Using model V11 LITE (Time-Domain)")
    with tf.variable_scope(name):

        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1] // 2,
            training,
            kernel_size=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_0",
        )
        outputs = layers.conv1d_prebn(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=None,
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v11_norm(inputs, params, training, name="model_v11_norm"):
    """conv 1D and BLSTM to make a prediction.

    This models has a standard convolutional stage on time-domain
    (pre-activation BN). After this, the outputs is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v11') A name for the network.
    """
    print("Using model V11 NORM (Time-Domain)")
    with tf.variable_scope(name):

        # Normalize inputs
        with tf.variable_scope("normalization"):
            # Input shape [batch, time_len]
            inputs_mean = tf.reduce_mean(inputs, axis=1)
            inputs_mean = tf.expand_dims(inputs_mean, axis=1)
            inputs = inputs - inputs_mean  # zero-mean
            # for a zero-mean x, variance is just the mean of x^2
            inputs_variance = tf.reduce_mean(inputs**2, axis=1)
            inputs_std = tf.sqrt(inputs_variance + 1e-6)
            inputs_std = tf.expand_dims(inputs_std, axis=1)
            inputs = inputs / inputs_std

        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v11_pr_1(inputs, params, training, name="model_v11_pr_1"):
    """
    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string) A name for the network.
    """
    print("Using model V11 + Power Ratios Fixed from literature (v11_pr_1)")
    with tf.variable_scope(name):
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        # Compute power ratios
        power_ratios = layers.power_ratio_literature_fixed_layer(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            params[pkeys.LOWER_FREQ],
            params[pkeys.UPPER_FREQ],
            params[pkeys.N_SCALES],
            training,
            border_crop=border_crop,
            return_power_bands=params[pkeys.PR_RETURN_BANDS],
            return_power_ratios=params[pkeys.PR_RETURN_RATIOS],
            use_log=params[pkeys.USE_LOG],
        )

        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # Now we concatenate with power ratios at the input
        outputs = tf.concat([outputs, power_ratios], axis=2)

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v11_pr_2p(inputs, params, training, name="model_v11_pr_2p"):
    """
    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string) A name for the network.
    """
    print("Using model V11 + Power Ratios Fixed from literature (v11_pr_2p)")
    with tf.variable_scope(name):
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        # Compute power ratios
        power_ratios = layers.power_ratio_literature_fixed_layer(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            params[pkeys.LOWER_FREQ],
            params[pkeys.UPPER_FREQ],
            params[pkeys.N_SCALES],
            training,
            border_crop=border_crop,
            return_power_bands=params[pkeys.PR_RETURN_BANDS],
            return_power_ratios=params[pkeys.PR_RETURN_RATIOS],
            use_log=params[pkeys.USE_LOG],
        )
        power_ratios = tf.keras.layers.AveragePooling1D(
            pool_size=params[pkeys.TOTAL_DOWNSAMPLING_FACTOR]
        )(power_ratios)

        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # Now we concatenate with power ratios at output of CNN
        outputs = tf.concat([outputs, power_ratios], axis=2)

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v11_pr_3p(inputs, params, training, name="model_v11_pr_3p"):
    """
    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string) A name for the network.
    """
    print("Using model V11 + Power Ratios Fixed from literature (v11_pr_3p)")
    with tf.variable_scope(name):
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        # Compute power ratios
        power_ratios = layers.power_ratio_literature_fixed_layer(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            params[pkeys.LOWER_FREQ],
            params[pkeys.UPPER_FREQ],
            params[pkeys.N_SCALES],
            training,
            border_crop=border_crop,
            return_power_bands=params[pkeys.PR_RETURN_BANDS],
            return_power_ratios=params[pkeys.PR_RETURN_RATIOS],
            use_log=params[pkeys.USE_LOG],
        )
        power_ratios = tf.keras.layers.AveragePooling1D(
            pool_size=params[pkeys.TOTAL_DOWNSAMPLING_FACTOR]
        )(power_ratios)

        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        # Now we concatenate with power ratios at output of LSTM
        outputs = tf.concat([outputs, power_ratios], axis=2)

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v11_pr_2c(inputs, params, training, name="model_v11_pr_2c"):
    """
    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string) A name for the network.
    """
    print("Using model V11 + Power Ratios Fixed from literature (v11_pr_2c)")
    with tf.variable_scope(name):
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        # Compute power ratios
        power_ratios = layers.power_ratio_literature_fixed_layer(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            params[pkeys.LOWER_FREQ],
            params[pkeys.UPPER_FREQ],
            params[pkeys.N_SCALES],
            training,
            border_crop=border_crop,
            return_power_bands=params[pkeys.PR_RETURN_BANDS],
            return_power_ratios=params[pkeys.PR_RETURN_RATIOS],
            use_log=params[pkeys.USE_LOG],
        )
        # Convolutional stage (standard feed-forward)
        power_ratios = layers.conv1d_prebn_block(
            power_ratios,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="pr_convblock_1d_1",
        )
        power_ratios = layers.conv1d_prebn_block(
            power_ratios,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="pr_convblock_1d_2",
        )
        power_ratios = layers.conv1d_prebn_block(
            power_ratios,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="pr_convblock_1d_3",
        )

        # -----------------------------------------------------------

        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # Now we concatenate with power ratios at output of CNN
        outputs = tf.concat([outputs, power_ratios], axis=2)

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v11_pr_3c(inputs, params, training, name="model_v11_pr_3c"):
    """
    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string) A name for the network.
    """
    print("Using model V11 + Power Ratios Fixed from literature (v11_pr_3c)")
    with tf.variable_scope(name):
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        # Compute power ratios
        power_ratios = layers.power_ratio_literature_fixed_layer(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            params[pkeys.LOWER_FREQ],
            params[pkeys.UPPER_FREQ],
            params[pkeys.N_SCALES],
            training,
            border_crop=border_crop,
            return_power_bands=params[pkeys.PR_RETURN_BANDS],
            return_power_ratios=params[pkeys.PR_RETURN_RATIOS],
            use_log=params[pkeys.USE_LOG],
        )
        # Convolutional stage (standard feed-forward)
        power_ratios = layers.conv1d_prebn_block(
            power_ratios,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="pr_convblock_1d_1",
        )
        power_ratios = layers.conv1d_prebn_block(
            power_ratios,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="pr_convblock_1d_2",
        )
        power_ratios = layers.conv1d_prebn_block(
            power_ratios,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="pr_convblock_1d_3",
        )

        # -----------------------------------------------------------

        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        # Now we concatenate with power ratios at output of LSTM
        outputs = tf.concat([outputs, power_ratios], axis=2)

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v11_llc_stft(inputs, params, training, name="model_v11_llc_stft"):
    """
    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string) A name for the network.
    """
    print("Using model V11 + LLC-STFT")
    with tf.variable_scope(name):
        # We assume the border is very big
        inputs = tf.expand_dims(inputs, axis=2)
        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        with tf.variable_scope("stft_module"):
            window_length = params[pkeys.LLC_STFT_N_SAMPLES]
            outputs_llc = tf.signal.stft(
                outputs[:, :, 0],
                frame_length=window_length,
                frame_step=window_length // 2,
                name="stft",
            )
            norm_factor = 2 / np.sum(np.hanning(window_length))
            outputs_llc = tf.abs(outputs_llc) * norm_factor  # complex to real
            frequency_axis = np.linspace(
                0, params[pkeys.FS] // 2, window_length // 2 + 1
            )
            # Drop O Hz
            outputs_llc = outputs_llc[..., 1:]
            frequency_axis = frequency_axis[1:]
            # Now we have a spectrogram of shape
            # [batch, times, freq=window_length//2] which is a power of 2
            # Now we could pool frequencies nicely
            pool_size = params[pkeys.LLC_STFT_FREQ_POOL]
            if pool_size is not None:
                outputs_llc = outputs_llc[..., tf.newaxis]
                outputs_llc = tf.keras.layers.AvgPool2D(pool_size=(1, pool_size))(
                    outputs_llc
                )
                outputs_llc = outputs_llc[..., 0]
                frequency_axis = (
                    frequency_axis.reshape((-1, pool_size)).mean(axis=1).flatten()
                )
            # Now we remove frequencies above 35Hz (our preprocessing range)
            remove_idx = np.where(frequency_axis > 35)[0][0]
            outputs_llc = outputs_llc[..., :remove_idx]
            if params[pkeys.LLC_STFT_USE_LOG]:
                outputs_llc = tf.log(outputs_llc + 1e-6)
            outputs_llc = tf.layers.batch_normalization(
                inputs=outputs_llc, training=training, name="bn_stft"
            )
            outputs_llc = tf.keras.layers.GlobalAvgPool1D()(outputs_llc)
            # output shape [batch, freq]
            # Now the dense layer
            n_hid = params[pkeys.LLC_STFT_N_HIDDEN]
            if n_hid > 0:
                outputs_llc = tf.keras.layers.Dense(n_hid)(outputs_llc)
                outputs_llc = tf.layers.batch_normalization(
                    inputs=outputs_llc, training=training, name="bn_stft_hid"
                )
                outputs_llc = tf.nn.relu(outputs_llc)
                outputs_llc = tf.layers.dropout(
                    outputs_llc,
                    training=training,
                    rate=params[pkeys.LLC_STFT_DROP_RATE],
                )
            # Now we are ready to produce linear projections of this vector

        # -----------------------------------------------------------
        # Now the convolutional stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        end_crop = (-border_crop) if (border_crop > 0) else None
        outputs = outputs[:, start_crop:end_crop, :]

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block_with_context(
            outputs,
            outputs_llc,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block_with_context(
            outputs,
            outputs_llc,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block_with_context(
            outputs,
            outputs_llc,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v11_llc_stft_1(
    inputs, params, training, name="model_v11_llc_stft_1"
):
    """
    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string) A name for the network.
    """
    print("Using model V11 + LLC-STFT 1 (only 3rd conv block)")
    with tf.variable_scope(name):
        # We assume the border is very big
        inputs = tf.expand_dims(inputs, axis=2)
        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        with tf.variable_scope("stft_module"):
            window_length = params[pkeys.LLC_STFT_N_SAMPLES]
            use_log = params[pkeys.LLC_STFT_USE_LOG]
            n_hidden = params[pkeys.LLC_STFT_N_HIDDEN]
            drop_rate = params[pkeys.LLC_STFT_DROP_RATE]

            outputs_llc = tf.signal.stft(
                outputs[:, :, 0],
                frame_length=window_length,
                frame_step=window_length // 2,
                name="stft",
            )
            norm_factor = 2 / np.sum(np.hanning(window_length))
            outputs_llc = tf.abs(outputs_llc) * norm_factor  # complex to real
            # Drop frequencies outside [0.5, 30] Hz
            frequency_axis = np.linspace(
                0, params[pkeys.FS] // 2, window_length // 2 + 1
            )
            lower_idx = np.where(frequency_axis < 0.5)[0][-1]
            upper_idx = np.where(frequency_axis > 30)[0][0]
            outputs_llc = outputs_llc[..., (lower_idx + 1) : upper_idx]
            # Optional logarithm
            outputs_llc = tf.log(outputs_llc + 1e-6) if use_log else outputs_llc
            # BN and global avg pooling -> [batch, freq]
            outputs_llc = tf.layers.batch_normalization(
                inputs=outputs_llc, training=training, name="bn_stft"
            )
            outputs_llc = tf.keras.layers.GlobalAvgPool1D()(outputs_llc)
            # First dense layer
            outputs_llc = tf.layers.dropout(
                outputs_llc, training=training, rate=drop_rate
            )
            outputs_llc = tf.keras.layers.Dense(n_hidden)(outputs_llc)
            outputs_llc = tf.nn.relu(outputs_llc)
            # Second dense layer
            outputs_llc = tf.layers.dropout(
                outputs_llc, training=training, rate=drop_rate
            )
            outputs_llc = tf.keras.layers.Dense(n_hidden)(outputs_llc)
            outputs_llc = tf.nn.relu(outputs_llc)
            # Now we are ready to produce linear projections of this vector

        # -----------------------------------------------------------
        # Now the convolutional stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        end_crop = (-border_crop) if (border_crop > 0) else None
        outputs = outputs[:, start_crop:end_crop, :]

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block_with_context(
            outputs,
            outputs_llc,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v11_llc_stft_2(
    inputs, params, training, name="model_v11_llc_stft_2"
):
    """
    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string) A name for the network.
    """
    print("Using model V11 + LLC-STFT 2 (first FC)")
    with tf.variable_scope(name):
        # We assume the border is very big
        inputs = tf.expand_dims(inputs, axis=2)
        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        with tf.variable_scope("stft_module"):
            window_length = params[pkeys.LLC_STFT_N_SAMPLES]
            use_log = params[pkeys.LLC_STFT_USE_LOG]
            n_hidden = params[pkeys.LLC_STFT_N_HIDDEN]
            drop_rate = params[pkeys.LLC_STFT_DROP_RATE]

            outputs_llc = tf.signal.stft(
                outputs[:, :, 0],
                frame_length=window_length,
                frame_step=window_length // 2,
                name="stft",
            )
            norm_factor = 2 / np.sum(np.hanning(window_length))
            outputs_llc = tf.abs(outputs_llc) * norm_factor  # complex to real
            # Drop frequencies outside [0.5, 30] Hz
            frequency_axis = np.linspace(
                0, params[pkeys.FS] // 2, window_length // 2 + 1
            )
            lower_idx = np.where(frequency_axis < 0.5)[0][-1]
            upper_idx = np.where(frequency_axis > 30)[0][0]
            outputs_llc = outputs_llc[..., (lower_idx + 1) : upper_idx]
            # Optional logarithm
            outputs_llc = tf.log(outputs_llc + 1e-6) if use_log else outputs_llc
            # BN and global avg pooling -> [batch, freq]
            outputs_llc = tf.layers.batch_normalization(
                inputs=outputs_llc, training=training, name="bn_stft"
            )
            outputs_llc = tf.keras.layers.GlobalAvgPool1D()(outputs_llc)
            # First dense layer
            outputs_llc = tf.layers.dropout(
                outputs_llc, training=training, rate=drop_rate
            )
            outputs_llc = tf.keras.layers.Dense(n_hidden)(outputs_llc)
            outputs_llc = tf.nn.relu(outputs_llc)
            # Second dense layer
            outputs_llc = tf.layers.dropout(
                outputs_llc, training=training, rate=drop_rate
            )
            outputs_llc = tf.keras.layers.Dense(n_hidden)(outputs_llc)
            outputs_llc = tf.nn.relu(outputs_llc)
            # Now we are ready to produce linear projections of this vector

        # -----------------------------------------------------------
        # Now the convolutional stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        end_crop = (-border_crop) if (border_crop > 0) else None
        outputs = outputs[:, start_crop:end_crop, :]

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer_with_context(
                outputs,
                outputs_llc,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v11_llc_stft_3(
    inputs, params, training, name="model_v11_llc_stft_3"
):
    """
    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string) A name for the network.
    """
    print("Using model V11 + LLC-STFT 3 (logits)")
    with tf.variable_scope(name):
        # We assume the border is very big
        inputs = tf.expand_dims(inputs, axis=2)
        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        with tf.variable_scope("stft_module"):
            window_length = params[pkeys.LLC_STFT_N_SAMPLES]
            use_log = params[pkeys.LLC_STFT_USE_LOG]
            n_hidden = params[pkeys.LLC_STFT_N_HIDDEN]
            drop_rate = params[pkeys.LLC_STFT_DROP_RATE]

            outputs_llc = tf.signal.stft(
                outputs[:, :, 0],
                frame_length=window_length,
                frame_step=window_length // 2,
                name="stft",
            )
            norm_factor = 2 / np.sum(np.hanning(window_length))
            outputs_llc = tf.abs(outputs_llc) * norm_factor  # complex to real
            # Drop frequencies outside [0.5, 30] Hz
            frequency_axis = np.linspace(
                0, params[pkeys.FS] // 2, window_length // 2 + 1
            )
            lower_idx = np.where(frequency_axis < 0.5)[0][-1]
            upper_idx = np.where(frequency_axis > 30)[0][0]
            outputs_llc = outputs_llc[..., (lower_idx + 1) : upper_idx]
            # Optional logarithm
            outputs_llc = tf.log(outputs_llc + 1e-6) if use_log else outputs_llc
            # BN and global avg pooling -> [batch, freq]
            outputs_llc = tf.layers.batch_normalization(
                inputs=outputs_llc, training=training, name="bn_stft"
            )
            outputs_llc = tf.keras.layers.GlobalAvgPool1D()(outputs_llc)
            # First dense layer
            outputs_llc = tf.layers.dropout(
                outputs_llc, training=training, rate=drop_rate
            )
            outputs_llc = tf.keras.layers.Dense(n_hidden)(outputs_llc)
            outputs_llc = tf.nn.relu(outputs_llc)
            # Second dense layer
            outputs_llc = tf.layers.dropout(
                outputs_llc, training=training, rate=drop_rate
            )
            outputs_llc = tf.keras.layers.Dense(n_hidden)(outputs_llc)
            outputs_llc = tf.nn.relu(outputs_llc)
            # Now we are ready to produce linear projections of this vector

        # -----------------------------------------------------------
        # Now the convolutional stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        end_crop = (-border_crop) if (border_crop > 0) else None
        outputs = outputs[:, start_crop:end_crop, :]

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer_with_context(
            outputs,
            outputs_llc,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v19_llc_stft_2(
    inputs, params, training, name="model_v19_llc_stft_2"
):
    print("Using model V19_llc_stft_2 (general cwt)")
    with tf.variable_scope(name):

        with tf.variable_scope("stft_module"):
            window_length = params[pkeys.LLC_STFT_N_SAMPLES]
            use_log = params[pkeys.LLC_STFT_USE_LOG]
            n_hidden = params[pkeys.LLC_STFT_N_HIDDEN]
            drop_rate = params[pkeys.LLC_STFT_DROP_RATE]

            outputs_llc = tf.signal.stft(
                inputs,
                frame_length=window_length,
                frame_step=window_length // 2,
                name="stft",
            )
            norm_factor = 2 / np.sum(np.hanning(window_length))
            outputs_llc = tf.abs(outputs_llc) * norm_factor  # complex to real
            # Drop frequencies outside [0.5, 30] Hz
            frequency_axis = np.linspace(
                0, params[pkeys.FS] // 2, window_length // 2 + 1
            )
            lower_idx = np.where(frequency_axis < 0.5)[0][-1]
            upper_idx = np.where(frequency_axis > 30)[0][0]
            outputs_llc = outputs_llc[..., (lower_idx + 1) : upper_idx]
            # Optional logarithm
            outputs_llc = tf.log(outputs_llc + 1e-6) if use_log else outputs_llc
            # BN and global avg pooling -> [batch, freq]
            outputs_llc = tf.layers.batch_normalization(
                inputs=outputs_llc, training=training, name="bn_stft"
            )
            outputs_llc = tf.keras.layers.GlobalAvgPool1D()(outputs_llc)
            # First dense layer
            outputs_llc = tf.layers.dropout(
                outputs_llc, training=training, rate=drop_rate
            )
            outputs_llc = tf.keras.layers.Dense(n_hidden)(outputs_llc)
            outputs_llc = tf.nn.relu(outputs_llc)
            # Second dense layer
            outputs_llc = tf.layers.dropout(
                outputs_llc, training=training, rate=drop_rate
            )
            outputs_llc = tf.keras.layers.Dense(n_hidden)(outputs_llc)
            outputs_llc = tf.nn.relu(outputs_llc)
            # Now we are ready to produce linear projections of this vector

        # CWT stage
        # Crop 5 seconds less of borders
        border_total = params[pkeys.BORDER_DURATION]
        border_pre_cwt = border_total - 5
        border_crop_pre_cwt = int(border_pre_cwt * params[pkeys.FS])
        start_crop = border_crop_pre_cwt
        end_crop = (-border_crop_pre_cwt) if (border_crop_pre_cwt > 0) else None
        inputs = inputs[:, start_crop:end_crop]
        border_crop = int(5 * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer_general(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            return_real_part=params[pkeys.CWT_RETURN_REAL_PART],
            return_imag_part=params[pkeys.CWT_RETURN_IMAG_PART],
            return_magnitude=params[pkeys.CWT_RETURN_MAGNITUDE],
            return_phase=params[pkeys.CWT_RETURN_PHASE],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1",
        )

        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_2",
        )

        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, "flatten")

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer_with_context(
                outputs,
                outputs_llc,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)

        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v19_llc_stft_3(
    inputs, params, training, name="model_v19_llc_stft_3"
):
    print("Using model V19_llc_stft_3 (general cwt)")
    with tf.variable_scope(name):
        with tf.variable_scope("stft_module"):
            window_length = params[pkeys.LLC_STFT_N_SAMPLES]
            use_log = params[pkeys.LLC_STFT_USE_LOG]
            n_hidden = params[pkeys.LLC_STFT_N_HIDDEN]
            drop_rate = params[pkeys.LLC_STFT_DROP_RATE]

            outputs_llc = tf.signal.stft(
                inputs,
                frame_length=window_length,
                frame_step=window_length // 2,
                name="stft",
            )
            norm_factor = 2 / np.sum(np.hanning(window_length))
            outputs_llc = tf.abs(outputs_llc) * norm_factor  # complex to real
            # Drop frequencies outside [0.5, 30] Hz
            frequency_axis = np.linspace(
                0, params[pkeys.FS] // 2, window_length // 2 + 1
            )
            lower_idx = np.where(frequency_axis < 0.5)[0][-1]
            upper_idx = np.where(frequency_axis > 30)[0][0]
            outputs_llc = outputs_llc[..., (lower_idx + 1) : upper_idx]
            # Optional logarithm
            outputs_llc = tf.log(outputs_llc + 1e-6) if use_log else outputs_llc
            # BN and global avg pooling -> [batch, freq]
            outputs_llc = tf.layers.batch_normalization(
                inputs=outputs_llc, training=training, name="bn_stft"
            )
            outputs_llc = tf.keras.layers.GlobalAvgPool1D()(outputs_llc)
            # First dense layer
            outputs_llc = tf.layers.dropout(
                outputs_llc, training=training, rate=drop_rate
            )
            outputs_llc = tf.keras.layers.Dense(n_hidden)(outputs_llc)
            outputs_llc = tf.nn.relu(outputs_llc)
            # Second dense layer
            outputs_llc = tf.layers.dropout(
                outputs_llc, training=training, rate=drop_rate
            )
            outputs_llc = tf.keras.layers.Dense(n_hidden)(outputs_llc)
            outputs_llc = tf.nn.relu(outputs_llc)
            # Now we are ready to produce linear projections of this vector

        # CWT stage
        # Crop 5 seconds less of borders
        border_total = params[pkeys.BORDER_DURATION]
        border_pre_cwt = border_total - 5
        border_crop_pre_cwt = int(border_pre_cwt * params[pkeys.FS])
        start_crop = border_crop_pre_cwt
        end_crop = (-border_crop_pre_cwt) if (border_crop_pre_cwt > 0) else None
        inputs = inputs[:, start_crop:end_crop]
        border_crop = int(5 * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer_general(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            return_real_part=params[pkeys.CWT_RETURN_REAL_PART],
            return_imag_part=params[pkeys.CWT_RETURN_IMAG_PART],
            return_magnitude=params[pkeys.CWT_RETURN_MAGNITUDE],
            return_phase=params[pkeys.CWT_RETURN_PHASE],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1",
        )

        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_2",
        )

        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, "flatten")

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer_with_context(
            outputs,
            outputs_llc,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)

        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_tcn01(inputs, params, training, name="model_tcn01"):
    print("Using model TCN01 (Time-Domain)")
    with tf.variable_scope(name):
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)
        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # Now the TCN part
        n_blocks = params[pkeys.TCN_N_BLOCKS]
        for i in range(n_blocks):
            dilation = 2**i
            outputs = layers.tcn_block(
                outputs,
                params[pkeys.TCN_FILTERS],
                params[pkeys.TCN_KERNEL_SIZE],
                dilation,
                params[pkeys.TCN_DROP_RATE],
                training,
                bottleneck=params[pkeys.TCN_USE_BOTTLENECK],
                is_first_unit=(i == 0),
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                kernel_init=tf.initializers.he_normal(),
                name="tcn_block_%d" % i,
            )

        # The crop is performed at the end
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS] / 8)
        outputs = outputs[:, border_crop:-border_crop, :]

        batchnorm = params[pkeys.TYPE_BATCHNORM]
        if batchnorm:
            outputs = layers.batchnorm_layer(
                outputs, "bn_last", batchnorm=batchnorm, training=training, scale=False
            )
        outputs = tf.nn.relu(outputs)

        for i in range(params[pkeys.TCN_LAST_CONV_N_LAYERS]):
            outputs = layers.conv1d_prebn(
                outputs,
                params[pkeys.TCN_LAST_CONV_FILTERS],
                params[pkeys.TCN_LAST_CONV_KERNEL_SIZE],
                training,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                kernel_init=tf.initializers.he_normal(),
                name="last_conv_%i" % i,
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_tcn02(inputs, params, training, name="model_tcn02"):
    print("Using model TCN02 (Time-Domain)")
    with tf.variable_scope(name):
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)
        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # Now the TCN part
        n_blocks = params[pkeys.TCN_N_BLOCKS]
        for i in range(n_blocks):
            dilation = 2**i
            print("Dilation", dilation)
            outputs = layers.tcn_block(
                outputs,
                params[pkeys.TCN_FILTERS],
                params[pkeys.TCN_KERNEL_SIZE],
                dilation,
                params[pkeys.TCN_DROP_RATE],
                training,
                bottleneck=params[pkeys.TCN_USE_BOTTLENECK],
                is_first_unit=(i == 0),
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                kernel_init=tf.initializers.he_normal(),
                name="tcn_block_up_%d" % i,
            )
        for i in range(n_blocks - 1):
            dilation = 2 ** (n_blocks - 2 - i)
            print("Dilation", dilation)
            outputs = layers.tcn_block(
                outputs,
                params[pkeys.TCN_FILTERS],
                params[pkeys.TCN_KERNEL_SIZE],
                dilation,
                params[pkeys.TCN_DROP_RATE],
                training,
                bottleneck=params[pkeys.TCN_USE_BOTTLENECK],
                is_first_unit=False,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                kernel_init=tf.initializers.he_normal(),
                name="tcn_block_down_%d" % i,
            )

        # The crop is performed at the end
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS] / 8)
        outputs = outputs[:, border_crop:-border_crop, :]

        batchnorm = params[pkeys.TYPE_BATCHNORM]
        if batchnorm:
            outputs = layers.batchnorm_layer(
                outputs, "bn_last", batchnorm=batchnorm, training=training, scale=False
            )
        outputs = tf.nn.relu(outputs)

        for i in range(params[pkeys.TCN_LAST_CONV_N_LAYERS]):
            outputs = layers.conv1d_prebn(
                outputs,
                params[pkeys.TCN_LAST_CONV_FILTERS],
                params[pkeys.TCN_LAST_CONV_KERNEL_SIZE],
                training,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                kernel_init=tf.initializers.he_normal(),
                name="last_conv_%i" % i,
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_tcn03(inputs, params, training, name="model_tcn03"):
    print("Using model TCN03 (Time-Domain, TCN01 without residual)")
    with tf.variable_scope(name):
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)
        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # Now the TCN part
        n_blocks = params[pkeys.TCN_N_BLOCKS]
        for i in range(n_blocks):
            dilation = 2**i
            outputs = layers.tcn_block_simple(
                outputs,
                params[pkeys.TCN_FILTERS],
                params[pkeys.TCN_KERNEL_SIZE],
                dilation,
                params[pkeys.TCN_DROP_RATE],
                training,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                kernel_init=tf.initializers.he_normal(),
                name="tcn_%d" % i,
            )

        # The crop is performed at the end
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS] / 8)
        outputs = outputs[:, border_crop:-border_crop, :]

        for i in range(params[pkeys.TCN_LAST_CONV_N_LAYERS]):
            outputs = layers.conv1d_prebn(
                outputs,
                params[pkeys.TCN_LAST_CONV_FILTERS],
                params[pkeys.TCN_LAST_CONV_KERNEL_SIZE],
                training,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                kernel_init=tf.initializers.he_normal(),
                name="last_conv_%i" % i,
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_tcn04(inputs, params, training, name="model_tcn04"):
    print("Using model TCN04 (Time-Domain, TCN02 without residual)")
    with tf.variable_scope(name):
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)
        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # 1D convolutions expect shape [batch, time_len, n_feats]

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_1",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_2",
        )

        outputs = layers.conv1d_prebn_block(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3",
        )

        # Now the TCN part
        n_blocks = params[pkeys.TCN_N_BLOCKS]
        for i in range(n_blocks):
            dilation = 2**i
            print("Dilation", dilation)
            outputs = layers.tcn_block_simple(
                outputs,
                params[pkeys.TCN_FILTERS],
                params[pkeys.TCN_KERNEL_SIZE],
                dilation,
                params[pkeys.TCN_DROP_RATE],
                training,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                kernel_init=tf.initializers.he_normal(),
                name="tcn_block_up_%d" % i,
            )
        for i in range(n_blocks - 1):
            dilation = 2 ** (n_blocks - 2 - i)
            print("Dilation", dilation)
            outputs = layers.tcn_block_simple(
                outputs,
                params[pkeys.TCN_FILTERS],
                params[pkeys.TCN_KERNEL_SIZE],
                dilation,
                params[pkeys.TCN_DROP_RATE],
                training,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                kernel_init=tf.initializers.he_normal(),
                name="tcn_block_down_%d" % i,
            )

        # The crop is performed at the end
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS] / 8)
        outputs = outputs[:, border_crop:-border_crop, :]

        for i in range(params[pkeys.TCN_LAST_CONV_N_LAYERS]):
            outputs = layers.conv1d_prebn(
                outputs,
                params[pkeys.TCN_LAST_CONV_FILTERS],
                params[pkeys.TCN_LAST_CONV_KERNEL_SIZE],
                training,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                kernel_init=tf.initializers.he_normal(),
                name="last_conv_%i" % i,
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v19_frozen(inputs, params, training, name="model_v19_frozen"):
    print("Using model V19 with BN at cwt frozen(general cwt)")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer_general(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            return_real_part=params[pkeys.CWT_RETURN_REAL_PART],
            return_imag_part=params[pkeys.CWT_RETURN_IMAG_PART],
            return_magnitude=params[pkeys.CWT_RETURN_MAGNITUDE],
            return_phase=params[pkeys.CWT_RETURN_PHASE],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=None,
            name="spectrum",
        )

        # Fixed batchnorm
        print("Using fixed normalization after CWT")
        bn_cwt_weights = np.load(os.path.join(PATH_RESOURCES, "bn_cwt_weights.npz"))
        bn_cwt_beta = bn_cwt_weights["beta"].reshape(1, 1, 32, 1)
        bn_cwt_gamma = bn_cwt_weights["gamma"].reshape(1, 1, 32, 1)
        bn_cwt_moving_mean = bn_cwt_weights["moving_mean"].reshape(1, 1, 32, 1)
        bn_cwt_moving_variance = bn_cwt_weights["moving_variance"].reshape(1, 1, 32, 1)
        epsilon = 0.001
        bn_cwt_std = np.sqrt(bn_cwt_moving_variance + epsilon)
        outputs = (outputs - bn_cwt_moving_mean) / bn_cwt_std
        outputs = bn_cwt_gamma * outputs + bn_cwt_beta

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1",
        )

        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_2",
        )

        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, "flatten")

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)

        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v19_var(inputs, params, training, name="model_v19_var"):
    """Wavelet transform, conv, and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a standard convolutional stage
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v19') A name for the network.
    """
    print("Using model V19 with pooled scales (general cwt)")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer_general(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            return_real_part=params[pkeys.CWT_RETURN_REAL_PART],
            return_imag_part=params[pkeys.CWT_RETURN_IMAG_PART],
            return_magnitude=params[pkeys.CWT_RETURN_MAGNITUDE],
            return_phase=params[pkeys.CWT_RETURN_PHASE],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=128,
            pool_scales=4,
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1",
        )

        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_2",
        )

        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, "flatten")

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)

        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v19_noisy(inputs, params, training, name="model_v19_noisy"):
    """Wavelet transform, conv, and BLSTM to make a prediction.

    This models first computes the CWT to form scalograms, and then this
    scalograms are processed by a standard convolutional stage
    (pre-activation BN). After this, the outputs is flatten and is passed to
    a 2-layers BLSTM.
    The final classification is made with a 2 layers FC with 2 outputs.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        params: (dict) Parameters to configure the model (see common.param_keys)
        training: (boolean) Indicates if it is the training phase or not.
        name: (Optional, string, defaults to 'model_v19') A name for the network.
    """
    print("Using model V19 noisy (general cwt + noise at scales)")
    with tf.variable_scope(name):
        # CWT stage
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])

        outputs, cwt_prebn = layers.cmorlet_layer_general_noisy(
            inputs,
            params[pkeys.FB_LIST],
            params[pkeys.FS],
            noise_intensity=params[pkeys.CWT_NOISE_INTENSITY],
            return_real_part=params[pkeys.CWT_RETURN_REAL_PART],
            return_imag_part=params[pkeys.CWT_RETURN_IMAG_PART],
            return_magnitude=params[pkeys.CWT_RETURN_MAGNITUDE],
            return_phase=params[pkeys.CWT_RETURN_PHASE],
            lower_freq=params[pkeys.LOWER_FREQ],
            upper_freq=params[pkeys.UPPER_FREQ],
            n_scales=params[pkeys.N_SCALES],
            stride=2,
            size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
            border_crop=border_crop,
            training=training,
            trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            name="spectrum",
        )

        # Convolutional stage (standard feed-forward)
        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_1],
            training,
            kernel_size_1=params[pkeys.INITIAL_KERNEL_SIZE],
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1",
        )

        outputs = layers.conv2d_prebn_block(
            outputs,
            params[pkeys.CWT_CONV_FILTERS_2],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_2",
        )

        # Flattening for dense part
        outputs = layers.sequence_flatten(outputs, "flatten")

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            params[pkeys.INITIAL_LSTM_UNITS],
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=params[pkeys.DROP_RATE_BEFORE_LSTM],
            drop_rate_rest_lstm=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="multi_layer_blstm",
        )

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.relu,
                name="fc_1",
            )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)

        return logits, probabilities, cwt_prebn
