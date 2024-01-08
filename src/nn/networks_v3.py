from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

PATH_THIS_DIR = os.path.dirname(__file__)
PATH_RESOURCES = os.path.join(PATH_THIS_DIR, '..', '..', 'resources')

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from src.nn import layers
from src.common import constants
from src.common import pkeys


def stage_cwt(inputs, params, training):
    border_crop = int(np.round(params[pkeys.BORDER_DURATION_CWT] * params[pkeys.FS]))
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
        use_avg_pool=False,
        size_factor=params[pkeys.WAVELET_SIZE_FACTOR],
        expansion_factor=params[pkeys.CWT_EXPANSION_FACTOR],
        border_crop=border_crop,
        training=training,
        trainable_wavelet=params[pkeys.TRAINABLE_WAVELET],
        batchnorm=params[pkeys.TYPE_BATCHNORM],
        name='spectrum')
    other_outputs_dict = {'cwt': cwt_prebn}
    return outputs, other_outputs_dict


def stage_multi_dilated_convolutions(inputs, params, training, filters, name, is_2d=False):
    stage_config = []
    max_exponent = int(np.round(np.log(params[pkeys.BIGGER_MAX_DILATION]) / np.log(2)))
    for single_exponent in range(max_exponent + 1):
        f = int(filters / (2 ** (single_exponent + 1)))
        d = int(2 ** single_exponent)
        stage_config.append((f, d))
    stage_config[-1] = (2 * stage_config[-1][0], stage_config[-1][1])

    with tf.variable_scope(name):
        if is_2d:
            # input is [batch_size, time_len, scales, feats]
            outputs = inputs
            kernel_size_adapt = lambda x: (x, x)
        else:
            # [batch_size, time_len, n_feats] -> [batch_size, time_len, 1, feats]
            outputs = tf.expand_dims(inputs, axis=2)
            kernel_size_adapt = lambda x: (x, 1)

        outputs_branches = []
        for branch_filters, branch_dilation in stage_config:
            with tf.variable_scope("branch-d%d" % branch_dilation):
                branch_outputs = outputs
                for layer_id in ['a', 'b']:
                    branch_outputs = tf.keras.layers.Conv2D(
                        filters=branch_filters,
                        kernel_size=kernel_size_adapt(3),
                        padding=constants.PAD_SAME,
                        dilation_rate=(branch_dilation, 1),
                        use_bias=False,
                        kernel_initializer=tf.initializers.he_normal(),
                        name='conv3-d%d-%s' % (branch_dilation, layer_id))(branch_outputs)
                    branch_outputs = layers.batchnorm_layer(
                        branch_outputs, 'bn-%s' % layer_id, batchnorm=params[pkeys.TYPE_BATCHNORM],
                        training=training, scale=False)
                    branch_outputs = tf.nn.relu(branch_outputs)
                outputs_branches.append(branch_outputs)
        outputs = tf.concat(outputs_branches, axis=-1)
        if not is_2d:
            # [batch_size, time_len, 1, n_units] -> [batch_size, time_len, n_units]
            outputs = tf.squeeze(outputs, axis=2, name="squeeze")
    return outputs


def stage_recurrent(inputs, params, training):
    outputs = inputs
    if params[pkeys.BIGGER_LSTM_1_SIZE] > 0:
        outputs = layers.lstm_layer(
            outputs,
            num_units=params[pkeys.BIGGER_LSTM_1_SIZE],
            num_dirs=constants.BIDIRECTIONAL,
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_BEFORE_LSTM],
            training=training,
            name='lstm_1')
    if params[pkeys.BIGGER_LSTM_2_SIZE] > 0:
        outputs = layers.lstm_layer(
            outputs,
            num_units=params[pkeys.BIGGER_LSTM_2_SIZE],
            num_dirs=constants.BIDIRECTIONAL,
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name='lstm_2')
    return outputs


def stage_classification(inputs, params, training):
    outputs = inputs
    if params[pkeys.FC_UNITS] > 0:
        outputs = layers.sequence_fc_layer(
            outputs,
            params[pkeys.FC_UNITS],
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            activation=tf.nn.relu,
            name='conv1')
    logits = layers.sequence_output_2class_layer(
        outputs,
        kernel_init=tf.initializers.he_normal(),
        dropout=params[pkeys.TYPE_DROPOUT],
        drop_rate=params[pkeys.DROP_RATE_OUTPUT],
        training=training,
        init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
        name='logits')
    with tf.variable_scope('probabilities'):
        probabilities = tf.nn.softmax(logits)
    other_outputs_dict = {'last_hidden': outputs}
    return logits, probabilities, other_outputs_dict


def crop_time_in_tensor(tensor, border_crop):
    # shape assumed: [batch, time, ...]
    start_crop = border_crop
    end_crop = None if (border_crop <= 0) else -border_crop
    tensor = tensor[:, start_crop:end_crop]
    return tensor


def redv2_time(
        inputs,
        params,
        training,
        name='model_redv2_time'
):
    print('Using model REDv2-Time')
    fs_after_conv = params[pkeys.FS] // 8
    border_duration_lstm = params[pkeys.BORDER_DURATION] - params[pkeys.BORDER_DURATION_CONV]
    border_crop_conv = int(np.round(params[pkeys.BORDER_DURATION_CONV] * fs_after_conv))
    border_crop_lstm = int(np.round(border_duration_lstm * fs_after_conv))
    with tf.variable_scope(name):
        outputs = tf.expand_dims(inputs, axis=2)  # [batch, time_len] -> [batch, time_len, 1]
        outputs = layers.batchnorm_layer(
            outputs, 'bn_input', batchnorm=params[pkeys.TYPE_BATCHNORM], training=training)
        with tf.variable_scope("stem"):
            for layer_num in [1, 2]:
                outputs = tf.keras.layers.Conv1D(
                    filters=params[pkeys.BIGGER_STEM_FILTERS],
                    kernel_size=3,
                    padding=constants.PAD_SAME,
                    use_bias=False,
                    kernel_initializer=tf.initializers.he_normal(),
                    name='conv3_%d' % layer_num)(outputs)
                outputs = layers.batchnorm_layer(
                    outputs, 'bn_%d' % layer_num, batchnorm=params[pkeys.TYPE_BATCHNORM],
                    training=training, scale=False)
                outputs = tf.nn.relu(outputs)
        outputs = tf.keras.layers.AvgPool1D(name='pool1')(outputs)
        filters = params[pkeys.BIGGER_STEM_FILTERS] * 2
        outputs = stage_multi_dilated_convolutions(outputs, params, training, filters, 'mdconv_1', is_2d=False)
        outputs = tf.keras.layers.AvgPool1D(name='pool2')(outputs)
        filters = params[pkeys.BIGGER_STEM_FILTERS] * 4
        outputs = stage_multi_dilated_convolutions(outputs, params, training, filters, 'mdconv_2', is_2d=False)
        outputs = tf.keras.layers.AvgPool1D(name='pool3')(outputs)
        outputs = crop_time_in_tensor(outputs, border_crop_conv)
        outputs = stage_recurrent(outputs, params, training)
        outputs = crop_time_in_tensor(outputs, border_crop_lstm)
        logits, probabilities, other_outputs_dict = stage_classification(outputs, params, training)
    return logits, probabilities, other_outputs_dict


def redv2_cwt1d(
        inputs,
        params,
        training,
        name='model_redv2_cwt1d'
):
    print('Using model REDv2-CWT1D')
    fs_after_conv = params[pkeys.FS] // 8
    border_duration_lstm = params[pkeys.BORDER_DURATION] - params[pkeys.BORDER_DURATION_CONV] - params[pkeys.BORDER_DURATION_CWT]
    border_crop_conv = int(np.round(params[pkeys.BORDER_DURATION_CONV] * fs_after_conv))
    border_crop_lstm = int(np.round(border_duration_lstm * fs_after_conv))
    with tf.variable_scope(name):
        outputs, other_outputs_dict_cwt = stage_cwt(inputs, params, training)  # [batch, time, scales, channels]
        outputs = layers.sequence_flatten(outputs, 'flatten')  # [batch, time, scales * channels]
        with tf.variable_scope("stem"):
            outputs = tf.keras.layers.Conv1D(
                filters=params[pkeys.BIGGER_STEM_FILTERS],
                kernel_size=3,
                padding=constants.PAD_SAME,
                use_bias=False,
                kernel_initializer=tf.initializers.he_normal(),
                name='conv3')(outputs)
            outputs = layers.batchnorm_layer(
                outputs, 'bn', batchnorm=params[pkeys.TYPE_BATCHNORM],
                training=training, scale=False)
            outputs = tf.nn.relu(outputs)
        filters = params[pkeys.BIGGER_STEM_FILTERS] * 2
        outputs = stage_multi_dilated_convolutions(outputs, params, training, filters, 'mdconv_1', is_2d=False)
        outputs = tf.keras.layers.AvgPool1D(name='pool2')(outputs)
        filters = params[pkeys.BIGGER_STEM_FILTERS] * 4
        outputs = stage_multi_dilated_convolutions(outputs, params, training, filters, 'mdconv_2', is_2d=False)
        outputs = tf.keras.layers.AvgPool1D(name='pool3')(outputs)
        outputs = crop_time_in_tensor(outputs, border_crop_conv)
        outputs = stage_recurrent(outputs, params, training)
        outputs = crop_time_in_tensor(outputs, border_crop_lstm)
        logits, probabilities, other_outputs_dict = stage_classification(outputs, params, training)
        other_outputs_dict.update(other_outputs_dict_cwt)
    return logits, probabilities, other_outputs_dict


def redv2_cwt2d(
        inputs,
        params,
        training,
        name='model_redv2_cwt2d'
):
    print('Using model REDv2-CWT2D')
    fs_after_conv = params[pkeys.FS] // 8
    border_duration_lstm = params[pkeys.BORDER_DURATION] - params[pkeys.BORDER_DURATION_CONV] - params[pkeys.BORDER_DURATION_CWT]
    border_crop_conv = int(np.round(params[pkeys.BORDER_DURATION_CONV] * fs_after_conv))
    border_crop_lstm = int(np.round(border_duration_lstm * fs_after_conv))
    with tf.variable_scope(name):
        outputs, other_outputs_dict_cwt = stage_cwt(inputs, params, training)  # [batch, time, scales, channels]
        with tf.variable_scope("stem"):
            outputs = tf.keras.layers.Conv2D(
                filters=params[pkeys.BIGGER_STEM_FILTERS],
                kernel_size=3,
                padding=constants.PAD_SAME,
                use_bias=False,
                kernel_initializer=tf.initializers.he_normal(),
                name='conv3')(outputs)
            outputs = layers.batchnorm_layer(
                outputs, 'bn', batchnorm=params[pkeys.TYPE_BATCHNORM],
                training=training, scale=False)
            outputs = tf.nn.relu(outputs)
        filters = params[pkeys.BIGGER_STEM_FILTERS] * 2
        outputs = stage_multi_dilated_convolutions(outputs, params, training, filters, 'mdconv_1', is_2d=True)
        outputs = tf.keras.layers.AvgPool2D(name='pool2')(outputs)
        filters = params[pkeys.BIGGER_STEM_FILTERS] * 4
        outputs = stage_multi_dilated_convolutions(outputs, params, training, filters, 'mdconv_2', is_2d=True)
        outputs = tf.keras.layers.AvgPool2D(name='pool3')(outputs)
        outputs = crop_time_in_tensor(outputs, border_crop_conv)
        outputs = layers.sequence_flatten(outputs, 'flatten')  # [batch, time, scales * channels]
        outputs = stage_recurrent(outputs, params, training)
        outputs = crop_time_in_tensor(outputs, border_crop_lstm)
        logits, probabilities, other_outputs_dict = stage_classification(outputs, params, training)
        other_outputs_dict.update(other_outputs_dict_cwt)
    return logits, probabilities, other_outputs_dict
