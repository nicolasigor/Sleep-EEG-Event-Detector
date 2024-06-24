from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

PATH_THIS_DIR = os.path.dirname(__file__)
PATH_RESOURCES = os.path.join(PATH_THIS_DIR, "..", "..", "resources")

import numpy as np
import tensorflow as tf

from sleeprnn.nn.expert_feats import a7_layer_tf, bandpass_tf_batch
from sleeprnn.nn import layers
from sleeprnn.nn import spectrum
from sleeprnn.nn import expert_feats
from sleeprnn.common import constants
from sleeprnn.common import checks
from sleeprnn.common import pkeys


def wavelet_blstm_net_att05(inputs, params, training, name="model_att05"):
    print("Using model ATT05 (CWT-Domain)")
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
        outputs_flatten = layers.sequence_flatten(outputs, "flatten")

        # --------------------------
        # Attention layer
        # input shape: [batch, time_len, n_feats]
        # --------------------------
        original_length = params[pkeys.PAGE_DURATION] * params[pkeys.FS]
        seq_len = int(original_length / params[pkeys.TOTAL_DOWNSAMPLING_FACTOR])
        att_dim = params[pkeys.ATT_DIM]
        n_heads = params[pkeys.ATT_N_HEADS]

        # bands parameters
        v_add_band_enc = params[pkeys.ATT_BANDS_V_ADD_BAND_ENC]
        k_add_band_enc = params[pkeys.ATT_BANDS_K_ADD_BAND_ENC]
        v_indep_linear = params[pkeys.ATT_BANDS_V_INDEP_LINEAR]
        k_indep_linear = params[pkeys.ATT_BANDS_K_INDEP_LINEAR]
        n_bands = params[pkeys.N_SCALES] // 4

        with tf.variable_scope("attention"):

            # Multilayer BLSTM (2 layers)
            after_lstm_outputs = layers.multilayer_lstm_block(
                outputs_flatten,
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

            # Prepare positional encoding
            with tf.variable_scope("pos_enc"):
                pos_enc = layers.get_positional_encoding(
                    seq_len=seq_len,
                    dims=att_dim,
                    pe_factor=params[pkeys.ATT_PE_FACTOR],
                    name="pos_enc",
                )
                pos_enc_1d = tf.expand_dims(pos_enc, axis=0)  # Add batch axis
                pos_enc = tf.expand_dims(pos_enc_1d, axis=2)  # Add band axis
                # shape [1, time, 1, dim]

            # Prepare band encoding
            if v_add_band_enc or k_add_band_enc:
                with tf.variable_scope("band_enc"):
                    bands_labels = list(range(n_bands))
                    bands_oh = tf.one_hot(bands_labels, depth=n_bands)
                    bands_oh = tf.expand_dims(bands_oh, axis=0)  # Add batch axis
                    # shape [1, n_bands, n_bands]
                    bands_enc = layers.sequence_fc_layer(
                        bands_oh,
                        att_dim,
                        kernel_init=tf.initializers.he_normal(),
                        training=training,
                        use_bias=False,
                        name="band_enc",
                    )
                    # shape [1, n_bands, dim]
                    bands_enc = tf.expand_dims(bands_enc, axis=1)  # Add time axis
                    # shape [1, 1, n_bands, dim]

            # Prepare input for values
            with tf.variable_scope("values_prep"):
                # input shape [batch, time, n_bands, feats]
                if v_indep_linear:  # indep projections
                    with tf.variable_scope("fc_embed_v_indep"):
                        outputs_unstack = tf.unstack(outputs, axis=2)
                        projected_list = []
                        for i, output_band in enumerate(outputs_unstack):
                            output_band = tf.expand_dims(output_band, axis=2)
                            # shape [batch, time, 1, feats]
                            output_band = tf.layers.conv2d(
                                inputs=output_band,
                                filters=att_dim,
                                kernel_size=1,
                                name="fc_embed_v_%d" % i,
                                use_bias=False,
                                kernel_initializer=tf.initializers.he_normal(),
                            )
                            projected_list.append(output_band)
                            # shape [batch, time, 1, dim]
                        v_outputs = tf.concat(projected_list, axis=2)
                else:  # shared projection
                    v_outputs = tf.layers.conv2d(
                        inputs=outputs,
                        filters=att_dim,
                        kernel_size=1,
                        name="fc_embed_v",
                        use_bias=False,
                        kernel_initializer=tf.initializers.he_normal(),
                    )
                v_outputs = v_outputs + pos_enc
                if v_add_band_enc:
                    v_outputs = v_outputs + bands_enc
                # shape [batch, time, n_bands, dim]
                v_outputs = tf.reshape(
                    v_outputs,
                    shape=(-1, seq_len * n_bands, att_dim),
                    name="flatten_bands_v",
                )
                # shape [batch, time * n_bands, dim]
                v_outputs = layers.dropout_layer(
                    v_outputs,
                    "drop_embed_v",
                    drop_rate=params[pkeys.ATT_DROP_RATE],
                    dropout=params[pkeys.TYPE_DROPOUT],
                    training=training,
                )
                # shape [batch, time * n_bands, dim]

            # Prepare input for queries
            with tf.variable_scope("queries_prep"):
                q_outputs = layers.sequence_fc_layer(
                    after_lstm_outputs,
                    att_dim,
                    kernel_init=tf.initializers.he_normal(),
                    training=training,
                    use_bias=False,
                    name="fc_embed_q",
                )
                q_outputs = q_outputs + pos_enc_1d
                q_outputs = layers.dropout_layer(
                    q_outputs,
                    "drop_embed_q",
                    drop_rate=params[pkeys.ATT_DROP_RATE],
                    dropout=params[pkeys.TYPE_DROPOUT],
                    training=training,
                )
                # shape [batch, time, dim]

            # Prepare input for keys
            with tf.variable_scope("keys_prep"):
                after_lstm_outputs_2d = tf.expand_dims(after_lstm_outputs, axis=2)
                # input shape [batch, time, 1, dim]
                if k_indep_linear:  # indep projections
                    with tf.variable_scope("fc_embed_k_indep"):
                        projected_list = []
                        for i in range(n_bands):
                            output_band = tf.layers.conv2d(
                                inputs=after_lstm_outputs_2d,
                                filters=att_dim,
                                kernel_size=1,
                                name="fc_embed_k_%d" % i,
                                use_bias=False,
                                kernel_initializer=tf.initializers.he_normal(),
                            )
                            projected_list.append(output_band)
                            # shape [batch, time, 1, dim]
                        k_outputs = tf.concat(projected_list, axis=2)
                        # shape [batch, time, n_bands, dim]
                else:  # shared projection
                    k_outputs = tf.layers.conv2d(
                        inputs=after_lstm_outputs_2d,
                        filters=att_dim,
                        kernel_size=1,
                        name="fc_embed_k",
                        use_bias=False,
                        kernel_initializer=tf.initializers.he_normal(),
                    )
                    # shape [batch, time, 1, dim]
                k_outputs = k_outputs + pos_enc
                if k_add_band_enc:
                    k_outputs = k_outputs + bands_enc
                # shape [batch, time, n_bands, dim]
                k_outputs = tf.reshape(
                    k_outputs,
                    shape=(-1, seq_len * n_bands, att_dim),
                    name="flatten_bands_k",
                )
                # shape [batch, time * n_bands, dim]
                k_outputs = layers.dropout_layer(
                    k_outputs,
                    "drop_embed_k",
                    drop_rate=params[pkeys.ATT_DROP_RATE],
                    dropout=params[pkeys.TYPE_DROPOUT],
                    training=training,
                )
                # shape [batch, time * n_bands, dim]

            # Prepare queries, keys, and values
            queries = layers.sequence_fc_layer(
                q_outputs,
                att_dim,
                kernel_init=tf.initializers.he_normal(),
                training=training,
                use_bias=False,
                name="queries",
            )
            keys = layers.sequence_fc_layer(
                k_outputs,
                att_dim,
                kernel_init=tf.initializers.he_normal(),
                training=training,
                use_bias=False,
                name="keys",
            )
            values = layers.sequence_fc_layer(
                v_outputs,
                att_dim,
                kernel_init=tf.initializers.he_normal(),
                training=training,
                use_bias=False,
                name="values",
            )

            outputs = layers.naive_multihead_attention_layer(
                queries, keys, values, n_heads, name="multi_head_att"
            )
            # should be [batch, time, dim]

            # FFN
            outputs = layers.sequence_fc_layer(
                outputs,
                2 * params[pkeys.ATT_DIM],
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


def deep_a7_v1(inputs, params, training, name="model_a7_v1"):
    print("Using model A7_V1 (A7 feats, convolutional)")
    with tf.variable_scope(name):
        # input is [batch, time_len]
        inputs = a7_layer_tf(
            inputs,
            fs=params[pkeys.FS],
            window_duration=params[pkeys.A7_WINDOW_DURATION],
            window_duration_relSigPow=params[pkeys.A7_WINDOW_DURATION_REL_SIG_POW],
            use_log_absSigPow=params[pkeys.A7_USE_LOG_ABS_SIG_POW],
            use_log_relSigPow=params[pkeys.A7_USE_LOG_REL_SIG_POW],
            use_log_sigCov=params[pkeys.A7_USE_LOG_SIG_COV],
            use_log_sigCorr=params[pkeys.A7_USE_LOG_SIG_CORR],
            use_zscore_relSigPow=params[pkeys.A7_USE_ZSCORE_REL_SIG_POW],
            use_zscore_sigCov=params[pkeys.A7_USE_ZSCORE_SIG_COV],
            use_zscore_sigCorr=params[pkeys.A7_USE_ZSCORE_SIG_CORR],
            remove_delta_in_cov=params[pkeys.A7_REMOVE_DELTA_IN_COV],
            dispersion_mode=params[pkeys.A7_DISPERSION_MODE],
        )

        # Now is [batch, time_len, 4]
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop, :]

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # Now pool to get proper length
        outputs = tf.keras.layers.AveragePooling1D(pool_size=8)(outputs)

        # Now convolutions
        kernel_size = params[pkeys.A7_CNN_KERNEL_SIZE]
        n_layers = params[pkeys.A7_CNN_N_LAYERS]
        filters = params[pkeys.A7_CNN_FILTERS]
        drop_rate_conv = params[pkeys.A7_CNN_DROP_RATE]
        for i in range(n_layers):
            with tf.variable_scope("conv_%d" % i):
                if i > 0 and drop_rate_conv > 0:
                    outputs = layers.dropout_layer(
                        outputs,
                        "drop_%d" % i,
                        training,
                        dropout=params[pkeys.TYPE_DROPOUT],
                        drop_rate=drop_rate_conv,
                    )
                outputs = tf.keras.layers.Conv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    padding="same",
                    use_bias=False,
                    kernel_initializer=tf.initializers.he_normal(),
                )(outputs)
                outputs = layers.batchnorm_layer(
                    outputs,
                    "bn_%d" % i,
                    batchnorm=params[pkeys.TYPE_BATCHNORM],
                    training=training,
                    scale=False,
                )
                outputs = tf.nn.relu(outputs)

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


def deep_a7_v2(inputs, params, training, name="model_a7_v2"):
    print("Using model A7_V2 (A7 feats, recurrent)")
    with tf.variable_scope(name):
        # input is [batch, time_len]
        inputs = a7_layer_tf(
            inputs,
            fs=params[pkeys.FS],
            window_duration=params[pkeys.A7_WINDOW_DURATION],
            window_duration_relSigPow=params[pkeys.A7_WINDOW_DURATION_REL_SIG_POW],
            use_log_absSigPow=params[pkeys.A7_USE_LOG_ABS_SIG_POW],
            use_log_relSigPow=params[pkeys.A7_USE_LOG_REL_SIG_POW],
            use_log_sigCov=params[pkeys.A7_USE_LOG_SIG_COV],
            use_log_sigCorr=params[pkeys.A7_USE_LOG_SIG_CORR],
            use_zscore_relSigPow=params[pkeys.A7_USE_ZSCORE_REL_SIG_POW],
            use_zscore_sigCov=params[pkeys.A7_USE_ZSCORE_SIG_COV],
            use_zscore_sigCorr=params[pkeys.A7_USE_ZSCORE_SIG_CORR],
            remove_delta_in_cov=params[pkeys.A7_REMOVE_DELTA_IN_COV],
            dispersion_mode=params[pkeys.A7_DISPERSION_MODE],
        )

        # Now is [batch, time_len, 4]
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop, :]

        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # Now pool to get proper length
        outputs = tf.keras.layers.AveragePooling1D(pool_size=8)(outputs)

        # Now recurrent
        lstm_units = params[pkeys.A7_RNN_LSTM_UNITS]
        fc_units = params[pkeys.A7_RNN_FC_UNITS]
        drop_rate_hidden = params[pkeys.A7_RNN_DROP_RATE]

        # Multilayer BLSTM (2 layers)
        outputs = layers.multilayer_lstm_block(
            outputs,
            lstm_units,
            n_layers=2,
            num_dirs=constants.BIDIRECTIONAL,
            dropout_first_lstm=params[pkeys.TYPE_DROPOUT],
            dropout_rest_lstm=params[pkeys.TYPE_DROPOUT],
            drop_rate_first_lstm=0,
            drop_rate_rest_lstm=drop_rate_hidden,
            training=training,
            name="multi_layer_blstm",
        )

        if fc_units > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                fc_units,
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=drop_rate_hidden,
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


def deep_a7_v3(inputs, params, training, name="model_a7_v3"):
    print("Using model A7_V3 (A7 feats input, RED architecture)")
    with tf.variable_scope(name):
        # input is [batch, time_len]
        inputs = a7_layer_tf(
            inputs,
            fs=params[pkeys.FS],
            window_duration=params[pkeys.A7_WINDOW_DURATION],
            window_duration_relSigPow=params[pkeys.A7_WINDOW_DURATION_REL_SIG_POW],
            use_log_absSigPow=params[pkeys.A7_USE_LOG_ABS_SIG_POW],
            use_log_relSigPow=params[pkeys.A7_USE_LOG_REL_SIG_POW],
            use_log_sigCov=params[pkeys.A7_USE_LOG_SIG_COV],
            use_log_sigCorr=params[pkeys.A7_USE_LOG_SIG_CORR],
            use_zscore_relSigPow=params[pkeys.A7_USE_ZSCORE_REL_SIG_POW],
            use_zscore_sigCov=params[pkeys.A7_USE_ZSCORE_SIG_COV],
            use_zscore_sigCorr=params[pkeys.A7_USE_ZSCORE_SIG_CORR],
            remove_delta_in_cov=params[pkeys.A7_REMOVE_DELTA_IN_COV],
            dispersion_mode=params[pkeys.A7_DISPERSION_MODE],
        )

        # Now is [batch, time_len, 4]
        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS])
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        inputs = inputs[:, start_crop:end_crop, :]

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


def wavelet_blstm_net_v11_bp(inputs, params, training, name="model_v11_bp"):
    print("Using model V11-BP (Time-Domain on band-passed signal)")
    with tf.variable_scope(name):
        # Band-pass
        print(
            "Applying bandpass between %s Hz and %s Hz"
            % (params[pkeys.BP_INPUT_LOWCUT], params[pkeys.BP_INPUT_HIGHCUT])
        )
        inputs = bandpass_tf_batch(
            inputs,
            fs=params[pkeys.FS],
            lowcut=params[pkeys.BP_INPUT_LOWCUT],
            highcut=params[pkeys.BP_INPUT_HIGHCUT],
        )

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


def wavelet_blstm_net_v19_bp(inputs, params, training, name="model_v19_bp"):
    print("Using model V19-BP (general cwt on band-passed signal)")
    with tf.variable_scope(name):
        # Band-pass
        print(
            "Applying bandpass between %s Hz and %s Hz"
            % (params[pkeys.BP_INPUT_LOWCUT], params[pkeys.BP_INPUT_HIGHCUT])
        )
        inputs = bandpass_tf_batch(
            inputs,
            fs=params[pkeys.FS],
            lowcut=params[pkeys.BP_INPUT_LOWCUT],
            highcut=params[pkeys.BP_INPUT_HIGHCUT],
        )

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

        return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v11_ln(inputs, params, training, name="model_v11_ln"):
    print("Using model V11-LN (Time-Domain with zscore at conv)")
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

        outputs = layers.conv1d_prebn_block_with_zscore(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3_zscore",
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


def wavelet_blstm_net_v11_ln2(inputs, params, training, name="model_v11_ln2"):
    print("Using model V11-LN2 (Time-Domain with zscore at logits)")
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
            outputs = layers.sequence_fc_layer_with_zscore(
                outputs,
                params[pkeys.FC_UNITS],
                training=training,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                kernel_init=tf.initializers.he_normal(),
                name="fc_1_zscore",
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


def wavelet_blstm_net_v19_ln2(inputs, params, training, name="model_v19_ln2"):
    print("Using model V19-LN2 (general cwt with zscore at logits)")
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
            outputs = layers.sequence_fc_layer_with_zscore(
                outputs,
                params[pkeys.FC_UNITS],
                training=training,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                kernel_init=tf.initializers.he_normal(),
                name="fc_1_zscore",
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


def wavelet_blstm_net_v11_ln3(inputs, params, training, name="model_v11_ln3"):
    print("Using model V11-LN3 (Time-Domain with zscore at logits and last conv)")
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

        outputs = layers.conv1d_prebn_block_with_zscore(
            outputs,
            params[pkeys.TIME_CONV_FILTERS_3],
            training,
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            downsampling=params[pkeys.CONV_DOWNSAMPLING],
            kernel_init=tf.initializers.he_normal(),
            name="convblock_1d_3_zscore",
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
            outputs = layers.sequence_fc_layer_with_zscore(
                outputs,
                params[pkeys.FC_UNITS],
                training=training,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                kernel_init=tf.initializers.he_normal(),
                name="fc_1_zscore",
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


def wavelet_blstm_net_v11_mk(inputs, params, training, name="model_v11_mk"):
    print("Using model V11-MK (Time-Domain with multi-kernel convolutions)")
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
        drop_rate_conv = params[pkeys.TIME_CONV_MK_DROP_RATE]
        drop_rate_conv = 0 if (drop_rate_conv is None) else drop_rate_conv
        drop_conv = params[pkeys.TYPE_DROPOUT] if (drop_rate_conv > 0) else None

        print("Conv dropout type %s and rate %s" % (drop_conv, drop_rate_conv))
        print("Projection first flag: %s" % params[pkeys.TIME_CONV_MK_PROJECT_FIRST])

        print("First convolutional block")
        tmp_out_list = []
        for kernel_size, n_filters in params[pkeys.TIME_CONV_MK_FILTERS_1]:
            print("    k %d and f %d" % (kernel_size, n_filters))
            tmp_out = layers.conv1d_prebn_block_with_projection(
                outputs,
                n_filters,
                training,
                kernel_size=kernel_size,
                project_first=params[pkeys.TIME_CONV_MK_PROJECT_FIRST],
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                dropout=drop_conv,
                drop_rate=drop_rate_conv,
                kernel_init=tf.initializers.he_normal(),
                name="convblock_1d_k%d_1" % kernel_size,
            )
            tmp_out_list.append(tmp_out)
        outputs_1 = tf.concat(tmp_out_list, axis=-1)

        print("Second convolutional block")
        tmp_out_list = []
        for kernel_size, n_filters in params[pkeys.TIME_CONV_MK_FILTERS_2]:
            print("    k %d and f %d" % (kernel_size, n_filters))
            tmp_out = layers.conv1d_prebn_block_with_projection(
                outputs_1,
                n_filters,
                training,
                kernel_size=kernel_size,
                project_first=params[pkeys.TIME_CONV_MK_PROJECT_FIRST],
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                dropout=drop_conv,
                drop_rate=drop_rate_conv,
                kernel_init=tf.initializers.he_normal(),
                name="convblock_1d_k%d_2" % kernel_size,
            )
            tmp_out_list.append(tmp_out)
        outputs_2 = tf.concat(tmp_out_list, axis=-1)

        print("Third convolutional block")
        tmp_out_list = []
        for kernel_size, n_filters in params[pkeys.TIME_CONV_MK_FILTERS_3]:
            print("    k %d and f %d" % (kernel_size, n_filters))
            tmp_out = layers.conv1d_prebn_block_with_projection(
                outputs_2,
                n_filters,
                training,
                kernel_size=kernel_size,
                project_first=params[pkeys.TIME_CONV_MK_PROJECT_FIRST],
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                dropout=drop_conv,
                drop_rate=drop_rate_conv,
                kernel_init=tf.initializers.he_normal(),
                name="convblock_1d_k%d_3" % kernel_size,
            )
            tmp_out_list.append(tmp_out)
        outputs_3 = tf.concat(tmp_out_list, axis=-1)

        if params[pkeys.TIME_CONV_MK_SKIPS]:
            print("Passing feature pyramid to LSTM")
            # outputs_1 needs 2 additional pooling
            outputs_1 = tf.expand_dims(outputs_1, axis=2)
            outputs_1 = tf.layers.average_pooling2d(
                inputs=outputs_1, pool_size=(4, 1), strides=(4, 1)
            )
            outputs_1 = tf.squeeze(outputs_1, axis=2, name="squeeze")
            # outputs_2 needs 1 additional pooling
            outputs_2 = tf.expand_dims(outputs_2, axis=2)
            outputs_2 = tf.layers.average_pooling2d(
                inputs=outputs_2, pool_size=(2, 1), strides=(2, 1)
            )
            outputs_2 = tf.squeeze(outputs_2, axis=2, name="squeeze")
            # Concat each block for multi-scale features
            outputs = tf.concat([outputs_1, outputs_2, outputs_3], axis=-1)
        else:
            print("Passing last output to LSTM")
            # Just the last output
            outputs = outputs_3

        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS] // 8)
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        outputs = outputs[:, start_crop:end_crop]

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


def wavelet_blstm_net_v11_mkd(inputs, params, training, name="model_v11_mkd"):
    print("Using model V11-MKD (Time-Domain with multi-dilated convolutions)")
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
        drop_rate_conv = params[pkeys.TIME_CONV_MK_DROP_RATE]
        drop_rate_conv = 0 if (drop_rate_conv is None) else drop_rate_conv
        drop_conv = params[pkeys.TYPE_DROPOUT] if (drop_rate_conv > 0) else None

        print("Conv dropout type %s and rate %s" % (drop_conv, drop_rate_conv))
        print("Projection first flag: %s" % params[pkeys.TIME_CONV_MK_PROJECT_FIRST])

        print("First convolutional block")
        tmp_out_list = []
        for kernel_size, n_filters, dilation in params[pkeys.TIME_CONV_MKD_FILTERS_1]:
            print("    k %d, f %d and d %d" % (kernel_size, n_filters, dilation))
            tmp_out = layers.conv1d_prebn_block_with_dilation(
                outputs,
                n_filters,
                training,
                kernel_size=kernel_size,
                dilation=dilation,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                dropout=drop_conv,
                drop_rate=drop_rate_conv,
                kernel_init=tf.initializers.he_normal(),
                name="convblock_1d_k%d_d%d_1" % (kernel_size, dilation),
            )
            tmp_out_list.append(tmp_out)
        outputs_1 = tf.concat(tmp_out_list, axis=-1)

        print("Second convolutional block")
        tmp_out_list = []
        for kernel_size, n_filters, dilation in params[pkeys.TIME_CONV_MKD_FILTERS_2]:
            print("    k %d, f %d and d %d" % (kernel_size, n_filters, dilation))
            tmp_out = layers.conv1d_prebn_block_with_dilation(
                outputs_1,
                n_filters,
                training,
                kernel_size=kernel_size,
                dilation=dilation,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                dropout=drop_conv,
                drop_rate=drop_rate_conv,
                kernel_init=tf.initializers.he_normal(),
                name="convblock_1d_k%d_d%d_2" % (kernel_size, dilation),
            )
            tmp_out_list.append(tmp_out)
        outputs_2 = tf.concat(tmp_out_list, axis=-1)

        print("Third convolutional block")
        tmp_out_list = []
        for kernel_size, n_filters, dilation in params[pkeys.TIME_CONV_MKD_FILTERS_3]:
            print("    k %d, f %d and d %d" % (kernel_size, n_filters, dilation))
            tmp_out = layers.conv1d_prebn_block_with_dilation(
                outputs_2,
                n_filters,
                training,
                kernel_size=kernel_size,
                dilation=dilation,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                dropout=drop_conv,
                drop_rate=drop_rate_conv,
                kernel_init=tf.initializers.he_normal(),
                name="convblock_1d_k%d_d%d_3" % (kernel_size, dilation),
            )
            tmp_out_list.append(tmp_out)
        outputs_3 = tf.concat(tmp_out_list, axis=-1)

        if params[pkeys.TIME_CONV_MK_SKIPS]:
            print("Passing feature pyramid to LSTM")
            # outputs_1 needs 2 additional pooling
            outputs_1 = tf.expand_dims(outputs_1, axis=2)
            outputs_1 = tf.layers.average_pooling2d(
                inputs=outputs_1, pool_size=(4, 1), strides=(4, 1)
            )
            outputs_1 = tf.squeeze(outputs_1, axis=2, name="squeeze")
            # outputs_2 needs 1 additional pooling
            outputs_2 = tf.expand_dims(outputs_2, axis=2)
            outputs_2 = tf.layers.average_pooling2d(
                inputs=outputs_2, pool_size=(2, 1), strides=(2, 1)
            )
            outputs_2 = tf.squeeze(outputs_2, axis=2, name="squeeze")
            # Concat each block for multi-scale features
            outputs = tf.concat([outputs_1, outputs_2, outputs_3], axis=-1)
        else:
            print("Passing last output to LSTM")
            # Just the last output
            outputs = outputs_3

        border_crop = int(params[pkeys.BORDER_DURATION] * params[pkeys.FS] // 8)
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        outputs = outputs[:, start_crop:end_crop]

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


def wavelet_blstm_net_v11_mkd2(inputs, params, training, name="model_v11_mkd2"):
    print(
        "Using model V11-MKD-2 (Time-Domain with multi-dilated convolutions, border crop AFTER lstm)"
    )
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
        drop_rate_conv = params[pkeys.TIME_CONV_MK_DROP_RATE]
        drop_rate_conv = 0 if (drop_rate_conv is None) else drop_rate_conv
        drop_conv = params[pkeys.TYPE_DROPOUT] if (drop_rate_conv > 0) else None

        print("Conv dropout type %s and rate %s" % (drop_conv, drop_rate_conv))
        print("Projection first flag: %s" % params[pkeys.TIME_CONV_MK_PROJECT_FIRST])

        print("First convolutional block")
        tmp_out_list = []
        for kernel_size, n_filters, dilation in params[pkeys.TIME_CONV_MKD_FILTERS_1]:
            print("    k %d, f %d and d %d" % (kernel_size, n_filters, dilation))
            tmp_out = layers.conv1d_prebn_block_with_dilation(
                outputs,
                n_filters,
                training,
                kernel_size=kernel_size,
                dilation=dilation,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                dropout=drop_conv,
                drop_rate=drop_rate_conv,
                kernel_init=tf.initializers.he_normal(),
                name="convblock_1d_k%d_d%d_1" % (kernel_size, dilation),
            )
            tmp_out_list.append(tmp_out)
        outputs_1 = tf.concat(tmp_out_list, axis=-1)

        print("Second convolutional block")
        tmp_out_list = []
        for kernel_size, n_filters, dilation in params[pkeys.TIME_CONV_MKD_FILTERS_2]:
            print("    k %d, f %d and d %d" % (kernel_size, n_filters, dilation))
            tmp_out = layers.conv1d_prebn_block_with_dilation(
                outputs_1,
                n_filters,
                training,
                kernel_size=kernel_size,
                dilation=dilation,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                dropout=drop_conv,
                drop_rate=drop_rate_conv,
                kernel_init=tf.initializers.he_normal(),
                name="convblock_1d_k%d_d%d_2" % (kernel_size, dilation),
            )
            tmp_out_list.append(tmp_out)
        outputs_2 = tf.concat(tmp_out_list, axis=-1)

        print("Third convolutional block")
        tmp_out_list = []
        for kernel_size, n_filters, dilation in params[pkeys.TIME_CONV_MKD_FILTERS_3]:
            print("    k %d, f %d and d %d" % (kernel_size, n_filters, dilation))
            tmp_out = layers.conv1d_prebn_block_with_dilation(
                outputs_2,
                n_filters,
                training,
                kernel_size=kernel_size,
                dilation=dilation,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                dropout=drop_conv,
                drop_rate=drop_rate_conv,
                kernel_init=tf.initializers.he_normal(),
                name="convblock_1d_k%d_d%d_3" % (kernel_size, dilation),
            )
            tmp_out_list.append(tmp_out)
        outputs_3 = tf.concat(tmp_out_list, axis=-1)

        if params[pkeys.TIME_CONV_MK_SKIPS]:
            print("Passing feature pyramid to LSTM")
            # outputs_1 needs 2 additional pooling
            outputs_1 = tf.expand_dims(outputs_1, axis=2)
            outputs_1 = tf.layers.average_pooling2d(
                inputs=outputs_1, pool_size=(4, 1), strides=(4, 1)
            )
            outputs_1 = tf.squeeze(outputs_1, axis=2, name="squeeze")
            # outputs_2 needs 1 additional pooling
            outputs_2 = tf.expand_dims(outputs_2, axis=2)
            outputs_2 = tf.layers.average_pooling2d(
                inputs=outputs_2, pool_size=(2, 1), strides=(2, 1)
            )
            outputs_2 = tf.squeeze(outputs_2, axis=2, name="squeeze")
            # Concat each block for multi-scale features
            outputs = tf.concat([outputs_1, outputs_2, outputs_3], axis=-1)
        else:
            print("Passing last output to LSTM")
            # Just the last output
            outputs = outputs_3

        border_duration_to_crop_after_conv = 1
        border_duration_to_crop_after_lstm = (
            params[pkeys.BORDER_DURATION] - border_duration_to_crop_after_conv
        )

        border_crop = int(border_duration_to_crop_after_conv * params[pkeys.FS] // 8)
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        outputs = outputs[:, start_crop:end_crop]

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

        # Now crop the rest
        border_crop = int(border_duration_to_crop_after_lstm * params[pkeys.FS] // 8)
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        outputs = outputs[:, start_crop:end_crop]

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


def stat_net_conv(inputs_normalized, params, training, name="stat_net_conv"):
    with tf.variable_scope(name):
        n_layers = params[pkeys.STAT_NET_CONV_DEPTH]
        kernel_size = params[pkeys.STAT_NET_CONV_KERNEL_SIZE]
        type_pool = params[pkeys.STAT_NET_CONV_TYPE_POOL]
        init_filters = params[pkeys.STAT_NET_CONV_INITIAL_FILTERS]
        max_filters = params[pkeys.STAT_NET_CONV_MAX_FILTERS]

        batchnorm = params[pkeys.TYPE_BATCHNORM]

        # [batch_size, time_len, n_feats] -> [batch_size, time_len, 1, feats]
        outputs = tf.expand_dims(inputs_normalized, axis=2)
        use_bias = batchnorm is None

        for i in range(1, n_layers + 1):
            with tf.variable_scope("conv%d_%d" % (kernel_size, i)):
                filters = init_filters * (2 ** (i - 1))
                filters = min(filters, max_filters)
                outputs = tf.layers.conv2d(
                    inputs=outputs,
                    filters=filters,
                    kernel_size=(kernel_size, 1),
                    padding=constants.PAD_VALID,
                    name="conv%d" % kernel_size,
                    kernel_initializer=tf.initializers.he_normal(),
                    use_bias=use_bias,
                )
                if batchnorm:
                    outputs = layers.batchnorm_layer(
                        outputs,
                        "bn",
                        batchnorm=batchnorm,
                        training=training,
                        scale=False,
                    )
                outputs = tf.nn.relu(outputs)
                outputs = layers.pooling1d(outputs, type_pool)

        # [batch_size, time_len, 1, n_units] -> [batch_size, time_len, n_units]
        outputs = tf.squeeze(outputs, axis=2, name="squeeze")
    return outputs


def stat_net_lstm(inputs_normalized, params, training, name="stat_net_lstm"):
    with tf.variable_scope(name):
        outputs = layers.lstm_layer(
            inputs_normalized,
            num_units=params[pkeys.STAT_NET_LSTM_UNITS],
            num_dirs=constants.BIDIRECTIONAL,
            training=training,
            name="blstm",
        )
    return outputs


def stat_net(
    inputs_normalized, params, training, output_activation=tf.nn.relu, name="stat_net"
):
    with tf.variable_scope(name):
        # Select backbone
        backbone = params[pkeys.STAT_NET_TYPE_BACKBONE]
        if backbone == "conv":
            outputs = stat_net_conv(inputs_normalized, params, training)
        elif backbone == "lstm":
            outputs = stat_net_lstm(inputs_normalized, params, training)
        else:
            raise ValueError("%s not a valid backbone type" % backbone)

        # Select collapse function
        type_collapse = params[pkeys.STAT_NET_TYPE_COLLAPSE]
        if type_collapse == "average":
            outputs = tf.keras.layers.GlobalAvgPool1D(name="average")(outputs)
        elif type_collapse in ["softmax", "sigmoid"]:
            # [batch_size, time_len, n_feats] -> [batch_size, time_len, 1, feats]
            outputs = tf.expand_dims(outputs, axis=2)
            # Predict scores for each time step
            # This is a score tensor of shape [batch_size, time_len, 1, 1]
            scores = tf.layers.conv2d(
                inputs=outputs,
                filters=1,
                kernel_size=(1, 1),
                padding=constants.PAD_VALID,
                name="scores",
                kernel_initializer=tf.initializers.he_normal(),
            )
            # Normalize scores to sum 1
            with tf.variable_scope("scores_normalization"):
                if type_collapse == "sigmoid":
                    sigmoid_scores = tf.math.sigmoid(scores)
                    normalization_factor = (
                        tf.reduce_sum(sigmoid_scores, axis=1, keepdims=True) + 1e-8
                    )
                    normalized_scores = sigmoid_scores / normalization_factor
                else:
                    normalized_scores = tf.nn.softmax(scores, axis=1)
            # Compute weighted global average
            outputs = tf.reduce_sum(normalized_scores * outputs, axis=[1, 2])
        else:
            raise ValueError("%s not a valid type_collapse" % type_collapse)
        # output now should be [batch_size, n_units]

        # Do the rest
        drop_rate = params[pkeys.STAT_NET_CONTEXT_DROP_RATE]
        context_dim = params[pkeys.STAT_NET_CONTEXT_DIM]
        outputs = tf.layers.dropout(outputs, training=training, rate=drop_rate)
        outputs = tf.keras.layers.Dense(
            context_dim,
            kernel_initializer=tf.initializers.he_normal(),
            activation=output_activation,
        )(outputs)
        # output is [batch_size, n_units]
    return outputs


def stat_mod_net(
    inputs_normalized,
    output_size,
    params,
    training,
    scale_base_value=0.0,
    bias_base_value=0.0,
    name="stat_mod_net",
):
    print("Using Mod Net")
    modulation_scale = scale_base_value
    modulation_bias = bias_base_value
    with tf.variable_scope(name):
        # output is [batch_size, n_units]
        outputs = stat_net(inputs_normalized, params, training)
        # [batch, n_feats] -> [batch, 1, n_feats]
        outputs = tf.expand_dims(outputs, axis=1)
        # Scale and bias
        modulation_scale += layers.sequence_fc_layer(
            outputs,
            output_size,
            kernel_init=tf.initializers.he_normal(),
            training=training,
            use_bias=params[pkeys.STAT_MOD_NET_BIASED_SCALE],
            name="mod_scale",
        )
        if params[pkeys.STAT_MOD_NET_USE_BIAS]:
            modulation_bias += layers.sequence_fc_layer(
                outputs,
                output_size,
                kernel_init=tf.initializers.he_normal(),
                training=training,
                use_bias=params[pkeys.STAT_MOD_NET_BIASED_BIAS],
                name="mod_bias",
            )
        # output has shape [batch, time_len, dim]
    return modulation_scale, modulation_bias


def stat_dot_net(inputs_normalized, output_size, params, training, name="stat_dot_net"):
    print("Using Dot Net")
    with tf.variable_scope(name):
        # output is [batch_size, n_units]
        outputs = stat_net(inputs_normalized, params, training)
        # [batch, n_feats] -> [batch, 1, n_feats]
        outputs = tf.expand_dims(outputs, axis=1)
        # kernel and bias
        dot_kernel = layers.sequence_fc_layer(
            outputs,
            output_size,
            kernel_init=tf.initializers.he_normal(),
            training=training,
            use_bias=params[pkeys.STAT_DOT_NET_BIASED_KERNEL],
            name="dot_kernel",
        )
        if params[pkeys.STAT_DOT_NET_USE_BIAS]:
            dot_bias = layers.sequence_fc_layer(
                outputs,
                1,
                kernel_init=tf.initializers.he_normal(),
                training=training,
                use_bias=params[pkeys.STAT_DOT_NET_BIASED_BIAS],
                name="dot_bias",
            )
        else:
            dot_bias = 0.0
        # output has shape [batch, time_len, dim]
    return dot_kernel, dot_bias


def segment_net(
    inputs_normalized,
    params,
    training,
    output_activation=tf.nn.relu,
    border_conv=1,
    border_lstm=5,
    return_blstm_output=False,
    name="segment_net",
):
    print("Using V11-MKD2 as segment network")
    with tf.variable_scope(name):
        # 1D convolutions expect shape [batch, time_len, n_feats]
        outputs = inputs_normalized

        # Only keep border_conv + border_lstm
        border_duration_to_keep = border_conv + border_lstm
        border_duration_to_crop = (
            params[pkeys.BORDER_DURATION] - border_duration_to_keep
        )
        border_crop = int(border_duration_to_crop * params[pkeys.FS])
        start_crop = border_crop
        end_crop = None if (border_crop <= 0) else -border_crop
        outputs = outputs[:, start_crop:end_crop]

        # Convolutional stage (standard feed-forward)
        drop_rate_conv = params[pkeys.TIME_CONV_MK_DROP_RATE]
        drop_rate_conv = 0 if (drop_rate_conv is None) else drop_rate_conv
        drop_conv = params[pkeys.TYPE_DROPOUT] if (drop_rate_conv > 0) else None

        print("Conv dropout type %s and rate %s" % (drop_conv, drop_rate_conv))
        print("Projection first flag: %s" % params[pkeys.TIME_CONV_MK_PROJECT_FIRST])

        print("First convolutional block")
        tmp_out_list = []
        for kernel_size, n_filters, dilation in params[pkeys.TIME_CONV_MKD_FILTERS_1]:
            print("    k %d, f %d and d %d" % (kernel_size, n_filters, dilation))
            tmp_out = layers.conv1d_prebn_block_with_dilation(
                outputs,
                n_filters,
                training,
                kernel_size=kernel_size,
                dilation=dilation,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                dropout=drop_conv,
                drop_rate=drop_rate_conv,
                kernel_init=tf.initializers.he_normal(),
                name="convblock_1d_k%d_d%d_1" % (kernel_size, dilation),
            )
            tmp_out_list.append(tmp_out)
        outputs_1 = tf.concat(tmp_out_list, axis=-1)

        print("Second convolutional block")
        tmp_out_list = []
        for kernel_size, n_filters, dilation in params[pkeys.TIME_CONV_MKD_FILTERS_2]:
            print("    k %d, f %d and d %d" % (kernel_size, n_filters, dilation))
            tmp_out = layers.conv1d_prebn_block_with_dilation(
                outputs_1,
                n_filters,
                training,
                kernel_size=kernel_size,
                dilation=dilation,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                dropout=drop_conv,
                drop_rate=drop_rate_conv,
                kernel_init=tf.initializers.he_normal(),
                name="convblock_1d_k%d_d%d_2" % (kernel_size, dilation),
            )
            tmp_out_list.append(tmp_out)
        outputs_2 = tf.concat(tmp_out_list, axis=-1)

        print("Third convolutional block")
        tmp_out_list = []
        for kernel_size, n_filters, dilation in params[pkeys.TIME_CONV_MKD_FILTERS_3]:
            print("    k %d, f %d and d %d" % (kernel_size, n_filters, dilation))
            tmp_out = layers.conv1d_prebn_block_with_dilation(
                outputs_2,
                n_filters,
                training,
                kernel_size=kernel_size,
                dilation=dilation,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                dropout=drop_conv,
                drop_rate=drop_rate_conv,
                kernel_init=tf.initializers.he_normal(),
                name="convblock_1d_k%d_d%d_3" % (kernel_size, dilation),
            )
            tmp_out_list.append(tmp_out)
        outputs_3 = tf.concat(tmp_out_list, axis=-1)

        if params[pkeys.TIME_CONV_MK_SKIPS]:
            print("Passing feature pyramid to LSTM")
            # outputs_1 needs 2 additional pooling
            outputs_1 = tf.expand_dims(outputs_1, axis=2)
            outputs_1 = tf.layers.average_pooling2d(
                inputs=outputs_1, pool_size=(4, 1), strides=(4, 1)
            )
            outputs_1 = tf.squeeze(outputs_1, axis=2, name="squeeze")
            # outputs_2 needs 1 additional pooling
            outputs_2 = tf.expand_dims(outputs_2, axis=2)
            outputs_2 = tf.layers.average_pooling2d(
                inputs=outputs_2, pool_size=(2, 1), strides=(2, 1)
            )
            outputs_2 = tf.squeeze(outputs_2, axis=2, name="squeeze")
            # Concat each block for multi-scale features
            outputs = tf.concat([outputs_1, outputs_2, outputs_3], axis=-1)
        else:
            print("Passing last output to LSTM")
            # Just the last output
            outputs = outputs_3

        # Only keep border_lstm
        border_duration_to_crop = border_duration_to_keep - border_lstm
        border_crop = int(border_duration_to_crop * params[pkeys.FS] // 8)
        start_crop = border_crop
        end_crop = None if (border_crop <= 0) else -border_crop
        outputs = outputs[:, start_crop:end_crop]

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

        # Now crop the rest
        border_crop = int(border_lstm * params[pkeys.FS] // 8)
        start_crop = border_crop
        end_crop = None if (border_crop <= 0) else -border_crop
        outputs = outputs[:, start_crop:end_crop]

        if return_blstm_output:
            return outputs

        outputs = layers.sequence_fc_layer(
            outputs,
            params[pkeys.FC_UNITS],
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            activation=output_activation,
            name="fc",
        )
    return outputs


def wavelet_blstm_net_v11_mkd2_statmod(
    inputs, params, training, name="model_v11_mkd2_statmod"
):
    print(
        "Using model V11-MKD-2-STAT-MOD"
        "(Time-Domain with multi-dilated convolutions, border crop AFTER lstm, stat net modulation)"
    )
    with tf.variable_scope(name):
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)
        # BN at input
        inputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )
        outputs = segment_net(inputs, params, training, output_activation=None)
        if params[pkeys.STAT_MOD_NET_MODULATE_LOGITS]:
            print("Modulating logits")
            outputs = tf.nn.relu(outputs)
            logits = layers.sequence_output_2class_layer(
                outputs,
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_OUTPUT],
                training=training,
                init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
                name="logits",
            )
            mod_scale, mod_bias = stat_mod_net(inputs, 2, params, training)
            logits = mod_scale * logits + mod_bias
        else:
            print("Modulating last hidden layer before ReLU")
            mod_scale, mod_bias = stat_mod_net(
                inputs, params[pkeys.FC_UNITS], params, training
            )
            outputs = mod_scale * outputs + mod_bias
            outputs = tf.nn.relu(outputs)
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


def wavelet_blstm_net_v11_mkd2_statdot(
    inputs, params, training, name="model_v11_mkd2_statdot"
):
    print(
        "Using model V11-MKD-2-STAT-DOT"
        "(Time-Domain with multi-dilated convolutions, border crop AFTER lstm, stat net dot-product class scores)"
    )
    with tf.variable_scope(name):
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)
        # BN at input
        inputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )
        outputs = segment_net(inputs, params, training, output_activation=tf.nn.relu)
        segment_dim = outputs.get_shape().as_list()[-1]
        product_dim = params[pkeys.STAT_DOT_NET_PRODUCT_DIM]
        if product_dim != segment_dim:
            print(
                "Applying linear projection to match %d output to %d product"
                % (segment_dim, product_dim)
            )
            outputs = layers.sequence_fc_layer(
                outputs,
                product_dim,
                kernel_init=tf.initializers.he_normal(),
                training=training,
                use_bias=False,
                name="linear_projection",
            )
        dot_kernel, dot_bias = stat_dot_net(inputs, product_dim, params, training)
        # output and kernel is [batch, time_len, product_dim]
        outputs_positive = (
            tf.reduce_sum(dot_kernel * outputs, axis=-1, keepdims=True) + dot_bias
        )
        # output is [batch, time_len, 1]
        outputs_negative = tf.zeros_like(
            outputs_positive
        )  # Dummy logit for compatibility
        logits = tf.concat([outputs_negative, outputs_positive], axis=-1, name="logits")

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
    return logits, probabilities, cwt_prebn


def wavelet_blstm_net_v36(
    inputs, params, training, border_conv=1, border_lstm=5, name="model_v36"
):
    print("Using V36 (bandpass with independent branches)")
    with tf.variable_scope(name):
        # Band-pass signals
        # Function expects [batch, time_len]
        bands = layers.signal_decomposition_bandpass(inputs, params[pkeys.FS], "bands")

        # Independent branches
        key_list = ["16-32Hz", "8-16Hz", "4-8Hz", "0-4Hz"]
        use_dilation = params[pkeys.DECOMP_BP_USE_DILATION]
        tmp_band_outputs = []
        for i, key in enumerate(key_list):
            dilation = 2**i if use_dilation else 1
            print("Branch for %s band using dilation %d" % (key, dilation))
            band_inputs = bands[key]
            # Transform [batch, time_len] -> [batch, time_len, 1]
            band_inputs = tf.expand_dims(band_inputs, axis=2)
            # Only keep border_conv + border_lstm
            border_duration_to_keep = border_conv + border_lstm
            border_duration_to_crop = (
                params[pkeys.BORDER_DURATION] - border_duration_to_keep
            )
            border_crop = int(border_duration_to_crop * params[pkeys.FS])
            start_crop = border_crop
            end_crop = None if (border_crop <= 0) else -border_crop
            band_inputs = band_inputs[:, start_crop:end_crop]
            # BN at input
            band_outputs = layers.batchnorm_layer(
                band_inputs,
                "%s_bn_input" % key,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                training=training,
            )
            # Convolutional stage (standard feed-forward)
            # 1D convolutions expect shape [batch, time_len, n_feats]
            band_outputs = layers.conv1d_prebn_block_with_dilation(
                band_outputs,
                params[pkeys.DECOMP_BP_INITIAL_FILTERS],
                training,
                kernel_size=3,
                dilation=dilation,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                kernel_init=tf.initializers.he_normal(),
                name="%s_convblock_1d_d%d_1" % (key, dilation),
            )
            band_outputs = layers.conv1d_prebn_block_with_dilation(
                band_outputs,
                params[pkeys.DECOMP_BP_INITIAL_FILTERS] * 2,
                training,
                kernel_size=3,
                dilation=dilation,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                kernel_init=tf.initializers.he_normal(),
                name="%s_convblock_1d_d%d_2" % (key, dilation),
            )
            band_outputs = layers.conv1d_prebn_block_with_dilation(
                band_outputs,
                params[pkeys.DECOMP_BP_INITIAL_FILTERS] * 4,
                training,
                kernel_size=3,
                dilation=dilation,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                kernel_init=tf.initializers.he_normal(),
                name="%s_convblock_1d_d%d_3" % (key, dilation),
            )
            tmp_band_outputs.append(band_outputs)
        outputs = tf.concat(tmp_band_outputs, axis=-1)

        if params[pkeys.DECOMP_BP_EXTRA_CONV_FILTERS] > 0:
            with tf.variable_scope("extra_conv"):
                batchnorm = params[pkeys.TYPE_BATCHNORM]
                kernel_size = 1
                # [batch_size, time_len, n_feats] -> [batch_size, time_len, 1, feats]
                outputs = tf.expand_dims(outputs, axis=2)
                use_bias = batchnorm is None
                outputs = tf.layers.conv2d(
                    inputs=outputs,
                    filters=params[pkeys.DECOMP_BP_EXTRA_CONV_FILTERS],
                    kernel_size=(kernel_size, 1),
                    padding=constants.PAD_SAME,
                    dilation_rate=1,
                    strides=1,
                    name="conv%d" % kernel_size,
                    reuse=False,
                    kernel_initializer=tf.initializers.he_normal(),
                    use_bias=use_bias,
                )
                if batchnorm:
                    outputs = layers.batchnorm_layer(
                        outputs,
                        "bn",
                        batchnorm=batchnorm,
                        reuse=False,
                        training=training,
                        scale=False,
                    )
                outputs = tf.nn.relu(outputs)
                # [batch_size, time_len, 1, n_units] -> [batch_size, time_len, n_units]
                outputs = tf.squeeze(outputs, axis=2, name="squeeze")

        # Only keep border_lstm
        border_duration_to_crop = border_duration_to_keep - border_lstm
        border_crop = int(border_duration_to_crop * params[pkeys.FS] // 8)
        start_crop = border_crop
        end_crop = None if (border_crop <= 0) else -border_crop
        outputs = outputs[:, start_crop:end_crop]

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

        # Now crop the rest
        border_crop = int(border_lstm * params[pkeys.FS] // 8)
        start_crop = border_crop
        end_crop = None if (border_crop <= 0) else -border_crop
        outputs = outputs[:, start_crop:end_crop]

        outputs = layers.sequence_fc_layer(
            outputs,
            params[pkeys.FC_UNITS],
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            activation=tf.nn.relu,
            name="fc",
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


def wavelet_blstm_net_v11_att(
    inputs, params, training, border_conv=1, border_lstm=5, name="v11_att"
):
    print("Using V11-MKD2 but with attention after BLSTM")
    with tf.variable_scope(name):
        kernel_init = tf.initializers.he_normal()

        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)
        # Only keep border_conv + border_lstm
        border_duration_to_keep = border_conv + border_lstm
        border_duration_to_crop = (
            params[pkeys.BORDER_DURATION] - border_duration_to_keep
        )
        border_crop = int(border_duration_to_crop * params[pkeys.FS])
        start_crop = border_crop
        end_crop = None if (border_crop <= 0) else -border_crop
        inputs = inputs[:, start_crop:end_crop]
        # BN at input
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # Convolutional stage (standard feed-forward)
        drop_rate_conv = params[pkeys.TIME_CONV_MK_DROP_RATE]
        drop_rate_conv = 0 if (drop_rate_conv is None) else drop_rate_conv
        drop_conv = params[pkeys.TYPE_DROPOUT] if (drop_rate_conv > 0) else None
        print("Conv dropout type %s and rate %s" % (drop_conv, drop_rate_conv))
        print("Projection first flag: %s" % params[pkeys.TIME_CONV_MK_PROJECT_FIRST])
        print("First convolutional block")
        tmp_out_list = []
        for kernel_size, n_filters, dilation in params[pkeys.TIME_CONV_MKD_FILTERS_1]:
            print("    k %d, f %d and d %d" % (kernel_size, n_filters, dilation))
            tmp_out = layers.conv1d_prebn_block_with_dilation(
                outputs,
                n_filters,
                training,
                kernel_size=kernel_size,
                dilation=dilation,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                dropout=drop_conv,
                drop_rate=drop_rate_conv,
                kernel_init=kernel_init,
                name="convblock_1d_k%d_d%d_1" % (kernel_size, dilation),
            )
            tmp_out_list.append(tmp_out)
        outputs_1 = tf.concat(tmp_out_list, axis=-1)
        print("Second convolutional block")
        tmp_out_list = []
        for kernel_size, n_filters, dilation in params[pkeys.TIME_CONV_MKD_FILTERS_2]:
            print("    k %d, f %d and d %d" % (kernel_size, n_filters, dilation))
            tmp_out = layers.conv1d_prebn_block_with_dilation(
                outputs_1,
                n_filters,
                training,
                kernel_size=kernel_size,
                dilation=dilation,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                dropout=drop_conv,
                drop_rate=drop_rate_conv,
                kernel_init=kernel_init,
                name="convblock_1d_k%d_d%d_2" % (kernel_size, dilation),
            )
            tmp_out_list.append(tmp_out)
        outputs_2 = tf.concat(tmp_out_list, axis=-1)
        print("Third convolutional block")
        tmp_out_list = []
        for kernel_size, n_filters, dilation in params[pkeys.TIME_CONV_MKD_FILTERS_3]:
            print("    k %d, f %d and d %d" % (kernel_size, n_filters, dilation))
            tmp_out = layers.conv1d_prebn_block_with_dilation(
                outputs_2,
                n_filters,
                training,
                kernel_size=kernel_size,
                dilation=dilation,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                dropout=drop_conv,
                drop_rate=drop_rate_conv,
                kernel_init=kernel_init,
                name="convblock_1d_k%d_d%d_3" % (kernel_size, dilation),
            )
            tmp_out_list.append(tmp_out)
        outputs_3 = tf.concat(tmp_out_list, axis=-1)

        if params[pkeys.TIME_CONV_MK_SKIPS]:
            print("Passing feature pyramid to LSTM")
            # outputs_1 needs 2 additional pooling
            outputs_1 = tf.expand_dims(outputs_1, axis=2)
            outputs_1 = tf.layers.average_pooling2d(
                inputs=outputs_1, pool_size=(4, 1), strides=(4, 1)
            )
            outputs_1 = tf.squeeze(outputs_1, axis=2, name="squeeze")
            # outputs_2 needs 1 additional pooling
            outputs_2 = tf.expand_dims(outputs_2, axis=2)
            outputs_2 = tf.layers.average_pooling2d(
                inputs=outputs_2, pool_size=(2, 1), strides=(2, 1)
            )
            outputs_2 = tf.squeeze(outputs_2, axis=2, name="squeeze")
            # Concat each block for multi-scale features
            outputs = tf.concat([outputs_1, outputs_2, outputs_3], axis=-1)
        else:
            print("Passing last output to LSTM")
            # Just the last output
            outputs = outputs_3

        # Only keep border_lstm
        border_duration_to_crop = border_duration_to_keep - border_lstm
        border_crop = int(border_duration_to_crop * params[pkeys.FS] // 8)
        start_crop = border_crop
        end_crop = None if (border_crop <= 0) else -border_crop
        outputs = outputs[:, start_crop:end_crop]

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

        # Now crop the rest
        border_crop = int(border_lstm * params[pkeys.FS] // 8)
        start_crop = border_crop
        end_crop = None if (border_crop <= 0) else -border_crop
        outputs = outputs[:, start_crop:end_crop]

        if params[pkeys.ATT_USE_ATTENTION_AFTER_BLSTM]:
            # --------------------------
            # Attention layer
            # input shape: [batch, time_len, n_feats]
            # --------------------------
            original_length = params[pkeys.PAGE_DURATION] * params[pkeys.FS]
            seq_len = int(original_length / params[pkeys.TOTAL_DOWNSAMPLING_FACTOR])
            att_dim = params[pkeys.ATT_DIM]
            att_dr = params[pkeys.ATT_DROP_RATE]

            with tf.variable_scope("attention"):
                # Prepare input
                pos_enc = layers.get_positional_encoding(
                    seq_len, att_dim, params[pkeys.ATT_PE_FACTOR], name="pos_enc"
                )
                pos_enc = tf.expand_dims(pos_enc, axis=0)  # Add batch axis
                outputs = layers.sequence_fc_layer(
                    outputs,
                    att_dim,
                    kernel_init=kernel_init,
                    training=training,
                    name="fc_embed",
                )
                outputs = outputs + pos_enc
                outputs = layers.dropout_layer(
                    outputs,
                    "drop_embed",
                    drop_rate=att_dr,
                    dropout=params[pkeys.TYPE_DROPOUT],
                    training=training,
                )
                # Prepare queries, keys, and values
                queries = layers.sequence_fc_layer(
                    outputs,
                    att_dim,
                    training,
                    kernel_init=kernel_init,
                    use_bias=False,
                    name="queries",
                )
                keys = layers.sequence_fc_layer(
                    outputs,
                    att_dim,
                    training,
                    kernel_init=kernel_init,
                    use_bias=False,
                    name="keys",
                )
                values = layers.sequence_fc_layer(
                    outputs,
                    att_dim,
                    training,
                    kernel_init=kernel_init,
                    use_bias=False,
                    name="values",
                )
                # Compute attention
                outputs = layers.naive_multihead_attention_layer(
                    queries,
                    keys,
                    values,
                    params[pkeys.ATT_N_HEADS],
                    name="multi_head_att",
                )
                # Output is the concatenation of the heads

            if params[pkeys.ATT_USE_EXTRA_FC]:
                outputs = layers.sequence_fc_layer(
                    outputs,
                    2 * params[pkeys.ATT_DIM],
                    training,
                    kernel_init=kernel_init,
                    dropout=params[pkeys.TYPE_DROPOUT],
                    drop_rate=att_dr,
                    activation=tf.nn.relu,
                    name="fc_extra",
                )
        else:
            att_dr = params[pkeys.DROP_RATE_HIDDEN]

        outputs = layers.sequence_fc_layer(
            outputs,
            params[pkeys.FC_UNITS],
            training,
            kernel_init=kernel_init,
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=att_dr,
            activation=tf.nn.relu,
            name="fc_1",
        )

        # Final FC classification layer
        logits = layers.sequence_output_2class_layer(
            outputs,
            training,
            kernel_init=kernel_init,
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            init_positive_proba=params[pkeys.INIT_POSITIVE_PROBA],
            name="logits",
        )

        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
    return logits, probabilities, cwt_prebn


def expert_branch_features(
    inputs_normalized,
    params,
    training,
    border_feats=1,
    batchnorm_use_scale=True,
    batchnorm_use_bias=True,
    trainable_window_duration=True,
    name="expert_branch_features",
):
    with tf.variable_scope(name):
        # Only keep border_feats
        border_duration_to_crop = params[pkeys.BORDER_DURATION] - border_feats
        border_crop = int(border_duration_to_crop * params[pkeys.FS])
        start_crop = border_crop
        end_crop = None if (border_crop <= 0) else -border_crop
        inputs_normalized = inputs_normalized[:, start_crop:end_crop]

        # Input should be [batch_size, time_len]
        time_len = inputs_normalized.get_shape().as_list()[1]
        inputs_normalized = tf.reshape(inputs_normalized, [-1, time_len])

        # Expert features
        outputs = expert_feats.a7_layer_v2_tf(
            inputs_normalized,
            params[pkeys.FS],
            params[pkeys.EXPERT_BRANCH_WINDOW_DURATION],
            sigma_frequencies=(11, 16),
            rel_power_broad_lowcut=params[pkeys.EXPERT_BRANCH_REL_POWER_BROAD_LOWCUT],
            covariance_broad_lowcut=params[pkeys.EXPERT_BRANCH_COVARIANCE_BROAD_LOWCUT],
            abs_power_transformation=params[
                pkeys.EXPERT_BRANCH_ABS_POWER_TRANSFORMATION
            ],
            rel_power_transformation=params[
                pkeys.EXPERT_BRANCH_REL_POWER_TRANSFORMATION
            ],
            covariance_transformation=params[
                pkeys.EXPERT_BRANCH_COVARIANCE_TRANSFORMATION
            ],
            correlation_transformation=params[
                pkeys.EXPERT_BRANCH_CORRELATION_TRANSFORMATION
            ],
            rel_power_use_zscore=params[pkeys.EXPERT_BRANCH_REL_POWER_USE_ZSCORE],
            covariance_use_zscore=params[pkeys.EXPERT_BRANCH_COVARIANCE_USE_ZSCORE],
            correlation_use_zscore=params[pkeys.EXPERT_BRANCH_CORRELATION_USE_ZSCORE],
            zscore_dispersion_mode=params[pkeys.EXPERT_BRANCH_ZSCORE_DISPERSION_MODE],
            trainable_window_duration=trainable_window_duration,
        )
        outputs_to_use = []
        if params[pkeys.EXPERT_BRANCH_USE_ABS_POWER]:
            print("Using abs power")
            outputs_to_use.append(outputs[..., 0])
        if params[pkeys.EXPERT_BRANCH_USE_REL_POWER]:
            print("Using rel power")
            outputs_to_use.append(outputs[..., 1])
        if params[pkeys.EXPERT_BRANCH_USE_COVARIANCE]:
            print("Using covariance")
            outputs_to_use.append(outputs[..., 2])
        if params[pkeys.EXPERT_BRANCH_USE_CORRELATION]:
            print("Using correlation")
            outputs_to_use.append(outputs[..., 3])
        outputs = tf.stack(outputs_to_use, axis=2)
        # output is [batch_size, time_len, n_feats_used]

        # Now crop the rest of the border
        border_crop = int(border_feats * params[pkeys.FS])
        start_crop = border_crop
        end_crop = None if (border_crop <= 0) else -border_crop
        outputs = outputs[:, start_crop:end_crop]

        # Normalize features before time averaging
        outputs = tf.layers.batch_normalization(
            inputs=outputs,
            training=training,
            center=batchnorm_use_bias,
            scale=batchnorm_use_scale,
        )

        # Prepare time axis
        if params[pkeys.EXPERT_BRANCH_COLLAPSE_TIME_MODE] is None:
            print("Time-collapse: None (avgpool of 8)")
            outputs = tf.keras.layers.AveragePooling1D(pool_size=8)(outputs)
        elif params[pkeys.EXPERT_BRANCH_COLLAPSE_TIME_MODE] == "average":
            print("Time-collapse: global avg pool")
            outputs = tf.keras.layers.GlobalAvgPool1D(name="average")(outputs)
            outputs = outputs[:, tf.newaxis, :]
        elif params[pkeys.EXPERT_BRANCH_COLLAPSE_TIME_MODE] == "softmax":
            print("Time-collapse: global softmax pooling")
            # [batch_size, time_len, n_feats] -> [batch_size, time_len, 1, feats]
            outputs = tf.expand_dims(outputs, axis=2)
            # Predict scores for each time step
            # This is a score tensor of shape [batch_size, time_len, 1, 1]
            scores = tf.layers.conv2d(
                inputs=outputs,
                filters=1,
                kernel_size=(1, 1),
                padding=constants.PAD_VALID,
                name="scores",
                kernel_initializer=tf.initializers.he_normal(),
            )
            # Normalize scores to sum 1
            normalized_scores = tf.nn.softmax(scores, axis=1)
            # Compute weighted global average
            outputs = tf.reduce_sum(normalized_scores * outputs, axis=[1, 2])
            outputs = outputs[:, tf.newaxis, :]
        else:
            raise ValueError(
                "%s not a valid time collapse"
                % params[pkeys.EXPERT_BRANCH_COLLAPSE_TIME_MODE]
            )
        # outputs is [batch_size, ?, n_feats], with ? either time/8 or 1.
    return outputs


def expert_branch_modulation(
    inputs_normalized, output_size, params, training, name="expert_branch_modulation"
):
    print("Using expert branch modulation")
    with tf.variable_scope(name):
        # output is [batch_size, ?, n_feats]
        outputs = expert_branch_features(inputs_normalized, params, training)

        # hidden layer - optional
        if params[pkeys.EXPERT_BRANCH_MODULATION_HIDDEN_FILTERS] is not None:
            # [batch_size, time_len, n_feats] -> [batch_size, time_len, 1, feats]
            outputs = tf.expand_dims(outputs, axis=2)
            outputs = tf.layers.conv2d(
                inputs=outputs,
                filters=params[pkeys.EXPERT_BRANCH_MODULATION_HIDDEN_FILTERS],
                kernel_size=(
                    params[pkeys.EXPERT_BRANCH_MODULATION_HIDDEN_KERNEL_SIZE],
                    1,
                ),
                activation=tf.nn.relu,
                padding=constants.PAD_SAME,
                kernel_initializer=tf.initializers.he_normal(),
                use_bias=True,
                name="hidden_conv%d"
                % params[pkeys.EXPERT_BRANCH_MODULATION_HIDDEN_KERNEL_SIZE],
            )
            # [batch_size, time_len, 1, n_units] -> [batch_size, time_len, n_units]
            outputs = tf.squeeze(outputs, axis=2)

        # Scale and bias
        if params[pkeys.EXPERT_BRANCH_MODULATION_USE_SCALE]:
            modulation_scale = layers.sequence_fc_layer(
                outputs,
                output_size,
                kernel_init=tf.initializers.he_normal(),
                training=training,
                name="mod_scale",
            )
            if params[pkeys.EXPERT_BRANCH_MODULATION_APPLY_SIGMOID_SCALE]:
                modulation_scale = tf.math.sigmoid(modulation_scale)
        else:
            modulation_scale = 1.0

        if params[pkeys.EXPERT_BRANCH_MODULATION_USE_BIAS]:
            modulation_bias = layers.sequence_fc_layer(
                outputs,
                output_size,
                kernel_init=tf.initializers.he_normal(),
                training=training,
                use_bias=False,
                name="mod_bias",
            )
        else:
            modulation_bias = 0.0
        # output has shape [batch, time_len, dim] or scalar
    return modulation_scale, modulation_bias


def wavelet_blstm_net_v11_mkd2_expertmod(
    inputs, params, training, name="model_v11_mkd2_expertmod"
):
    print(
        "Using model V11-MKD-2-EXPERT-MOD"
        "(Time-Domain with multi-dilated convolutions, border crop AFTER lstm, expert modulation)"
    )
    with tf.variable_scope(name):
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)
        # BN at input
        inputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )
        outputs = segment_net(inputs, params, training)
        logits = layers.sequence_fc_layer(
            outputs,
            2,
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_OUTPUT],
            training=training,
            use_bias=False,
            name="logits",
        )  # [batch, time_len, 2]
        print("Modulating logits with expert features")
        mod_scale, mod_bias = expert_branch_modulation(
            inputs, 2, params, training
        )  # [batch, ?, 2]
        logits = mod_scale * logits + mod_bias  # [batch, time_len, 2]
        with tf.variable_scope("add_bias_static"):
            static_bias = tf.Variable(
                initial_value=[0.0, 0.0],
                trainable=True,
                name="static_bias",
                dtype=tf.float32,
            )
            logits = logits + static_bias
        with tf.variable_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", probabilities)
        cwt_prebn = None
    return logits, probabilities, cwt_prebn


def expert_regression_branch(
    inputs_normalized, hidden_layer, params, training, name="expert_branch_regression"
):
    print("Using expert branch regression (self supervision)")
    with tf.variable_scope(name):
        targets = expert_branch_features(
            inputs_normalized,
            params,
            training,
            batchnorm_use_bias=False,
            batchnorm_use_scale=False,
            trainable_window_duration=False,
        )
        n_targets = targets.get_shape().as_list()[-1]
        # Regression
        outputs = hidden_layer
        n_hidden = params[pkeys.EXPERT_BRANCH_REGRESSION_HIDDEN_UNITS]
        if n_hidden:
            outputs = layers.sequence_fc_layer(
                outputs,
                n_hidden,
                kernel_init=tf.initializers.he_normal(),
                training=training,
                activation=tf.nn.relu,
                name="fc_hidden",
            )
        outputs = layers.sequence_fc_layer(
            outputs,
            n_targets,
            kernel_init=tf.initializers.he_normal(),
            training=training,
            name="prediction",
        )
        mse_loss = tf.reduce_mean((outputs - targets) ** 2)
    return mse_loss


def wavelet_blstm_net_v11_mkd2_expertreg(
    inputs, params, training, name="model_v11_mkd2_expertreg"
):
    print(
        "Using model V11-MKD-2-EXPERT-REG"
        "(Time-Domain with multi-dilated convolutions, border crop AFTER lstm, expert regression)"
    )
    with tf.variable_scope(name):
        # Transform [batch, time_len] -> [batch, time_len, 1]
        inputs = tf.expand_dims(inputs, axis=2)
        inputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        outputs_blstm = segment_net(inputs, params, training, return_blstm_output=True)

        # FC hidden
        outputs = layers.sequence_fc_layer(
            outputs_blstm,
            params[pkeys.FC_UNITS],
            kernel_init=tf.initializers.he_normal(),
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            activation=tf.nn.relu,
            name="fc",
        )

        # Regression loss
        if params[pkeys.EXPERT_BRANCH_REGRESSION_FROM_BLSTM]:
            hidden_layer_for_regression = outputs_blstm
        else:
            hidden_layer_for_regression = outputs
        regression_loss = expert_regression_branch(
            inputs, hidden_layer_for_regression, params, training
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

        other_outputs_dict = {"regression_loss": regression_loss}

    return logits, probabilities, other_outputs_dict


def wavelet_blstm_net_v11_mkd2_swish(
    inputs, params, training, name="model_v11_mkd2_swish"
):
    print(
        "Using model V11-MKD-2-SWISH (Time-Domain with multi-dilated convolutions, border crop AFTER lstm)"
    )
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
        drop_rate_conv = params[pkeys.TIME_CONV_MK_DROP_RATE]
        drop_rate_conv = 0 if (drop_rate_conv is None) else drop_rate_conv
        drop_conv = params[pkeys.TYPE_DROPOUT] if (drop_rate_conv > 0) else None

        print("Conv dropout type %s and rate %s" % (drop_conv, drop_rate_conv))
        print("Projection first flag: %s" % params[pkeys.TIME_CONV_MK_PROJECT_FIRST])

        print("First convolutional block")
        tmp_out_list = []
        for kernel_size, n_filters, dilation in params[pkeys.TIME_CONV_MKD_FILTERS_1]:
            print("    k %d, f %d and d %d" % (kernel_size, n_filters, dilation))
            tmp_out = layers.conv1d_prebn_block_with_dilation(
                outputs,
                n_filters,
                training,
                kernel_size=kernel_size,
                dilation=dilation,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                dropout=drop_conv,
                drop_rate=drop_rate_conv,
                kernel_init=tf.initializers.he_normal(),
                activation_fn=tf.nn.swish,
                use_scale_at_bn=True,
                name="convblock_1d_k%d_d%d_1" % (kernel_size, dilation),
            )
            tmp_out_list.append(tmp_out)
        outputs_1 = tf.concat(tmp_out_list, axis=-1)

        print("Second convolutional block")
        tmp_out_list = []
        for kernel_size, n_filters, dilation in params[pkeys.TIME_CONV_MKD_FILTERS_2]:
            print("    k %d, f %d and d %d" % (kernel_size, n_filters, dilation))
            tmp_out = layers.conv1d_prebn_block_with_dilation(
                outputs_1,
                n_filters,
                training,
                kernel_size=kernel_size,
                dilation=dilation,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                dropout=drop_conv,
                drop_rate=drop_rate_conv,
                kernel_init=tf.initializers.he_normal(),
                activation_fn=tf.nn.swish,
                use_scale_at_bn=True,
                name="convblock_1d_k%d_d%d_2" % (kernel_size, dilation),
            )
            tmp_out_list.append(tmp_out)
        outputs_2 = tf.concat(tmp_out_list, axis=-1)

        print("Third convolutional block")
        tmp_out_list = []
        for kernel_size, n_filters, dilation in params[pkeys.TIME_CONV_MKD_FILTERS_3]:
            print("    k %d, f %d and d %d" % (kernel_size, n_filters, dilation))
            tmp_out = layers.conv1d_prebn_block_with_dilation(
                outputs_2,
                n_filters,
                training,
                kernel_size=kernel_size,
                dilation=dilation,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                downsampling=params[pkeys.CONV_DOWNSAMPLING],
                dropout=drop_conv,
                drop_rate=drop_rate_conv,
                kernel_init=tf.initializers.he_normal(),
                activation_fn=tf.nn.swish,
                use_scale_at_bn=True,
                name="convblock_1d_k%d_d%d_3" % (kernel_size, dilation),
            )
            tmp_out_list.append(tmp_out)
        outputs_3 = tf.concat(tmp_out_list, axis=-1)

        if params[pkeys.TIME_CONV_MK_SKIPS]:
            print("Passing feature pyramid to LSTM")
            # outputs_1 needs 2 additional pooling
            outputs_1 = tf.expand_dims(outputs_1, axis=2)
            outputs_1 = tf.layers.average_pooling2d(
                inputs=outputs_1, pool_size=(4, 1), strides=(4, 1)
            )
            outputs_1 = tf.squeeze(outputs_1, axis=2, name="squeeze")
            # outputs_2 needs 1 additional pooling
            outputs_2 = tf.expand_dims(outputs_2, axis=2)
            outputs_2 = tf.layers.average_pooling2d(
                inputs=outputs_2, pool_size=(2, 1), strides=(2, 1)
            )
            outputs_2 = tf.squeeze(outputs_2, axis=2, name="squeeze")
            # Concat each block for multi-scale features
            outputs = tf.concat([outputs_1, outputs_2, outputs_3], axis=-1)
        else:
            print("Passing last output to LSTM")
            # Just the last output
            outputs = outputs_3

        border_duration_to_crop_after_conv = 1
        border_duration_to_crop_after_lstm = (
            params[pkeys.BORDER_DURATION] - border_duration_to_crop_after_conv
        )

        border_crop = int(border_duration_to_crop_after_conv * params[pkeys.FS] // 8)
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        outputs = outputs[:, start_crop:end_crop]

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

        # Now crop the rest
        border_crop = int(border_duration_to_crop_after_lstm * params[pkeys.FS] // 8)
        start_crop = border_crop
        if border_crop <= 0:
            end_crop = None
        else:
            end_crop = -border_crop
        outputs = outputs[:, start_crop:end_crop]

        if params[pkeys.FC_UNITS] > 0:
            # Additional FC layer to increase model flexibility
            outputs = layers.sequence_fc_layer(
                outputs,
                params[pkeys.FC_UNITS],
                kernel_init=tf.initializers.he_normal(),
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                activation=tf.nn.swish,
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


def residual_feature_extractor(
    inputs,
    params,
    training,
):
    with tf.variable_scope("conv1"):
        outputs = tf.keras.layers.Conv1D(
            filters=params[pkeys.BIGGER_STEM_FILTERS],
            kernel_size=params[pkeys.BIGGER_STEM_KERNEL_SIZE],
            padding=constants.PAD_SAME,
            use_bias=False,
            kernel_initializer=tf.initializers.he_normal(),
            name="conv%d" % params[pkeys.BIGGER_STEM_KERNEL_SIZE],
        )(inputs)
        outputs = layers.batchnorm_layer(
            outputs,
            "bn",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
            scale=False,
        )
        outputs = tf.nn.relu(outputs)

    # Residual blocks
    stage_sizes = [
        params[pkeys.BIGGER_STAGE_1_SIZE],
        params[pkeys.BIGGER_STAGE_2_SIZE],
        params[pkeys.BIGGER_STAGE_3_SIZE],
    ]
    last_stage = 3 if stage_sizes[2] > 0 else 2
    for stage in range(3):
        stage_num = stage + 1
        stage_filters = params[pkeys.BIGGER_STEM_FILTERS] * (2**stage_num)
        stage_kernel_size = params[pkeys.BIGGER_BLOCKS_KERNEL_SIZE]
        outputs = tf.keras.layers.AvgPool1D(name="pool%d" % stage_num)(outputs)
        dilation_base = 2 if stage_num == last_stage else 1
        for i in range(stage_sizes[stage]):
            block_num = i + 1
            dilation_rate = dilation_base**i
            dilation_rate = min(dilation_rate, params[pkeys.BIGGER_MAX_DILATION])
            is_first_block = (stage_num == 1) and (block_num == 1)

            with tf.variable_scope("stage%d-%d" % (stage_num, block_num)):

                shortcut = outputs

                if not is_first_block:
                    outputs = layers.batchnorm_layer(
                        outputs,
                        "bn-a",
                        batchnorm=params[pkeys.TYPE_BATCHNORM],
                        training=training,
                        scale=False,
                    )
                    outputs = tf.nn.relu(outputs)

                outputs = tf.keras.layers.Conv1D(
                    filters=stage_filters,
                    kernel_size=stage_kernel_size,
                    padding=constants.PAD_SAME,
                    dilation_rate=dilation_rate,
                    use_bias=False,
                    kernel_initializer=tf.initializers.he_normal(),
                    name="conv%d-d%d-a" % (stage_kernel_size, dilation_rate),
                )(outputs)

                outputs = layers.batchnorm_layer(
                    outputs,
                    "bn-b",
                    batchnorm=params[pkeys.TYPE_BATCHNORM],
                    training=training,
                    scale=False,
                )
                outputs = tf.nn.relu(outputs)

                outputs = tf.keras.layers.Conv1D(
                    filters=stage_filters,
                    kernel_size=stage_kernel_size,
                    padding=constants.PAD_SAME,
                    dilation_rate=dilation_rate,
                    use_bias=False,
                    kernel_initializer=tf.initializers.he_normal(),
                    name="conv%d-d%d-b" % (stage_kernel_size, dilation_rate),
                )(outputs)

                if block_num == 1:
                    shortcut = tf.keras.layers.Conv1D(
                        filters=stage_filters,
                        kernel_size=1,
                        padding=constants.PAD_SAME,
                        use_bias=False,
                        kernel_initializer=tf.initializers.he_normal(),
                        name="project",
                    )(shortcut)

                outputs = outputs + shortcut

                if (stage_num == last_stage) and (block_num == stage_sizes[stage]):
                    # Finish residuals
                    outputs = layers.batchnorm_layer(
                        outputs,
                        "bn-c",
                        batchnorm=params[pkeys.TYPE_BATCHNORM],
                        training=training,
                        scale=False,
                    )
                    outputs = tf.nn.relu(outputs)
    return outputs


def multi_dilated_feature_extractor(inputs, params, training):
    with tf.variable_scope("stem"):
        outputs = inputs
        for i in range(params[pkeys.BIGGER_STEM_DEPTH]):
            layer_num = i + 1
            outputs = tf.keras.layers.Conv1D(
                filters=params[pkeys.BIGGER_STEM_FILTERS],
                kernel_size=params[pkeys.BIGGER_STEM_KERNEL_SIZE],
                padding=constants.PAD_SAME,
                use_bias=False,
                kernel_initializer=tf.initializers.he_normal(),
                name="conv%d_%d" % (params[pkeys.BIGGER_STEM_KERNEL_SIZE], layer_num),
            )(outputs)
            outputs = layers.batchnorm_layer(
                outputs,
                "bn_%d" % layer_num,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                training=training,
                scale=False,
            )
            outputs = tf.nn.relu(outputs)

    max_exponent = int(np.round(np.log(params[pkeys.BIGGER_MAX_DILATION]) / np.log(2)))
    stage_sizes = [1, 1, 0]
    stage_kernel_size = params[pkeys.BIGGER_BLOCKS_KERNEL_SIZE]

    for stage in range(len(stage_sizes)):
        stage_num = stage + 1

        outputs = tf.keras.layers.AvgPool1D(name="pool%d" % stage_num)(outputs)

        stage_filters = params[pkeys.BIGGER_STEM_FILTERS] * (2**stage_num)

        stage_config = []
        for single_exponent in range(max_exponent + 1):
            f = int(stage_filters / (2 ** (single_exponent + 1)))
            d = int(2**single_exponent)
            stage_config.append((f, d))
        stage_config[-1] = (2 * stage_config[-1][0], stage_config[-1][1])

        for i in range(stage_sizes[stage]):
            block_num = i + 1
            with tf.variable_scope("stage%d-%d" % (stage_num, block_num)):
                # Here we split
                # Apply computed config
                outputs_branches = []
                for branch_filters, branch_dilation in stage_config:
                    with tf.variable_scope("branch-d%d" % branch_dilation):
                        branch_outputs = tf.keras.layers.Conv1D(
                            filters=branch_filters,
                            kernel_size=stage_kernel_size,
                            padding=constants.PAD_SAME,
                            dilation_rate=branch_dilation,
                            use_bias=False,
                            kernel_initializer=tf.initializers.he_normal(),
                            name="conv%d-d%d-a" % (stage_kernel_size, branch_dilation),
                        )(outputs)
                        branch_outputs = layers.batchnorm_layer(
                            branch_outputs,
                            "bn-a",
                            batchnorm=params[pkeys.TYPE_BATCHNORM],
                            training=training,
                            scale=False,
                        )
                        branch_outputs = tf.nn.relu(branch_outputs)
                        branch_outputs = tf.keras.layers.Conv1D(
                            filters=branch_filters,
                            kernel_size=stage_kernel_size,
                            padding=constants.PAD_SAME,
                            dilation_rate=branch_dilation,
                            use_bias=False,
                            kernel_initializer=tf.initializers.he_normal(),
                            name="conv%d-d%d-b" % (stage_kernel_size, branch_dilation),
                        )(branch_outputs)
                        branch_outputs = layers.batchnorm_layer(
                            branch_outputs,
                            "bn-b",
                            batchnorm=params[pkeys.TYPE_BATCHNORM],
                            training=training,
                            scale=False,
                        )
                        branch_outputs = tf.nn.relu(branch_outputs)
                        outputs_branches.append(branch_outputs)
                outputs = tf.concat(outputs_branches, axis=-1)
                transformation = params[pkeys.BIGGER_MULTI_TRANSFORMATION_BEFORE_ADD]
                if transformation == "nonlinear":
                    outputs = tf.keras.layers.Conv1D(
                        filters=stage_filters,
                        kernel_size=1,
                        padding=constants.PAD_SAME,
                        use_bias=False,
                        kernel_initializer=tf.initializers.he_normal(),
                        name="conv1",
                    )(outputs)
                    outputs = layers.batchnorm_layer(
                        outputs,
                        "bn_extra",
                        batchnorm=params[pkeys.TYPE_BATCHNORM],
                        training=training,
                        scale=False,
                    )
                    outputs = tf.nn.relu(outputs)
    return outputs


def residual_multi_dilated_feature_extractor(inputs, params, training):
    with tf.variable_scope("stem"):
        outputs = inputs
        for i in range(params[pkeys.BIGGER_STEM_DEPTH]):
            layer_num = i + 1
            outputs = tf.keras.layers.Conv1D(
                filters=params[pkeys.BIGGER_STEM_FILTERS],
                kernel_size=params[pkeys.BIGGER_STEM_KERNEL_SIZE],
                padding=constants.PAD_SAME,
                use_bias=False,
                kernel_initializer=tf.initializers.he_normal(),
                name="conv%d_%d" % (params[pkeys.BIGGER_STEM_KERNEL_SIZE], layer_num),
            )(outputs)
            outputs = layers.batchnorm_layer(
                outputs,
                "bn_%d" % layer_num,
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                training=training,
                scale=False,
            )
            outputs = tf.nn.relu(outputs)

    # Residual blocks
    max_exponent = int(np.round(np.log(params[pkeys.BIGGER_MAX_DILATION]) / np.log(2)))
    stage_sizes = [
        params[pkeys.BIGGER_STAGE_1_SIZE],
        params[pkeys.BIGGER_STAGE_2_SIZE],
        params[pkeys.BIGGER_STAGE_3_SIZE],
    ]
    stage_kernel_size = params[pkeys.BIGGER_BLOCKS_KERNEL_SIZE]

    last_stage = 3 if stage_sizes[2] > 0 else 2
    for stage in range(len(stage_sizes)):
        stage_num = stage + 1

        outputs = tf.keras.layers.AvgPool1D(name="pool%d" % stage_num)(outputs)

        stage_filters = params[pkeys.BIGGER_STEM_FILTERS] * (2**stage_num)

        stage_config = []
        for single_exponent in range(max_exponent + 1):
            f = int(stage_filters / (2 ** (single_exponent + 1)))
            d = int(2**single_exponent)
            stage_config.append((f, d))
        stage_config[-1] = (2 * stage_config[-1][0], stage_config[-1][1])

        for i in range(stage_sizes[stage]):
            block_num = i + 1
            is_first_block = (stage_num == 1) and (block_num == 1)

            with tf.variable_scope("stage%d-%d" % (stage_num, block_num)):

                shortcut = outputs

                if not is_first_block:
                    outputs = layers.batchnorm_layer(
                        outputs,
                        "bn-a",
                        batchnorm=params[pkeys.TYPE_BATCHNORM],
                        training=training,
                        scale=False,
                    )
                    outputs = tf.nn.relu(outputs)

                # Here we split
                # Apply computed config
                outputs_branches = []
                for branch_filters, branch_dilation in stage_config:
                    with tf.variable_scope("branch-d%d" % branch_dilation):
                        branch_outputs = tf.keras.layers.Conv1D(
                            filters=branch_filters,
                            kernel_size=stage_kernel_size,
                            padding=constants.PAD_SAME,
                            dilation_rate=branch_dilation,
                            use_bias=False,
                            kernel_initializer=tf.initializers.he_normal(),
                            name="conv%d-d%d-a" % (stage_kernel_size, branch_dilation),
                        )(outputs)
                        branch_outputs = layers.batchnorm_layer(
                            branch_outputs,
                            "bn-b",
                            batchnorm=params[pkeys.TYPE_BATCHNORM],
                            training=training,
                            scale=False,
                        )
                        branch_outputs = tf.nn.relu(branch_outputs)
                        branch_outputs = tf.keras.layers.Conv1D(
                            filters=branch_filters,
                            kernel_size=stage_kernel_size,
                            padding=constants.PAD_SAME,
                            dilation_rate=branch_dilation,
                            use_bias=False,
                            kernel_initializer=tf.initializers.he_normal(),
                            name="conv%d-d%d-b" % (stage_kernel_size, branch_dilation),
                        )(branch_outputs)
                        outputs_branches.append(branch_outputs)
                outputs = tf.concat(outputs_branches, axis=-1)

                transformation = params[pkeys.BIGGER_MULTI_TRANSFORMATION_BEFORE_ADD]
                if transformation == "linear":
                    outputs = tf.keras.layers.Conv1D(
                        filters=stage_filters,
                        kernel_size=1,
                        padding=constants.PAD_SAME,
                        use_bias=False,
                        kernel_initializer=tf.initializers.he_normal(),
                        name="conv1",
                    )(outputs)
                elif transformation == "nonlinear":
                    outputs = layers.batchnorm_layer(
                        outputs,
                        "bn_extra",
                        batchnorm=params[pkeys.TYPE_BATCHNORM],
                        training=training,
                        scale=False,
                    )
                    outputs = tf.nn.relu(outputs)
                    outputs = tf.keras.layers.Conv1D(
                        filters=stage_filters,
                        kernel_size=1,
                        padding=constants.PAD_SAME,
                        use_bias=False,
                        kernel_initializer=tf.initializers.he_normal(),
                        name="conv1",
                    )(outputs)

                if block_num == 1:
                    shortcut = tf.keras.layers.Conv1D(
                        filters=stage_filters,
                        kernel_size=1,
                        padding=constants.PAD_SAME,
                        use_bias=False,
                        kernel_initializer=tf.initializers.he_normal(),
                        name="project",
                    )(shortcut)

                outputs = outputs + shortcut

                if (stage_num == last_stage) and (block_num == stage_sizes[stage]):
                    # Finish residuals
                    outputs = layers.batchnorm_layer(
                        outputs,
                        "bn-c",
                        batchnorm=params[pkeys.TYPE_BATCHNORM],
                        training=training,
                        scale=False,
                    )
                    outputs = tf.nn.relu(outputs)
    return outputs


def lstm_contextualization(
    inputs,
    params,
    training,
):
    outputs = inputs
    # Lstm - First layer
    if params[pkeys.BIGGER_LSTM_1_SIZE] > 0:
        outputs = layers.lstm_layer(
            outputs,
            num_units=params[pkeys.BIGGER_LSTM_1_SIZE],
            num_dirs=constants.BIDIRECTIONAL,
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_BEFORE_LSTM],
            training=training,
            name="lstm_1",
        )
    # Lstm - Second layer
    if params[pkeys.BIGGER_LSTM_2_SIZE] > 0:
        outputs = layers.lstm_layer(
            outputs,
            num_units=params[pkeys.BIGGER_LSTM_2_SIZE],
            num_dirs=constants.BIDIRECTIONAL,
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_HIDDEN],
            training=training,
            name="lstm_2",
        )
    # Additional FC layer to increase model flexibility
    if params[pkeys.FC_UNITS] > 0:
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
    return outputs


def lstm_skip_early_contextualization(
    inputs,
    params,
    training,
):
    outputs = layers.dropout_layer(
        inputs,
        "drop_0",
        drop_rate=params[pkeys.DROP_RATE_BEFORE_LSTM],
        dropout=params[pkeys.TYPE_DROPOUT],
        training=training,
    )
    shortcut = outputs
    # Lstm - First layer
    if params[pkeys.BIGGER_LSTM_1_SIZE] > 0:
        outputs = layers.lstm_layer(
            outputs,
            num_units=params[pkeys.BIGGER_LSTM_1_SIZE],
            num_dirs=constants.BIDIRECTIONAL,
            training=training,
            name="lstm_1",
        )
        outputs = layers.dropout_layer(
            outputs,
            "drop_1",
            drop_rate=params[pkeys.DROP_RATE_HIDDEN],
            dropout=params[pkeys.TYPE_DROPOUT],
            training=training,
        )
    # Lstm - Second layer
    if params[pkeys.BIGGER_LSTM_2_SIZE] > 0:
        outputs = layers.lstm_layer(
            outputs,
            num_units=params[pkeys.BIGGER_LSTM_2_SIZE],
            num_dirs=constants.BIDIRECTIONAL,
            training=training,
            name="lstm_2",
        )
        outputs = layers.dropout_layer(
            outputs,
            "drop_2",
            drop_rate=params[pkeys.DROP_RATE_HIDDEN],
            dropout=params[pkeys.TYPE_DROPOUT],
            training=training,
        )
    outputs = tf.concat([outputs, shortcut], axis=-1)
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
    return outputs


def lstm_skip_late_contextualization(
    inputs,
    params,
    training,
):
    outputs = layers.dropout_layer(
        inputs,
        "drop_0",
        drop_rate=params[pkeys.DROP_RATE_BEFORE_LSTM],
        dropout=params[pkeys.TYPE_DROPOUT],
        training=training,
    )
    shortcut = outputs
    # Lstm - First layer
    if params[pkeys.BIGGER_LSTM_1_SIZE] > 0:
        outputs = layers.lstm_layer(
            outputs,
            num_units=params[pkeys.BIGGER_LSTM_1_SIZE],
            num_dirs=constants.BIDIRECTIONAL,
            training=training,
            name="lstm_1",
        )
        outputs = layers.dropout_layer(
            outputs,
            "drop_1",
            drop_rate=params[pkeys.DROP_RATE_HIDDEN],
            dropout=params[pkeys.TYPE_DROPOUT],
            training=training,
        )
    # Lstm - Second layer
    if params[pkeys.BIGGER_LSTM_2_SIZE] > 0:
        outputs = layers.lstm_layer(
            outputs,
            num_units=params[pkeys.BIGGER_LSTM_2_SIZE],
            num_dirs=constants.BIDIRECTIONAL,
            training=training,
            name="lstm_2",
        )
        outputs = layers.dropout_layer(
            outputs,
            "drop_2",
            drop_rate=params[pkeys.DROP_RATE_HIDDEN],
            dropout=params[pkeys.TYPE_DROPOUT],
            training=training,
        )
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
    outputs = tf.concat([outputs, shortcut], axis=-1)
    return outputs


def attention_contextualization(
    inputs,
    params,
    training,
):
    outputs = inputs
    # Self-attention specs
    seq_len = outputs.get_shape().as_list()[1]
    att_dim = params[pkeys.ATT_DIM]
    n_heads = params[pkeys.ATT_N_HEADS]
    att_drop_rate = params[pkeys.ATT_DROP_RATE]

    with tf.variable_scope("add_pos_enc"):
        outputs = tf.keras.layers.Conv1D(
            filters=att_dim,
            kernel_size=1,
            padding=constants.PAD_SAME,
            use_bias=False,
            kernel_initializer=tf.initializers.he_normal(),
            name="project",
        )(outputs)
        pos_enc = layers.get_positional_encoding(
            seq_len=seq_len,
            dims=att_dim,
            pe_factor=params[pkeys.ATT_PE_FACTOR],
            name="pos_enc",
        )
        pos_enc = tf.expand_dims(pos_enc, axis=0)  # Add batch axis
        outputs = outputs + pos_enc
        outputs = tf.layers.dropout(outputs, training=training, rate=att_drop_rate)

    att_weights_dict = {}
    for att_block in range(params[pkeys.BIGGER_ATT_N_BLOCKS]):
        att_block_num = att_block + 1
        with tf.variable_scope("att%d" % att_block_num):
            # Sublayer 1: self-attention
            shortcut = outputs
            queries = tf.keras.layers.Conv1D(
                filters=att_dim,
                kernel_size=1,
                kernel_initializer=tf.initializers.he_normal(),
                name="q",
            )(outputs)
            keys = tf.keras.layers.Conv1D(
                filters=att_dim,
                kernel_size=1,
                kernel_initializer=tf.initializers.he_normal(),
                name="k",
            )(outputs)
            values = tf.keras.layers.Conv1D(
                filters=att_dim,
                kernel_size=1,
                kernel_initializer=tf.initializers.he_normal(),
                name="v",
            )(outputs)
            concat_heads, attention_weights = layers.multihead_attention_layer(
                queries, keys, values, n_heads, name="multi-att"
            )
            att_weights_dict["att_weights_%d" % att_block_num] = attention_weights
            outputs = tf.keras.layers.Conv1D(
                filters=att_dim,
                kernel_size=1,
                kernel_initializer=tf.initializers.he_normal(),
                name="o",
            )(concat_heads)
            outputs = tf.layers.dropout(outputs, training=training, rate=att_drop_rate)
            outputs = outputs + shortcut
            if params[pkeys.BIGGER_ATT_TYPE_NORM] == "layernorm":
                outputs = tf.keras.layers.LayerNormalization()(outputs)
            elif params[pkeys.BIGGER_ATT_TYPE_NORM] == "batchnorm":
                outputs = layers.batchnorm_layer(
                    outputs,
                    "bn1",
                    batchnorm=params[pkeys.TYPE_BATCHNORM],
                    training=training,
                )

            # Sublayer 2: ffn
            shortcut = outputs
            outputs = tf.keras.layers.Conv1D(
                filters=2 * att_dim,
                kernel_size=1,
                kernel_initializer=tf.initializers.he_normal(),
                name="fc1",
            )(outputs)
            outputs = tf.nn.relu(outputs)
            outputs = tf.keras.layers.Conv1D(
                filters=att_dim,
                kernel_size=1,
                kernel_initializer=tf.initializers.he_normal(),
                name="fc2",
            )(outputs)
            outputs = tf.layers.dropout(outputs, training=training, rate=att_drop_rate)
            outputs = outputs + shortcut
            if params[pkeys.BIGGER_ATT_TYPE_NORM] == "layernorm":
                outputs = tf.keras.layers.LayerNormalization()(outputs)
            elif params[pkeys.BIGGER_ATT_TYPE_NORM] == "batchnorm":
                outputs = layers.batchnorm_layer(
                    outputs,
                    "bn2",
                    batchnorm=params[pkeys.TYPE_BATCHNORM],
                    training=training,
                )
    return outputs, att_weights_dict


def residual_lstm_contextualization(
    inputs,
    params,
    training,
):
    outputs = layers.dropout_layer(
        inputs,
        "drop_in",
        training,
        dropout=params[pkeys.TYPE_DROPOUT],
        drop_rate=params[pkeys.DROP_RATE_BEFORE_LSTM],
    )
    # Lstm - First layer
    if params[pkeys.BIGGER_LSTM_1_SIZE] > 0:
        layer_dim = 2 * params[pkeys.BIGGER_LSTM_1_SIZE]
        shortcut = outputs
        outputs = layers.lstm_layer(
            outputs,
            num_units=layer_dim // 2,
            num_dirs=constants.BIDIRECTIONAL,
            training=training,
            name="lstm_1",
        )
        outputs = tf.keras.layers.Conv1D(
            filters=layer_dim,
            kernel_size=1,
            use_bias=False,
            kernel_initializer=tf.initializers.he_normal(),
            name="linear_1",
        )(outputs)
        outputs = layers.dropout_layer(
            outputs,
            "drop_1",
            training,
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_HIDDEN],
        )
        # project shortcut if necessary
        if shortcut.get_shape().as_list()[-1] != layer_dim:
            shortcut = tf.keras.layers.Conv1D(
                filters=layer_dim,
                kernel_size=1,
                use_bias=False,
                kernel_initializer=tf.initializers.he_normal(),
                name="project_1",
            )(shortcut)
        # add & norm
        outputs = outputs + shortcut
        if params[pkeys.BIGGER_ATT_TYPE_NORM] == "layernorm":
            outputs = tf.keras.layers.LayerNormalization()(outputs)
        elif params[pkeys.BIGGER_ATT_TYPE_NORM] == "batchnorm":
            outputs = layers.batchnorm_layer(
                outputs,
                "bn_1",
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                training=training,
            )

    # Lstm - Second layer
    if params[pkeys.BIGGER_LSTM_2_SIZE] > 0:
        layer_dim = 2 * params[pkeys.BIGGER_LSTM_2_SIZE]
        shortcut = outputs
        outputs = layers.lstm_layer(
            outputs,
            num_units=layer_dim // 2,
            num_dirs=constants.BIDIRECTIONAL,
            training=training,
            name="lstm_2",
        )
        outputs = tf.keras.layers.Conv1D(
            filters=layer_dim,
            kernel_size=1,
            use_bias=False,
            kernel_initializer=tf.initializers.he_normal(),
            name="linear_2",
        )(outputs)
        outputs = layers.dropout_layer(
            outputs,
            "drop_2",
            training,
            dropout=params[pkeys.TYPE_DROPOUT],
            drop_rate=params[pkeys.DROP_RATE_HIDDEN],
        )
        # project shortcut if necessary
        if shortcut.get_shape().as_list()[-1] != layer_dim:
            shortcut = tf.keras.layers.Conv1D(
                filters=layer_dim,
                kernel_size=1,
                use_bias=False,
                kernel_initializer=tf.initializers.he_normal(),
                name="project_2",
            )(shortcut)
        # add & norm
        outputs = outputs + shortcut
        if params[pkeys.BIGGER_ATT_TYPE_NORM] == "layernorm":
            outputs = tf.keras.layers.LayerNormalization()(outputs)
        elif params[pkeys.BIGGER_ATT_TYPE_NORM] == "batchnorm":
            outputs = layers.batchnorm_layer(
                outputs,
                "bn_2",
                batchnorm=params[pkeys.TYPE_BATCHNORM],
                training=training,
            )

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
    return outputs


def wavelet_blstm_net_v41(inputs, params, training, name="model_v41"):
    print(
        "Using model V41 (Time-Domain with residual (potentially dilated) stages, border crop AFTER lstm)"
    )
    with tf.variable_scope(name):
        inputs = tf.expand_dims(inputs, axis=2)  # [batch, time_len, 1]
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        outputs = residual_feature_extractor(outputs, params, training)

        border_duration_to_crop_after_conv = 1
        border_duration_to_crop_after_lstm = (
            params[pkeys.BORDER_DURATION] - border_duration_to_crop_after_conv
        )

        border_crop = int(border_duration_to_crop_after_conv * params[pkeys.FS] // 8)
        start_crop = border_crop
        end_crop = None if (border_crop <= 0) else -border_crop
        outputs = outputs[:, start_crop:end_crop]

        # Lstm - First layer
        if params[pkeys.BIGGER_LSTM_1_SIZE] > 0:
            outputs = layers.lstm_layer(
                outputs,
                num_units=params[pkeys.BIGGER_LSTM_1_SIZE],
                num_dirs=constants.BIDIRECTIONAL,
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_BEFORE_LSTM],
                training=training,
                name="lstm_1",
            )
        # Lstm - Second layer
        if params[pkeys.BIGGER_LSTM_2_SIZE] > 0:
            outputs = layers.lstm_layer(
                outputs,
                num_units=params[pkeys.BIGGER_LSTM_2_SIZE],
                num_dirs=constants.BIDIRECTIONAL,
                dropout=params[pkeys.TYPE_DROPOUT],
                drop_rate=params[pkeys.DROP_RATE_HIDDEN],
                training=training,
                name="lstm_2",
            )

        # Now crop the rest
        border_crop = int(border_duration_to_crop_after_lstm * params[pkeys.FS] // 8)
        start_crop = border_crop
        end_crop = None if (border_crop <= 0) else -border_crop
        outputs = outputs[:, start_crop:end_crop]

        # Additional FC layer to increase model flexibility
        if params[pkeys.FC_UNITS] > 0:
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


def wavelet_blstm_net_v42(inputs, params, training, name="model_v42"):
    print("Using model V42 (v41 but with self-attention instead of lstm)")
    with tf.variable_scope(name):
        inputs = tf.expand_dims(inputs, axis=2)  # [batch, time_len, 1]
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        outputs = residual_feature_extractor(outputs, params, training)

        border_duration_to_crop_after_conv = 1
        border_duration_to_crop_after_lstm = (
            params[pkeys.BORDER_DURATION] - border_duration_to_crop_after_conv
        )

        border_crop = int(border_duration_to_crop_after_conv * params[pkeys.FS] // 8)
        start_crop = border_crop
        end_crop = None if (border_crop <= 0) else -border_crop
        outputs = outputs[:, start_crop:end_crop]

        # Self-attention specs
        seq_len = outputs.get_shape().as_list()[1]
        att_dim = params[pkeys.ATT_DIM]
        n_heads = params[pkeys.ATT_N_HEADS]
        att_drop_rate = params[pkeys.ATT_DROP_RATE]

        with tf.variable_scope("add_pos_enc"):
            outputs = tf.keras.layers.Conv1D(
                filters=att_dim,
                kernel_size=1,
                padding=constants.PAD_SAME,
                use_bias=False,
                kernel_initializer=tf.initializers.he_normal(),
                name="project",
            )(outputs)
            pos_enc = layers.get_positional_encoding(
                seq_len=seq_len,
                dims=att_dim,
                pe_factor=params[pkeys.ATT_PE_FACTOR],
                name="pos_enc",
            )
            pos_enc = tf.expand_dims(pos_enc, axis=0)  # Add batch axis
            outputs = outputs + pos_enc
            outputs = tf.layers.dropout(outputs, training=training, rate=att_drop_rate)

        att_weights_dict = {}
        for att_block in range(params[pkeys.BIGGER_ATT_N_BLOCKS]):
            att_block_num = att_block + 1
            with tf.variable_scope("att%d" % att_block_num):
                # Sublayer 1: self-attention
                shortcut = outputs
                queries = tf.keras.layers.Conv1D(
                    filters=att_dim,
                    kernel_size=1,
                    kernel_initializer=tf.initializers.he_normal(),
                    name="q",
                )(outputs)
                keys = tf.keras.layers.Conv1D(
                    filters=att_dim,
                    kernel_size=1,
                    kernel_initializer=tf.initializers.he_normal(),
                    name="k",
                )(outputs)
                values = tf.keras.layers.Conv1D(
                    filters=att_dim,
                    kernel_size=1,
                    kernel_initializer=tf.initializers.he_normal(),
                    name="v",
                )(outputs)
                concat_heads, attention_weights = layers.multihead_attention_layer(
                    queries, keys, values, n_heads, name="multi-att"
                )
                att_weights_dict["att_weights_%d" % att_block_num] = attention_weights
                outputs = tf.keras.layers.Conv1D(
                    filters=att_dim,
                    kernel_size=1,
                    kernel_initializer=tf.initializers.he_normal(),
                    name="o",
                )(concat_heads)
                outputs = tf.layers.dropout(
                    outputs, training=training, rate=att_drop_rate
                )
                outputs = outputs + shortcut
                if params[pkeys.BIGGER_ATT_TYPE_NORM] == "layernorm":
                    outputs = tf.keras.layers.LayerNormalization()(outputs)
                elif params[pkeys.BIGGER_ATT_TYPE_NORM] == "batchnorm":
                    outputs = layers.batchnorm_layer(
                        outputs,
                        "bn1",
                        batchnorm=params[pkeys.TYPE_BATCHNORM],
                        training=training,
                    )

                # Sublayer 2: ffn
                shortcut = outputs
                outputs = tf.keras.layers.Conv1D(
                    filters=2 * att_dim,
                    kernel_size=1,
                    kernel_initializer=tf.initializers.he_normal(),
                    name="fc1",
                )(outputs)
                outputs = tf.nn.relu(outputs)
                outputs = tf.keras.layers.Conv1D(
                    filters=att_dim,
                    kernel_size=1,
                    kernel_initializer=tf.initializers.he_normal(),
                    name="fc2",
                )(outputs)
                outputs = tf.layers.dropout(
                    outputs, training=training, rate=att_drop_rate
                )
                outputs = outputs + shortcut
                if params[pkeys.BIGGER_ATT_TYPE_NORM] == "layernorm":
                    outputs = tf.keras.layers.LayerNormalization()(outputs)
                elif params[pkeys.BIGGER_ATT_TYPE_NORM] == "batchnorm":
                    outputs = layers.batchnorm_layer(
                        outputs,
                        "bn2",
                        batchnorm=params[pkeys.TYPE_BATCHNORM],
                        training=training,
                    )

        # Now crop the rest
        border_crop = int(border_duration_to_crop_after_lstm * params[pkeys.FS] // 8)
        start_crop = border_crop
        end_crop = None if (border_crop <= 0) else -border_crop
        outputs = outputs[:, start_crop:end_crop]

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
        other_outputs_dict.update(att_weights_dict)
        return logits, probabilities, other_outputs_dict


def wavelet_blstm_net_v43(inputs, params, training, name="model_v43"):
    print("Using model V43 (basically a lego for BigNet)")
    with tf.variable_scope(name):
        inputs = tf.expand_dims(inputs, axis=2)  # [batch, time_len, 1]
        outputs = layers.batchnorm_layer(
            inputs,
            "bn_input",
            batchnorm=params[pkeys.TYPE_BATCHNORM],
            training=training,
        )

        # Feature extraction with convolutions
        if params[pkeys.BIGGER_CONVOLUTION_PART_OPTION] == "multi_dilated":
            outputs = multi_dilated_feature_extractor(outputs, params, training)
        elif params[pkeys.BIGGER_CONVOLUTION_PART_OPTION] == "residual_multi_dilated":
            outputs = residual_multi_dilated_feature_extractor(
                outputs, params, training
            )
        elif params[pkeys.BIGGER_CONVOLUTION_PART_OPTION] == "residual":
            outputs = residual_feature_extractor(outputs, params, training)
        else:
            raise ValueError(
                "Conv part invalid: %s" % params[pkeys.BIGGER_CONVOLUTION_PART_OPTION]
            )

        border_duration_to_crop_after_conv = 1
        border_duration_to_crop_after_lstm = (
            params[pkeys.BORDER_DURATION] - border_duration_to_crop_after_conv
        )

        border_crop = int(border_duration_to_crop_after_conv * params[pkeys.FS] // 8)
        start_crop = border_crop
        end_crop = None if (border_crop <= 0) else -border_crop
        outputs = outputs[:, start_crop:end_crop]

        # Contextualization
        if params[pkeys.BIGGER_CONTEXT_PART_OPTION] == "lstm":
            outputs = lstm_contextualization(outputs, params, training)
            att_weights_dict = {}
        elif params[pkeys.BIGGER_CONTEXT_PART_OPTION] == "attention":
            outputs, att_weights_dict = attention_contextualization(
                outputs, params, training
            )
        elif params[pkeys.BIGGER_CONTEXT_PART_OPTION] == "residual_lstm":
            outputs = residual_lstm_contextualization(outputs, params, training)
            att_weights_dict = {}
        elif params[pkeys.BIGGER_CONTEXT_PART_OPTION] == "lstm_skip_early":
            outputs = lstm_skip_early_contextualization(outputs, params, training)
            att_weights_dict = {}
        elif params[pkeys.BIGGER_CONTEXT_PART_OPTION] == "lstm_skip_late":
            outputs = lstm_skip_late_contextualization(outputs, params, training)
            att_weights_dict = {}
        else:
            raise ValueError(
                "Context part invalid: %s" % params[pkeys.BIGGER_CONTEXT_PART_OPTION]
            )

        # Now crop the rest
        border_crop = int(border_duration_to_crop_after_lstm * params[pkeys.FS] // 8)
        start_crop = border_crop
        end_crop = None if (border_crop <= 0) else -border_crop
        outputs = outputs[:, start_crop:end_crop]

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

        other_outputs_dict = {"last_hidden": outputs}
        other_outputs_dict.update(att_weights_dict)

    return logits, probabilities, other_outputs_dict
