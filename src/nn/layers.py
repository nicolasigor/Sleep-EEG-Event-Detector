"""layers.py: Module that defines several useful layers for neural network
models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.ops import array_ops

from src.nn.spectrum import compute_cwt, compute_cwt_rectangular, compute_wavelets, apply_wavelets_rectangular, compute_wavelets_noisy
from src.nn.expert_feats import lowpass_tf_batch
from src.common import constants
from src.common import checks


def power_ratio_literature_fixed_layer(
        inputs,
        fb_list,
        fs,
        lower_freq,
        upper_freq,
        n_scales,
        training,
        border_crop=0,
        use_log=False,
        return_power_bands=True,
        return_power_ratios=True,
        name="pr_lit_fixed"):
    """Fixed power ratios from SS det literature."""

    print('Using fixed power ratios from CWT')
    with tf.variable_scope(name):
        # Input sequence has shape [batch_size, time_len]
        wavelets, freqs = compute_wavelets(
            fb_list=fb_list,
            fs=fs,
            lower_freq=lower_freq,
            upper_freq=upper_freq,
            n_scales=n_scales,
            name='cmorlet')
        cwt = apply_wavelets_rectangular(
            inputs=inputs,
            wavelets=wavelets,
            border_crop=border_crop,
            name='cwt')
        # Output sequence has shape [batch_size, time_len, n_scales, channels]
        # Collapse real and imaginary channels to magnitude channel
        cwt_mag = tf.sqrt(tf.reduce_sum(cwt ** 2, axis=-1))
        # output is [batch_size, time_len, n_scales], is 1D

        def get_band_weights(freq_vals, lim_low, lim_high):
            """Weights have shape [1, 1, n_scales]."""
            band_low = (freq_vals >= lim_low).astype(np.float32)
            band_high = (freq_vals <= lim_high).astype(np.float32)
            band_weights = band_low * band_high
            band_weights = np.reshape(band_weights, (1, 1, -1))
            return band_weights

        def get_band_power(power_vals, freq_vals, lim_low, lim_high, mode):
            if mode not in ["mean", "max"]:
                raise ValueError()
            w = get_band_weights(freq_vals, lim_low, lim_high)
            weighted_power = w * power_vals
            if mode == "mean":
                band_power = tf.reduce_sum(weighted_power, axis=2) / tf.reduce_sum(w)
            else:
                band_power = tf.reduce_max(weighted_power, axis=2)
            return band_power

        outputs_to_concat = []
        if return_power_ratios:
            # ----------------------
            # Kulkarni (SpindleNet) power ratio
            with tf.variable_scope("kulkarni"):
                num_power = get_band_power(cwt_mag, freqs, 9, 16, mode="mean")
                den_power = get_band_power(cwt_mag, freqs, 2, 8, mode="mean")
                pr_spindlendet = num_power / (den_power + 1e-6)  # [batch, time_len]

            # Lacourse (A7) power ratio
            with tf.variable_scope("lacourse"):
                num_power = get_band_power(cwt_mag, freqs, 11, 16, mode="mean")
                den_power = get_band_power(cwt_mag, freqs, 4.5, 30, mode="mean")
                pr_a7 = num_power / (den_power + 1e-6)

            # Huupponen (sigma index) power ratio
            with tf.variable_scope("huupponen"):
                num_power = get_band_power(cwt_mag, freqs, 10.5, 16, mode="max")
                den_power_1 = get_band_power(cwt_mag, freqs, 4, 10, mode="mean")
                den_power_2 = get_band_power(cwt_mag, freqs, 20, 40, mode="mean")
                alpha_power = get_band_power(cwt_mag, freqs, 7.5, 10, mode="max")
                pr_huupp = 2.0 * num_power / (den_power_1 + den_power_2 + 1e-6)
                pr_huupp_alfa = num_power / (alpha_power + 1e-6)

            outputs_to_concat.extend([
                pr_spindlendet, pr_a7, pr_huupp, pr_huupp_alfa
            ])
        if return_power_bands:
            # Known medical frequency bands
            with tf.variable_scope("medical_bands"):
                delta_1_power = get_band_power(cwt_mag, freqs, 0.5, 2, mode="mean")
                delta_2_power = get_band_power(cwt_mag, freqs, 2, 4, mode="mean")
                theta_power = get_band_power(cwt_mag, freqs, 4, 8, mode="mean")
                alpha_power = get_band_power(cwt_mag, freqs, 8, 12, mode="mean")
                sigma_power = get_band_power(cwt_mag, freqs, 12, 15, mode="mean")
                beta_power = get_band_power(cwt_mag, freqs, 15, 30, mode="mean")

            outputs_to_concat.extend([
                delta_1_power, delta_2_power, theta_power,
                alpha_power, sigma_power, beta_power
            ])

        power_ratios = tf.stack(outputs_to_concat, axis=2)

        # Optional logarithm
        if use_log:
            power_ratios = tf.log(power_ratios + 1e-6)
        # Batch normalization
        power_ratios = batchnorm_layer(power_ratios, 'bn_pr', training=training)

    return power_ratios


def upsampling_1d_linear(inputs, name, up_factor):
    """Upsampling of features by factor 'up_factor'.

    The upsampling is linear, aligned, and it is performed in the feature axis.

    Args:
        inputs: (tensor) Tensor of shape [batch_size, feat_len, channels].
        name: (string) A name for the operation.
        up_factor: (int) Upsampling factor. Output has shape
            [batch_size, up_factor * feat_len, channels].
    """
    with tf.variable_scope(name):
        outputs = tf.expand_dims(inputs, 2)
        outputs = tf.keras.layers.UpSampling2D(
            size=(up_factor, 1), interpolation='bilinear')(outputs)
        outputs = tf.squeeze(outputs, 2)
        pad = up_factor // 2
        outputs = tf.concat(
            [inputs[:, 0:1, :]] * pad + [outputs[:, :-pad, :]], axis=1)
        if up_factor % 2 == 0:
            outputs = tf.keras.layers.AveragePooling1D(
                pool_size=2, strides=1, padding="same")(outputs)
    return outputs


def downsampling_1d(inputs, name, down_factor, type_pooling):
    """Downsampling of features by factor 'down_factor'.

    The downsampling is type_pooling, and it is performed in the feature axis.

    Args:
        inputs: (tensor) Tensor of shape [batch_size, feat_len, channels].
        name: (string) A name for the operation.
        down_factor: (int) Downsampling factor. Output has shape
            [batch_size, feat_len / down_factor, channels].
        type_pooling: (string) One of ['maxpool', 'avgpool'].
    """
    checks.check_valid_value(
        type_pooling, 'pooling', [constants.AVGPOOL, constants.MAXPOOL])
    with tf.variable_scope(name):
        if type_pooling == constants.AVGPOOL:
            outputs = tf.keras.layers.AveragePooling1D(
                pool_size=down_factor, strides=down_factor)(inputs)
        else:  # MAXPOOL
            outputs = tf.keras.layers.MaxPool1D(
                pool_size=down_factor, strides=down_factor)(inputs)
    return outputs


def batchnorm_layer(
        inputs,
        name,
        training,
        batchnorm=constants.BN,
        scale=True,
        reuse=False,
        axis=-1
):
    """Buils a batchnormalization layer.

    Args:
        inputs: (tensor) Input tensor of shape [batch_size, ..., channels] or
            [batch_size, channels, ...].
        name: (string) A name for the operation.
        batchnorm: (Optional, {BN, BN_RENORM}, defaults to BN) Type of batchnorm
            to be used. BN is normal batchnorm, and BN_RENORM is a batchnorm
            with renorm activated.
        reuse: (Optional, boolean, defaults to False) Whether to reuse the layer
            variables.
        training: (Optional, boolean, defaults to False) Indicates if it is the
            training phase or not.
    """
    checks.check_valid_value(
        batchnorm, 'batchnorm', [constants.BN, constants.BN_RENORM, None])
    if batchnorm is None:
        # Bypass
        return inputs
    if batchnorm == constants.BN_RENORM:
        name = '%s_renorm' % name
    with tf.variable_scope(name):
        if batchnorm == constants.BN:
            outputs = tf.layers.batch_normalization(
                inputs=inputs, training=training,
                reuse=reuse, scale=scale, axis=axis)
        else:  # BN_RENORM
            outputs = tf.layers.batch_normalization(
                inputs=inputs, training=training,
                reuse=reuse, renorm=True, scale=scale, axis=axis)
    return outputs


def dropout_layer(
        inputs,
        name,
        training,
        dropout=constants.SEQUENCE_DROP,
        drop_rate=0.5,
        time_major=False):
    """Builds a dropout layer.

    Args:
        inputs: (3d tensor) Input tensor of shape [time_len, batch_size, feats]
            or [batch_size, time_len, feats].
        name: (string) A name for the operation.
        dropout: (Optional, {REGULAR_DROP, SEQUENCE_DROP}, defaults to
            REGULAR_DROP) Type of dropout to be used. REGULAR_DROP is regular
            dropout, and SEQUENCE_DROP is a dropout with the same noise shape
            for each time_step.
        drop_rate: (Optional, float, defaults to 0.5) Dropout rate. Fraction of
            units to be dropped.
        time_major: (Optional, boolean, defaults to False) Indicates if input is
            time major instead of batch major.
        training: (Optional, boolean, defaults to False) Indicates if it is the
            training phase or not.
    """
    checks.check_valid_value(
        dropout, 'dropout', [constants.SEQUENCE_DROP, constants.REGULAR_DROP, None])
    if dropout is None:
        # Bypass
        return inputs
    if dropout == constants.SEQUENCE_DROP:
        name = '%s_seq' % name
    with tf.variable_scope(name):
        if dropout == constants.SEQUENCE_DROP:
            in_shape = tf.shape(inputs)
            if time_major:  # Input has shape [time_len, batch, feats]
                noise_shape = [1, in_shape[1], in_shape[2]]
            else:  # Input has shape [batch, time_len, feats]
                noise_shape = [in_shape[0], 1, in_shape[2]]
            outputs = tf.layers.dropout(
                inputs, training=training, rate=drop_rate,
                noise_shape=noise_shape)
        else:  # REGULAR_DROP
            outputs = tf.layers.dropout(
                inputs, training=training, rate=drop_rate)
    return outputs


def cmorlet_layer(
        inputs,
        fb_list,
        fs,
        lower_freq,
        upper_freq,
        n_scales,
        stride,
        training,
        size_factor=1.0,
        border_crop=0,
        use_avg_pool=True,
        use_log=False,
        batchnorm=None,
        trainable_wavelet=False,
        reuse=False,
        name=None):
    """Builds the operations to compute a CWT with the complex morlet wavelet.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        fb_list: (list of floats) list of values for Fb (one for each scalogram)
        fs: (float) Sampling frequency of the signals of interest
        lower_freq: (float) Lower frequency to be considered for the scalogram.
        upper_freq: (float) Upper frequency to be considered for the scalogram.
        n_scales: (int) Number of scales to cover the frequency range.
        stride: (Optional, int, defaults to 1) The stride of the sliding window
            across the input. Default is 1.
        border_crop: (Optional, int, defaults to 0) Non-negative integer that
            specifies the number of samples to be removed at each border at the
            end. This parameter allows to input a longer signal than the final
            desired size to remove border effects of the CWT.
        use_avg_pool: (Optional, boolean, defaults to True) Whether to compute
            the CWT with stride 1 and then compute an average pooling in the
            time axis with the given stride.
        use_log: (Optional, boolean, defaults to True) whether to apply
            logarithm to the CWT output (after the avg pool if applicable).
        batchnorm: (Optional, {None, BN, BN_RENORM}, defaults to None) Type of
            batchnorm to be used. BN is normal batchnorm, and BN_RENORM is a
            batchnorm with renorm activated. If None, batchnorm is not applied.
            The batchnorm layer is applied after the transformations.
        training: (Optional, boolean, defaults to False) Indicates if it is the
            training phase or not.
        trainable_wavelet: (Optional, boolean, defaults to False) If True, the
            fb params will be trained with backprop.
        reuse: (Optional, boolean, defaults to False) Whether to reuse the layer
            variables.
        name: (Optional, string, defaults to None) A name for the operation.
    """

    with tf.variable_scope(name):
        # Input sequence has shape [batch_size, time_len]
        if use_avg_pool and stride > 1:
            print('Using avg pool after CWT with stride %s' % stride)
            cwt, wavelets = compute_cwt(
                inputs, fb_list, fs, lower_freq, upper_freq, n_scales,
                size_factor=size_factor,
                flattening=False, border_crop=border_crop, stride=1,
                trainable=trainable_wavelet)
            cwt = tf.layers.average_pooling2d(
                inputs=cwt, pool_size=(stride, 1), strides=(stride, 1))
        else:
            cwt, wavelets = compute_cwt(
                inputs, fb_list, fs, lower_freq, upper_freq, n_scales,
                size_factor=size_factor,
                flattening=False, border_crop=border_crop, stride=stride,
                trainable=trainable_wavelet)
        if use_log:
            # Apply log only to magnitude part of cwt
            # Unstack spectrograms
            n_spect = 2 * len(fb_list)
            cwt = tf.unstack(cwt, axis=-1)
            after_log = []
            for k in range(n_spect):
                if k % 2 == 0:  # 0, 2, 4, ... etc, this is magnitude
                    tmp = tf.log(cwt[k] + 1e-8)
                else:  # Angle remains unchanged
                    tmp = cwt[k]
                after_log.append(tmp)
            cwt = tf.stack(after_log, axis=-1)

        cwt_prebn = cwt

        if batchnorm:
            # Unstack spectrograms
            n_spect = 2 * len(fb_list)
            cwt = tf.unstack(cwt, axis=-1)
            after_bn = []
            for k in range(n_spect):
                tmp = batchnorm_layer(
                    cwt[k], 'bn_%d' % k, batchnorm=batchnorm,
                    reuse=reuse, training=training)
                after_bn.append(tmp)
            cwt = tf.stack(after_bn, axis=-1)
        # Output sequence has shape [batch_size, time_len, n_scales, channels]
    return cwt, cwt_prebn


def cmorlet_layer_rectangular(
        inputs,
        fb_list,
        fs,
        lower_freq,
        upper_freq,
        n_scales,
        stride,
        training,
        size_factor=1.0,
        border_crop=0,
        use_avg_pool=True,
        use_log=False,
        batchnorm=None,
        trainable_wavelet=False,
        reuse=False,
        name=None):
    """Builds the operations to compute a CWT with the complex morlet wavelet.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        fb_list: (list of floats) list of values for Fb (one for each scalogram)
        fs: (float) Sampling frequency of the signals of interest
        lower_freq: (float) Lower frequency to be considered for the scalogram.
        upper_freq: (float) Upper frequency to be considered for the scalogram.
        n_scales: (int) Number of scales to cover the frequency range.
        stride: (Optional, int, defaults to 1) The stride of the sliding window
            across the input. Default is 1.
        border_crop: (Optional, int, defaults to 0) Non-negative integer that
            specifies the number of samples to be removed at each border at the
            end. This parameter allows to input a longer signal than the final
            desired size to remove border effects of the CWT.
        use_avg_pool: (Optional, boolean, defaults to True) Whether to compute
            the CWT with stride 1 and then compute an average pooling in the
            time axis with the given stride.
        use_log: (Optional, boolean, defaults to True) whether to apply
            logarithm to the CWT output (after the avg pool if applicable).
        batchnorm: (Optional, {None, BN, BN_RENORM}, defaults to None) Type of
            batchnorm to be used. BN is normal batchnorm, and BN_RENORM is a
            batchnorm with renorm activated. If None, batchnorm is not applied.
            The batchnorm layer is applied after the transformations.
        training: (Optional, boolean, defaults to False) Indicates if it is the
            training phase or not.
        trainable_wavelet: (Optional, boolean, defaults to False) If True, the
            fb params will be trained with backprop.
        reuse: (Optional, boolean, defaults to False) Whether to reuse the layer
            variables.
        name: (Optional, string, defaults to None) A name for the operation.
    """
    print('Using rectangular version of CWT')
    with tf.variable_scope(name):
        # Input sequence has shape [batch_size, time_len]
        if use_avg_pool and stride > 1:
            print('Using avg pool after CWT with stride %s' % stride)
            cwt, wavelets = compute_cwt_rectangular(
                inputs, fb_list, fs, lower_freq, upper_freq, n_scales,
                size_factor=size_factor,
                flattening=False, border_crop=border_crop, stride=1,
                trainable=trainable_wavelet)
            cwt = tf.layers.average_pooling2d(
                inputs=cwt, pool_size=(stride, 1), strides=(stride, 1))
        else:
            cwt, wavelets = compute_cwt_rectangular(
                inputs, fb_list, fs, lower_freq, upper_freq, n_scales,
                size_factor=size_factor,
                flattening=False, border_crop=border_crop, stride=stride,
                trainable=trainable_wavelet)
        if use_log:
            cwt = tf.log(cwt + 1e-8)

        cwt_prebn = cwt

        if batchnorm:
            # Unstack spectrograms
            n_spect = 2 * len(fb_list)
            cwt = tf.unstack(cwt, axis=-1)
            after_bn = []
            for k in range(0, n_spect, 2):
                # BN is shared between real and imaginary parts
                tmp = tf.stack([cwt[k], cwt[k+1]], axis=-1)
                tmp = batchnorm_layer(
                    tmp, 'bn_%d' % k, batchnorm=batchnorm,
                    reuse=reuse, training=training, axis=2)
                after_bn.append(tmp)
            cwt = tf.concat(after_bn, axis=-1)
        # Output sequence has shape [batch_size, time_len, n_scales, channels]
    return cwt, cwt_prebn


def cmorlet_layer_general(
        inputs,
        fb_list,
        fs,
        lower_freq,
        upper_freq,
        n_scales,
        stride,
        training,
        return_real_part=True,
        return_imag_part=False,
        return_magnitude=False,
        return_phase=False,
        size_factor=1.0,
        border_crop=0,
        use_avg_pool=True,
        pool_scales=None,
        batchnorm=None,
        trainable_wavelet=False,
        reuse=False,
        name=None):
    """Builds the operations to compute a CWT with the complex morlet wavelet.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        fb_list: (list of floats) list of values for Fb (one for each scalogram)
        fs: (float) Sampling frequency of the signals of interest
        lower_freq: (float) Lower frequency to be considered for the scalogram.
        upper_freq: (float) Upper frequency to be considered for the scalogram.
        n_scales: (int) Number of scales to cover the frequency range.
        stride: (Optional, int, defaults to 1) The stride of the sliding window
            across the input. Default is 1.
        border_crop: (Optional, int, defaults to 0) Non-negative integer that
            specifies the number of samples to be removed at each border at the
            end. This parameter allows to input a longer signal than the final
            desired size to remove border effects of the CWT.
        use_avg_pool: (Optional, boolean, defaults to True) Whether to compute
            the CWT with stride 1 and then compute an average pooling in the
            time axis with the given stride.
        batchnorm: (Optional, {None, BN, BN_RENORM}, defaults to None) Type of
            batchnorm to be used. BN is normal batchnorm, and BN_RENORM is a
            batchnorm with renorm activated. If None, batchnorm is not applied.
            The batchnorm layer is applied after the transformations.
        training: (Optional, boolean, defaults to False) Indicates if it is the
            training phase or not.
        trainable_wavelet: (Optional, boolean, defaults to False) If True, the
            fb params will be trained with backprop.
        reuse: (Optional, boolean, defaults to False) Whether to reuse the layer
            variables.
        name: (Optional, string, defaults to None) A name for the operation.
    """
    if not (return_imag_part or return_real_part or return_magnitude or return_phase):
        raise ValueError('CWT must return something, but all returns are false')

    print('Using rectangular version of CWT')
    with tf.variable_scope(name):
        # Input sequence has shape [batch_size, time_len]
        if use_avg_pool and stride > 1:
            print('Using avg pool after CWT with stride %s' % stride)
            cwt, wavelets = compute_cwt_rectangular(
                inputs, fb_list, fs, lower_freq, upper_freq, n_scales,
                size_factor=size_factor,
                flattening=False, border_crop=border_crop, stride=1,
                trainable=trainable_wavelet)
            cwt = tf.layers.average_pooling2d(
                inputs=cwt, pool_size=(stride, 1), strides=(stride, 1))
        else:
            cwt, wavelets = compute_cwt_rectangular(
                inputs, fb_list, fs, lower_freq, upper_freq, n_scales,
                size_factor=size_factor,
                flattening=False, border_crop=border_crop, stride=stride,
                trainable=trainable_wavelet)
        if pool_scales is not None:
            cwt = tf.layers.average_pooling2d(
                cwt, pool_size=(1, pool_scales), strides=(1, pool_scales))

        # Output sequence has shape [batch_size, time_len, n_scales, channels]
        cwt_prebn = cwt

        # Unstack spectrograms
        n_spect = 2 * len(fb_list)
        cwt = tf.unstack(cwt, axis=-1)

        total_return = []

        for k in range(0, n_spect, 2):
            real_part = tf.expand_dims(cwt[k], axis=3)
            imag_part = tf.expand_dims(cwt[k+1], axis=3)

            if return_real_part and return_imag_part:
                # BN is shared between real and imaginary parts
                real_imag = tf.concat([real_part, imag_part], axis=-1)
                if batchnorm:
                    real_imag = batchnorm_layer(
                        real_imag, 'bn_real_imag_%d' % k, batchnorm=batchnorm,
                        reuse=reuse, training=training, axis=2)
                total_return.append(real_imag)

            elif return_real_part:
                if batchnorm:
                    real_part_postbn = batchnorm_layer(
                        real_part, 'bn_real_%d' % k, batchnorm=batchnorm,
                        reuse=reuse, training=training, axis=2)
                else:
                    real_part_postbn = real_part
                total_return.append(real_part_postbn)

            elif return_imag_part:
                if batchnorm:
                    imag_part_postbn = batchnorm_layer(
                        imag_part, 'bn_imag_%d' % k, batchnorm=batchnorm,
                        reuse=reuse, training=training, axis=2)
                else:
                    imag_part_postbn = imag_part
                total_return.append(imag_part_postbn)

            if return_magnitude:
                magnitude_part = tf.sqrt(
                    tf.square(real_part) + tf.square(imag_part))
                if batchnorm:
                    magnitude_part = batchnorm_layer(
                        magnitude_part, 'bn_magn_%d' % k, batchnorm=batchnorm,
                        reuse=reuse, training=training, axis=2)
                total_return.append(magnitude_part)

            if return_phase:
                phase_part = tf.atan2(imag_part, real_part)
                if batchnorm:
                    phase_part = batchnorm_layer(
                        phase_part, 'bn_phase_%d' % k, batchnorm=batchnorm,
                        reuse=reuse, training=training, axis=2)
                total_return.append(phase_part)

        cwt = tf.concat(total_return, axis=-1)
        # Output sequence has shape [batch_size, time_len, n_scales, channels]
    return cwt, cwt_prebn


def cmorlet_layer_general_noisy(
        inputs,
        fb_list,
        fs,
        lower_freq,
        upper_freq,
        n_scales,
        stride,
        training,
        noise_intensity,
        return_real_part=True,
        return_imag_part=True,
        return_magnitude=False,
        return_phase=False,
        size_factor=1.0,
        expansion_factor=1.0,
        border_crop=0,
        use_avg_pool=False,
        pool_scales=None,
        batchnorm=None,
        trainable_wavelet=False,
        reuse=False,
        name=None):
    """Builds the operations to compute a CWT with the complex morlet wavelet.

    Args:
        inputs: (2d tensor) input tensor of shape [batch_size, time_len]
        fb_list: (list of floats) list of values for Fb (one for each scalogram)
        fs: (float) Sampling frequency of the signals of interest
        lower_freq: (float) Lower frequency to be considered for the scalogram.
        upper_freq: (float) Upper frequency to be considered for the scalogram.
        n_scales: (int) Number of scales to cover the frequency range.
        stride: (Optional, int, defaults to 1) The stride of the sliding window
            across the input. Default is 1.
        border_crop: (Optional, int, defaults to 0) Non-negative integer that
            specifies the number of samples to be removed at each border at the
            end. This parameter allows to input a longer signal than the final
            desired size to remove border effects of the CWT.
        use_avg_pool: (Optional, boolean, defaults to True) Whether to compute
            the CWT with stride 1 and then compute an average pooling in the
            time axis with the given stride.
        batchnorm: (Optional, {None, BN, BN_RENORM}, defaults to None) Type of
            batchnorm to be used. BN is normal batchnorm, and BN_RENORM is a
            batchnorm with renorm activated. If None, batchnorm is not applied.
            The batchnorm layer is applied after the transformations.
        training: (Optional, boolean, defaults to False) Indicates if it is the
            training phase or not.
        trainable_wavelet: (Optional, boolean, defaults to False) If True, the
            fb params will be trained with backprop.
        reuse: (Optional, boolean, defaults to False) Whether to reuse the layer
            variables.
        name: (Optional, string, defaults to None) A name for the operation.
    """
    if not (return_imag_part or return_real_part or return_magnitude or return_phase):
        raise ValueError('CWT must return something, but all returns are false')

    print('Using rectangular version of CWT (NOISY)')
    with tf.variable_scope(name):
        # Input sequence has shape [batch_size, time_len]
        if use_avg_pool and stride > 1:
            print('Using avg pool after CWT with stride %s' % stride)
            wavelets, _ = compute_wavelets_noisy(
                fb_list=fb_list,
                fs=fs,
                lower_freq=lower_freq,
                upper_freq=upper_freq,
                n_scales=n_scales,
                size_factor=size_factor,
                flattening=False,
                trainable=trainable_wavelet,
                training_flag=training,
                noise_intensity=noise_intensity,
                expansion_factor=expansion_factor,
                name='cmorlet')
            cwt = apply_wavelets_rectangular(
                inputs=inputs,
                wavelets=wavelets,
                border_crop=border_crop,
                stride=1,
                name='cwt')
            cwt = tf.layers.average_pooling2d(
                inputs=cwt, pool_size=(stride, 1), strides=(stride, 1))
        else:
            wavelets, _ = compute_wavelets_noisy(
                fb_list=fb_list,
                fs=fs,
                lower_freq=lower_freq,
                upper_freq=upper_freq,
                n_scales=n_scales,
                size_factor=size_factor,
                flattening=False,
                trainable=trainable_wavelet,
                training_flag=training,
                noise_intensity=noise_intensity,
                expansion_factor=expansion_factor,
                name='cmorlet')
            cwt = apply_wavelets_rectangular(
                inputs=inputs,
                wavelets=wavelets,
                border_crop=border_crop,
                stride=stride,
                name='cwt')
        if pool_scales is not None:
            cwt = tf.layers.average_pooling2d(
                cwt, pool_size=(1, pool_scales), strides=(1, pool_scales))

        # Output sequence has shape [batch_size, time_len, n_scales, channels]
        cwt_prebn = cwt

        # Unstack spectrograms
        n_spect = 2 * len(fb_list)
        cwt = tf.unstack(cwt, axis=-1)

        total_return = []

        for k in range(0, n_spect, 2):
            real_part = tf.expand_dims(cwt[k], axis=3)
            imag_part = tf.expand_dims(cwt[k+1], axis=3)

            if return_real_part and return_imag_part:
                # BN is shared between real and imaginary parts
                real_imag = tf.concat([real_part, imag_part], axis=-1)
                if batchnorm:
                    real_imag = batchnorm_layer(
                        real_imag, 'bn_real_imag_%d' % k, batchnorm=batchnorm,
                        reuse=reuse, training=training, axis=2)
                total_return.append(real_imag)

            elif return_real_part:
                if batchnorm:
                    real_part_postbn = batchnorm_layer(
                        real_part, 'bn_real_%d' % k, batchnorm=batchnorm,
                        reuse=reuse, training=training, axis=2)
                else:
                    real_part_postbn = real_part
                total_return.append(real_part_postbn)

            elif return_imag_part:
                if batchnorm:
                    imag_part_postbn = batchnorm_layer(
                        imag_part, 'bn_imag_%d' % k, batchnorm=batchnorm,
                        reuse=reuse, training=training, axis=2)
                else:
                    imag_part_postbn = imag_part
                total_return.append(imag_part_postbn)

            if return_magnitude:
                magnitude_part = tf.sqrt(
                    tf.square(real_part) + tf.square(imag_part))
                if batchnorm:
                    magnitude_part = batchnorm_layer(
                        magnitude_part, 'bn_magn_%d' % k, batchnorm=batchnorm,
                        reuse=reuse, training=training, axis=2)
                total_return.append(magnitude_part)

            if return_phase:
                phase_part = tf.atan2(imag_part, real_part)
                if batchnorm:
                    phase_part = batchnorm_layer(
                        phase_part, 'bn_phase_%d' % k, batchnorm=batchnorm,
                        reuse=reuse, training=training, axis=2)
                total_return.append(phase_part)

        cwt = tf.concat(total_return, axis=-1)
        # Output sequence has shape [batch_size, time_len, n_scales, channels]
    return cwt, cwt_prebn


def conv2d_layer(
        inputs,
        filters,
        training,
        kernel_size=3,
        padding=constants.PAD_SAME,
        strides=1,
        batchnorm=None,
        activation=None,
        pooling=None,
        reuse=False,
        name=None):
    """Buils a 2d convolutional layer with batch normalization and pooling.

    Args:
         inputs: (4d tensor) input tensor of shape
            [batch_size, height, width, n_channels]
         filters: (int) Number of filters to apply.
         kernel_size: (Optional, int or tuple of int, defaults to 3) Size of
            the kernels.
         padding: (Optional, {PAD_SAME, PAD_VALID}, defaults to PAD_SAME) Type
            of padding for the convolution.
         strides: (Optional, int or tuple of int, defaults to 1) Size of the
            strides of the convolutions.
         batchnorm: (Optional, {None, BN, BN_RENORM}, defaults to None) Type of
            batchnorm to be used. BN is normal batchnorm, and BN_RENORM is a
            batchnorm with renorm activated. If None, batchnorm is not applied.
            The batchnorm layer is applied before convolution.
         activation: (Optional, function, defaults to None) Type of activation
            to be used after convolution. If None, activation is linear.
         pooling: (Optional, {AVGPOOL, MAXPOOL, None}, defaults to None) Type of
            pooling to be used after convolution, which is always of stride 2
            and pool size 2. If None, pooling is not applied.
         training: (Optional, boolean, defaults to False) Indicates if it is the
            training phase or not.
         reuse: (Optional, boolean, defaults to False) Whether to reuse the
            layer variables.
         name: (Optional, string, defaults to None) A name for the operation.
    """
    checks.check_valid_value(
        pooling, 'pooling', [constants.AVGPOOL, constants.MAXPOOL, None])
    checks.check_valid_value(
        padding, 'padding', [constants.PAD_SAME, constants.PAD_VALID])

    with tf.variable_scope(name):
        if batchnorm:
            inputs = batchnorm_layer(
                inputs, 'bn', batchnorm=batchnorm,
                reuse=reuse, training=training)
        outputs = tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=kernel_size,
            activation=activation, padding=padding, strides=strides,
            name='conv', reuse=reuse)
        if pooling:
            if pooling == constants.AVGPOOL:
                outputs = tf.layers.average_pooling2d(
                    inputs=outputs, pool_size=2, strides=2)
            else:  # MAXPOOL
                outputs = tf.layers.max_pooling2d(
                    inputs=outputs, pool_size=2, strides=2)
    return outputs


def pooling2d(inputs, pooling):
    checks.check_valid_value(
        pooling, 'pooling', [constants.AVGPOOL, constants.MAXPOOL, None])
    if pooling:
        if pooling == constants.AVGPOOL:
            outputs = tf.layers.average_pooling2d(
                inputs=inputs, pool_size=2, strides=2)
        else:  # MAXPOOL
            outputs = tf.layers.max_pooling2d(
                inputs=inputs, pool_size=2, strides=2)
    else:
        outputs = inputs
    return outputs


def pooling1d(inputs, pooling):
    # [batch_size, time_len, 1, n_units]
    checks.check_valid_value(
        pooling, 'pooling', [constants.AVGPOOL, constants.MAXPOOL, None])
    if pooling:
        if pooling == constants.AVGPOOL:
            outputs = tf.layers.average_pooling2d(
                inputs=inputs, pool_size=(2, 1), strides=(2, 1))
        else:  # MAXPOOL
            outputs = tf.layers.max_pooling2d(
                inputs=inputs, pool_size=(2, 1), strides=(2, 1))
    else:
        outputs = inputs
    return outputs


def conv2d_residualv2_block(
        inputs,
        filters,
        training,
        is_first_unit=False,
        strides=1,
        batchnorm=None,
        reuse=False,
        kernel_init=None,
        name=None
):
    with tf.variable_scope(name):

        if is_first_unit:
            inputs = tf.layers.conv2d(
                inputs=inputs, filters=filters, kernel_size=5,
                padding=constants.PAD_SAME, kernel_initializer=kernel_init,
                strides=strides, name='conv5_1', reuse=reuse)
            inputs = tf.nn.relu(inputs)
            if batchnorm:
                inputs = batchnorm_layer(
                    inputs, 'bn_1', batchnorm=batchnorm,
                    reuse=reuse, training=training)

            shortcut = inputs

            outputs = tf.layers.conv2d(
                inputs=inputs, filters=filters, kernel_size=3,
                padding=constants.PAD_SAME, kernel_initializer=kernel_init,
                strides=1, name='conv3_1', reuse=reuse)
            outputs = tf.nn.relu(outputs)
            if batchnorm:
                outputs = batchnorm_layer(
                    outputs, 'bn_2', batchnorm=batchnorm,
                    reuse=reuse, training=training)
            outputs = tf.layers.conv2d(
                inputs=outputs, filters=filters, kernel_size=3,
                padding=constants.PAD_SAME, kernel_initializer=kernel_init,
                strides=1, name='conv3_2', reuse=reuse)

            outputs = outputs + shortcut

        else:
            shortcut = inputs

            outputs = tf.nn.relu(inputs)
            if batchnorm:
                outputs = batchnorm_layer(
                    outputs, 'bn_1', batchnorm=batchnorm,
                    reuse=reuse, training=training)
            outputs = tf.layers.conv2d(
                inputs=outputs, filters=filters, kernel_size=3,
                padding=constants.PAD_SAME, kernel_initializer=kernel_init,
                strides=strides, name='conv3_1', reuse=reuse)
            outputs = tf.nn.relu(outputs)
            if batchnorm:
                outputs = batchnorm_layer(
                    outputs, 'bn_2', batchnorm=batchnorm,
                    reuse=reuse, training=training)
            outputs = tf.layers.conv2d(
                inputs=outputs, filters=filters, kernel_size=3,
                padding=constants.PAD_SAME, kernel_initializer=kernel_init,
                strides=1, name='conv3_2', reuse=reuse)

            # Projection if necessary
            input_filters = shortcut.get_shape().as_list()[-1]
            if strides != 1 or input_filters != filters:
                shortcut = tf.layers.conv2d(
                    inputs=shortcut, filters=filters, kernel_size=1,
                    padding=constants.PAD_SAME, use_bias=False,
                    kernel_initializer=kernel_init,
                    strides=strides, name='conv1x1', reuse=reuse)

            outputs = outputs + shortcut

    return outputs


def conv2d_residualv2_prebn_block(
        inputs,
        filters,
        training,
        is_first_unit=False,
        strides=1,
        batchnorm=None,
        reuse=False,
        kernel_init=None,
        name=None
):
    with tf.variable_scope(name):

        if is_first_unit:
            if batchnorm:
                inputs = tf.layers.conv2d(
                    inputs=inputs, filters=filters, kernel_size=5,
                    padding=constants.PAD_SAME,
                    strides=strides, name='conv5_1', reuse=reuse,
                    kernel_initializer=kernel_init,
                    use_bias=False)
                inputs = batchnorm_layer(
                    inputs, 'bn_1', batchnorm=batchnorm,
                    reuse=reuse, training=training, scale=False)
            else:
                inputs = tf.layers.conv2d(
                    inputs=inputs, filters=filters, kernel_size=5,
                    padding=constants.PAD_SAME, kernel_initializer=kernel_init,
                    strides=strides, name='conv5_1', reuse=reuse)
            inputs = tf.nn.relu(inputs)

            shortcut = inputs

            if batchnorm:
                outputs = tf.layers.conv2d(
                    inputs=inputs, filters=filters, kernel_size=3,
                    padding=constants.PAD_SAME,
                    strides=1, name='conv3_1', reuse=reuse,
                    use_bias=False, kernel_initializer=kernel_init,
                )
                outputs = batchnorm_layer(
                    outputs, 'bn_2', batchnorm=batchnorm,
                    reuse=reuse, training=training, scale=False)
                outputs = tf.nn.relu(outputs)
                outputs = tf.layers.conv2d(
                    inputs=outputs, filters=filters, kernel_size=3,
                    padding=constants.PAD_SAME,
                    strides=1, name='conv3_2', reuse=reuse,
                    use_bias=False, kernel_initializer=kernel_init
                )
            else:
                outputs = tf.layers.conv2d(
                    inputs=inputs, filters=filters, kernel_size=3,
                    padding=constants.PAD_SAME, kernel_initializer=kernel_init,
                    strides=1, name='conv3_1', reuse=reuse)
                outputs = tf.nn.relu(outputs)
                outputs = tf.layers.conv2d(
                    inputs=outputs, filters=filters, kernel_size=3,
                    padding=constants.PAD_SAME, kernel_initializer=kernel_init,
                    strides=1, name='conv3_2', reuse=reuse)

            outputs = outputs + shortcut

        else:
            shortcut = inputs

            if batchnorm:
                outputs = batchnorm_layer(
                    inputs, 'bn_1', batchnorm=batchnorm,
                    reuse=reuse, training=training, scale=False)
                outputs = tf.nn.relu(outputs)
                outputs = tf.layers.conv2d(
                    inputs=outputs, filters=filters, kernel_size=3,
                    padding=constants.PAD_SAME, kernel_initializer=kernel_init,
                    strides=strides, name='conv3_1', reuse=reuse,
                    use_bias=False)
                outputs = batchnorm_layer(
                    outputs, 'bn_2', batchnorm=batchnorm,
                    reuse=reuse, training=training, scale=False)
                outputs = tf.nn.relu(outputs)
                outputs = tf.layers.conv2d(
                    inputs=outputs, filters=filters, kernel_size=3,
                    padding=constants.PAD_SAME, kernel_initializer=kernel_init,
                    strides=1, name='conv3_2', reuse=reuse, use_bias=False)
            else:
                outputs = tf.nn.relu(inputs)
                outputs = tf.layers.conv2d(
                    inputs=outputs, filters=filters, kernel_size=3,
                    padding=constants.PAD_SAME, kernel_initializer=kernel_init,
                    strides=strides, name='conv3_1', reuse=reuse)
                outputs = tf.nn.relu(outputs)
                outputs = tf.layers.conv2d(
                    inputs=outputs, filters=filters, kernel_size=3,
                    padding=constants.PAD_SAME, kernel_initializer=kernel_init,
                    strides=1, name='conv3_2', reuse=reuse)

            # Projection if necessary
            input_filters = shortcut.get_shape().as_list()[-1]
            if strides != 1 or input_filters != filters:
                shortcut = tf.layers.conv2d(
                    inputs=shortcut, filters=filters, kernel_size=1,
                    padding=constants.PAD_SAME, use_bias=False,
                    kernel_initializer=kernel_init,
                    strides=strides, name='conv1x1', reuse=reuse)

            outputs = outputs + shortcut

    return outputs


def conv2d_prebn_block(
        inputs,
        filters,
        training,
        kernel_size_1=3,
        kernel_size_2=3,
        batchnorm=None,
        downsampling=constants.MAXPOOL,
        reuse=False,
        kernel_init=None,
        name=None
):
    checks.check_valid_value(
        downsampling, 'downsampling',
        [constants.AVGPOOL, constants.MAXPOOL, constants.STRIDEDCONV, None])

    if downsampling == constants.STRIDEDCONV:
        strides = 2
        pooling = None
    else:
        strides = 1
        pooling = downsampling

    with tf.variable_scope(name):

        if batchnorm:
            outputs = tf.layers.conv2d(
                inputs=inputs, filters=filters, kernel_size=kernel_size_1,
                padding=constants.PAD_SAME,
                strides=strides, name='conv%d_1' % kernel_size_1, reuse=reuse,
                kernel_initializer=kernel_init,
                use_bias=False)
            outputs = batchnorm_layer(
                outputs, 'bn_1', batchnorm=batchnorm,
                reuse=reuse, training=training, scale=False)
            outputs = tf.nn.relu(outputs)

            outputs = tf.layers.conv2d(
                inputs=outputs, filters=filters, kernel_size=kernel_size_2,
                padding=constants.PAD_SAME,
                strides=1, name='conv%d_2' % kernel_size_2, reuse=reuse,
                kernel_initializer=kernel_init,
                use_bias=False)
            outputs = batchnorm_layer(
                outputs, 'bn_2', batchnorm=batchnorm,
                reuse=reuse, training=training, scale=False)
            outputs = tf.nn.relu(outputs)

            outputs = pooling2d(outputs, pooling)

        else:
            outputs = tf.layers.conv2d(
                inputs=inputs, filters=filters, kernel_size=kernel_size_1,
                padding=constants.PAD_SAME,
                strides=strides, name='conv%d_1' % kernel_size_1, reuse=reuse,
                kernel_initializer=kernel_init)
            outputs = tf.nn.relu(outputs)

            outputs = tf.layers.conv2d(
                inputs=outputs, filters=filters, kernel_size=kernel_size_2,
                padding=constants.PAD_SAME,
                strides=1, name='conv%d_2' % kernel_size_2, reuse=reuse,
                kernel_initializer=kernel_init)
            outputs = tf.nn.relu(outputs)

            outputs = pooling2d(outputs, pooling)
    return outputs


# def conv1d_prebn(
#         inputs,
#         filters,
#         training,
#         kernel_size=3,
#         batchnorm=None,
#         downsampling=constants.MAXPOOL,
#         reuse=False,
#         kernel_init=None,
#         name=None
# ):
#     checks.check_valid_value(
#         downsampling, 'downsampling',
#         [constants.AVGPOOL, constants.MAXPOOL, constants.STRIDEDCONV, None])
#
#     if downsampling == constants.STRIDEDCONV:
#         strides = 2
#         pooling = None
#     else:
#         strides = 1
#         pooling = downsampling
#
#     with tf.variable_scope(name):
#
#         # [batch_size, time_len, n_feats] -> [batch_size, time_len, 1, feats]
#         inputs = tf.expand_dims(inputs, axis=2)
#
#         if batchnorm:
#             outputs = tf.layers.conv2d(
#                 inputs=inputs, filters=filters, kernel_size=(kernel_size, 1),
#                 padding=constants.PAD_SAME,
#                 strides=(strides, 1), name='conv%d' % kernel_size, reuse=reuse,
#                 kernel_initializer=kernel_init,
#                 use_bias=False)
#             outputs = batchnorm_layer(
#                 outputs, 'bn', batchnorm=batchnorm,
#                 reuse=reuse, training=training, scale=False)
#             outputs = tf.nn.relu(outputs)
#         else:
#             outputs = tf.layers.conv2d(
#                 inputs=inputs, filters=filters, kernel_size=(kernel_size, 1),
#                 padding=constants.PAD_SAME,
#                 strides=(strides, 1), name='conv%d' % kernel_size, reuse=reuse,
#                 kernel_initializer=kernel_init)
#             outputs = tf.nn.relu(outputs)
#
#         outputs = pooling1d(outputs, pooling)
#
#         # [batch_size, time_len, 1, n_units] -> [batch_size, time_len, n_units]
#         outputs = tf.squeeze(outputs, axis=2, name="squeeze")
#     return outputs


def conv1d_prebn_block_with_dilation(
        inputs,
        filters,
        training,
        dilation=1,
        kernel_size=3,
        dropout=None,
        drop_rate=0,
        batchnorm=None,
        downsampling=constants.MAXPOOL,
        reuse=False,
        kernel_init=None,
        activation_fn=tf.nn.relu,
        use_scale_at_bn=False,
        name=None
):
    checks.check_valid_value(
        downsampling, 'downsampling',
        [constants.AVGPOOL, constants.MAXPOOL, constants.STRIDEDCONV, None])

    if downsampling == constants.STRIDEDCONV:
        strides = 2
        pooling = None
    else:
        strides = 1
        pooling = downsampling

    with tf.variable_scope(name):

        # [batch_size, time_len, n_feats] -> [batch_size, time_len, 1, feats]
        inputs = tf.expand_dims(inputs, axis=2)
        use_bias = batchnorm is None

        # First convolution
        outputs = tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=(kernel_size, 1),
            padding=constants.PAD_SAME, dilation_rate=dilation,
            strides=(strides, 1), name='conv%d_d%d_1' % (kernel_size, dilation), reuse=reuse,
            kernel_initializer=kernel_init,
            use_bias=use_bias)
        if batchnorm:
            outputs = batchnorm_layer(
                outputs, 'bn_1', batchnorm=batchnorm,
                reuse=reuse, training=training, scale=use_scale_at_bn)
        outputs = activation_fn(outputs)

        # Optional dropout between convolutions
        if dropout:
            outputs = tf.squeeze(outputs, axis=2, name="squeeze")
            outputs = dropout_layer(
                outputs, 'drop', drop_rate=drop_rate, dropout=dropout,
                training=training)
            outputs = tf.expand_dims(outputs, axis=2)

        # Second convolution
        outputs = tf.layers.conv2d(
            inputs=outputs, filters=filters, kernel_size=(kernel_size, 1),
            padding=constants.PAD_SAME, dilation_rate=dilation,
            strides=1, name='conv%d_d%d_2' % (kernel_size, dilation), reuse=reuse,
            kernel_initializer=kernel_init,
            use_bias=use_bias)
        if batchnorm:
            outputs = batchnorm_layer(
                outputs, 'bn_2', batchnorm=batchnorm,
                reuse=reuse, training=training, scale=use_scale_at_bn)
        outputs = activation_fn(outputs)

        # Pooling
        outputs = pooling1d(outputs, pooling)

        # [batch_size, time_len, 1, n_units] -> [batch_size, time_len, n_units]
        outputs = tf.squeeze(outputs, axis=2, name="squeeze")
    return outputs


def conv1d_prebn_block_with_projection(
        inputs,
        filters,
        training,
        kernel_size=3,
        project_first=False,
        dropout=None,
        drop_rate=0,
        batchnorm=None,
        downsampling=constants.MAXPOOL,
        reuse=False,
        kernel_init=None,
        name=None
):
    checks.check_valid_value(
        downsampling, 'downsampling',
        [constants.AVGPOOL, constants.MAXPOOL, constants.STRIDEDCONV, None])

    if downsampling == constants.STRIDEDCONV:
        strides = 2
        pooling = None
    else:
        strides = 1
        pooling = downsampling

    with tf.variable_scope(name):

        # [batch_size, time_len, n_feats] -> [batch_size, time_len, 1, feats]
        inputs = tf.expand_dims(inputs, axis=2)
        use_bias = batchnorm is None

        input_channels = inputs.get_shape().as_list()[-1]
        # Only project if input dim > than filters
        if project_first and (input_channels > filters):
            # Linear projection to n filters
            inputs = tf.layers.conv2d(
                inputs=inputs, filters=filters, kernel_size=(1, 1),
                padding=constants.PAD_SAME,
                strides=(1, 1), name='conv1', reuse=reuse,
                kernel_initializer=kernel_init,
                use_bias=False)

        # First convolution
        outputs = tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=(kernel_size, 1),
            padding=constants.PAD_SAME,
            strides=(strides, 1), name='conv%d_1' % kernel_size, reuse=reuse,
            kernel_initializer=kernel_init,
            use_bias=use_bias)
        if batchnorm:
            outputs = batchnorm_layer(
                outputs, 'bn_1', batchnorm=batchnorm,
                reuse=reuse, training=training, scale=False)
        outputs = tf.nn.relu(outputs)

        # Optional dropout between convolutions
        if dropout:
            outputs = tf.squeeze(outputs, axis=2, name="squeeze")
            outputs = dropout_layer(
                outputs, 'drop', drop_rate=drop_rate, dropout=dropout,
                training=training)
            outputs = tf.expand_dims(outputs, axis=2)

        # Second convolution
        outputs = tf.layers.conv2d(
            inputs=outputs, filters=filters, kernel_size=(kernel_size, 1),
            padding=constants.PAD_SAME,
            strides=1, name='conv%d_2' % kernel_size, reuse=reuse,
            kernel_initializer=kernel_init,
            use_bias=use_bias)
        if batchnorm:
            outputs = batchnorm_layer(
                outputs, 'bn_2', batchnorm=batchnorm,
                reuse=reuse, training=training, scale=False)
        outputs = tf.nn.relu(outputs)

        # Pooling
        outputs = pooling1d(outputs, pooling)

        # [batch_size, time_len, 1, n_units] -> [batch_size, time_len, n_units]
        outputs = tf.squeeze(outputs, axis=2, name="squeeze")
    return outputs


def conv1d_prebn_block_with_zscore(
        inputs,
        filters,
        training,
        kernel_size_1=3,
        kernel_size_2=3,
        batchnorm=None,
        downsampling=constants.MAXPOOL,
        reuse=False,
        kernel_init=None,
        name=None
):
    checks.check_valid_value(
        downsampling, 'downsampling',
        [constants.AVGPOOL, constants.MAXPOOL, constants.STRIDEDCONV, None])

    if downsampling == constants.STRIDEDCONV:
        strides = 2
        pooling = None
    else:
        strides = 1
        pooling = downsampling

    with tf.variable_scope(name):

        # [batch_size, time_len, n_feats] -> [batch_size, time_len, 1, feats]
        inputs = tf.expand_dims(inputs, axis=2)

        use_bias = batchnorm is None

        outputs = tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=(kernel_size_1, 1),
            padding=constants.PAD_SAME,
            strides=(strides, 1), name='conv%d_1' % kernel_size_1, reuse=reuse,
            kernel_initializer=kernel_init,
            use_bias=use_bias)
        if batchnorm:
            outputs = batchnorm_layer(
                outputs, 'bn_1', batchnorm=batchnorm,
                reuse=reuse, training=training, scale=False)
        outputs = tf.nn.relu(outputs)

        outputs = tf.layers.conv2d(
            inputs=outputs, filters=filters, kernel_size=(kernel_size_2, 1),
            padding=constants.PAD_SAME,
            strides=1, name='conv%d_2' % kernel_size_2, reuse=reuse,
            kernel_initializer=kernel_init,
            use_bias=use_bias)

        with tf.variable_scope("zscore"):
            outputs_1 = outputs[..., :filters//2]  # Keep as-is
            outputs_2 = outputs[..., filters//2:]  # Transform to z-score

            outputs_2_mean = tf.reduce_mean(outputs_2, keepdims=True, axis=1)
            outputs_2 = outputs_2 - outputs_2_mean
            outputs_2_var = tf.reduce_mean(outputs_2 ** 2, keepdims=True, axis=1)
            outputs_2 = outputs_2 / tf.math.sqrt(outputs_2_var + 1e-4)

            # Now join again
            outputs = tf.concat([outputs_1, outputs_2], axis=-1)

        if batchnorm:
            outputs = batchnorm_layer(
                outputs, 'bn_2', batchnorm=batchnorm,
                reuse=reuse, training=training, scale=False)
        outputs = tf.nn.relu(outputs)

        outputs = pooling1d(outputs, pooling)

        # [batch_size, time_len, 1, n_units] -> [batch_size, time_len, n_units]
        outputs = tf.squeeze(outputs, axis=2, name="squeeze")
    return outputs


def conv1d_prebn_block(
        inputs,
        filters,
        training,
        kernel_size_1=3,
        kernel_size_2=3,
        batchnorm=None,
        downsampling=constants.MAXPOOL,
        reuse=False,
        kernel_init=None,
        name=None
):
    checks.check_valid_value(
        downsampling, 'downsampling',
        [constants.AVGPOOL, constants.MAXPOOL, constants.STRIDEDCONV, None])

    if downsampling == constants.STRIDEDCONV:
        strides = 2
        pooling = None
    else:
        strides = 1
        pooling = downsampling

    with tf.variable_scope(name):

        # [batch_size, time_len, n_feats] -> [batch_size, time_len, 1, feats]
        inputs = tf.expand_dims(inputs, axis=2)

        if batchnorm:
            outputs = tf.layers.conv2d(
                inputs=inputs, filters=filters, kernel_size=(kernel_size_1, 1),
                padding=constants.PAD_SAME,
                strides=(strides, 1), name='conv%d_1' % kernel_size_1, reuse=reuse,
                kernel_initializer=kernel_init,
                use_bias=False)
            outputs = batchnorm_layer(
                outputs, 'bn_1', batchnorm=batchnorm,
                reuse=reuse, training=training, scale=False)
            outputs = tf.nn.relu(outputs)

            outputs = tf.layers.conv2d(
                inputs=outputs, filters=filters, kernel_size=(kernel_size_2, 1),
                padding=constants.PAD_SAME,
                strides=1, name='conv%d_2' % kernel_size_2, reuse=reuse,
                kernel_initializer=kernel_init,
                use_bias=False)
            outputs = batchnorm_layer(
                outputs, 'bn_2', batchnorm=batchnorm,
                reuse=reuse, training=training, scale=False)
            outputs = tf.nn.relu(outputs)

        else:
            outputs = tf.layers.conv2d(
                inputs=inputs, filters=filters, kernel_size=(kernel_size_1, 1),
                padding=constants.PAD_SAME,
                strides=(strides, 1), name='conv%d_1' % kernel_size_1, reuse=reuse,
                kernel_initializer=kernel_init)
            outputs = tf.nn.relu(outputs)

            outputs = tf.layers.conv2d(
                inputs=outputs, filters=filters, kernel_size=(kernel_size_2, 1),
                padding=constants.PAD_SAME,
                strides=1, name='conv%d_2' % kernel_size_2, reuse=reuse,
                kernel_initializer=kernel_init)
            outputs = tf.nn.relu(outputs)

        outputs = pooling1d(outputs, pooling)

        # [batch_size, time_len, 1, n_units] -> [batch_size, time_len, n_units]
        outputs = tf.squeeze(outputs, axis=2, name="squeeze")
    return outputs


def conv1d_prebn_block_with_context(
        inputs,
        context,
        filters,
        training,
        kernel_size_1=3,
        kernel_size_2=3,
        batchnorm=None,
        downsampling=constants.MAXPOOL,
        reuse=False,
        kernel_init=None,
        name=None
):
    checks.check_valid_value(
        downsampling, 'downsampling',
        [constants.AVGPOOL, constants.MAXPOOL, constants.STRIDEDCONV, None])

    if downsampling == constants.STRIDEDCONV:
        strides = 2
        pooling = None
    else:
        strides = 1
        pooling = downsampling

    with tf.variable_scope(name):

        # Context vectors
        context_conv1 = tf.keras.layers.Dense(
            filters, use_bias=False, name="context1")(context)  # [batch, filters]
        context_conv1 = tf.expand_dims(context_conv1, axis=1)  # [batch, 1, filters]
        context_conv1 = tf.expand_dims(context_conv1, axis=1)  # [batch, 1, 1, filters]

        context_conv2 = tf.keras.layers.Dense(
            filters, use_bias=False, name="context2")(context)  # [batch, filters]
        context_conv2 = tf.expand_dims(context_conv2, axis=1)  # [batch, 1, filters]
        context_conv2 = tf.expand_dims(context_conv2, axis=1)  # [batch, 1, 1, filters]

        # [batch_size, time_len, n_feats] -> [batch_size, time_len, 1, feats]
        inputs = tf.expand_dims(inputs, axis=2)

        if batchnorm:
            outputs = tf.layers.conv2d(
                inputs=inputs, filters=filters, kernel_size=(kernel_size_1, 1),
                padding=constants.PAD_SAME,
                strides=(strides, 1), name='conv%d_1' % kernel_size_1, reuse=reuse,
                kernel_initializer=kernel_init,
                use_bias=False)
            outputs = outputs + context_conv1
            outputs = batchnorm_layer(
                outputs, 'bn_1', batchnorm=batchnorm,
                reuse=reuse, training=training, scale=False)
            outputs = tf.nn.relu(outputs)

            outputs = tf.layers.conv2d(
                inputs=outputs, filters=filters, kernel_size=(kernel_size_2, 1),
                padding=constants.PAD_SAME,
                strides=1, name='conv%d_2' % kernel_size_2, reuse=reuse,
                kernel_initializer=kernel_init,
                use_bias=False)
            outputs = outputs + context_conv2
            outputs = batchnorm_layer(
                outputs, 'bn_2', batchnorm=batchnorm,
                reuse=reuse, training=training, scale=False)
            outputs = tf.nn.relu(outputs)

        else:
            outputs = tf.layers.conv2d(
                inputs=inputs, filters=filters, kernel_size=(kernel_size_1, 1),
                padding=constants.PAD_SAME,
                strides=(strides, 1), name='conv%d_1' % kernel_size_1, reuse=reuse,
                kernel_initializer=kernel_init)
            outputs = outputs + context_conv1
            outputs = tf.nn.relu(outputs)

            outputs = tf.layers.conv2d(
                inputs=outputs, filters=filters, kernel_size=(kernel_size_2, 1),
                padding=constants.PAD_SAME,
                strides=1, name='conv%d_2' % kernel_size_2, reuse=reuse,
                kernel_initializer=kernel_init)
            outputs = outputs + context_conv2
            outputs = tf.nn.relu(outputs)

        outputs = pooling1d(outputs, pooling)

        # [batch_size, time_len, 1, n_units] -> [batch_size, time_len, n_units]
        outputs = tf.squeeze(outputs, axis=2, name="squeeze")
    return outputs


def sequence_fc_layer_with_context(
        inputs,
        context,
        num_units,
        training,
        batchnorm=None,
        dropout=None,
        drop_rate=0,
        activation=None,
        kernel_init=None,
        reuse=False,
        name=None):
    """ Builds a FC layer that can be applied directly to a sequence.

    Each time-step is passed through to the same FC layer.

    Args:
        inputs: (3d tensor) input tensor of shape
            [batch_size, time_len, n_feats].
        num_units: (int) Number of neurons for the FC layer.
        batchnorm: (Optional, {None, BN, BN_RENORM}, defaults to None) Type of
            batchnorm to be used. BN is normal batchnorm, and BN_RENORM is a
            batchnorm with renorm activated. If None, batchnorm is not applied.
            The batchnorm layer is applied before the fc layer.
        dropout: (Optional, {None REGULAR_DROP, SEQUENCE_DROP}, defaults to
            None) Type of dropout to be used. REGULAR_DROP is regular
            dropout, and SEQUENCE_DROP is a dropout with the same noise shape
            for each time_step. If None, dropout is not applied. The
            dropout layer is applied before the fc layer, after the batchnorm.
        drop_rate: (Optional, float, defaults to 0.5) Dropout rate. Fraction of
            units to be dropped. If dropout is None, this is ignored.
        activation: (Optional, function, defaults to None) Type of activation
            to be used at the output. If None, activation is linear.
        training: (Optional, boolean, defaults to False) Indicates if it is the
            training phase or not
        reuse: (Optional, boolean, defaults to False) Whether to reuse the layer
            variables.
        name: (Optional, string, defaults to None) A name for the operation.
    """
    with tf.variable_scope(name):

        context_conv1 = tf.keras.layers.Dense(
            num_units, use_bias=False, name="context1")(context)
        context_conv1 = tf.expand_dims(context_conv1, axis=1)
        context_conv1 = tf.expand_dims(context_conv1, axis=1)

        if batchnorm:
            inputs = batchnorm_layer(
                inputs, 'bn', batchnorm=batchnorm, reuse=reuse,
                training=training)
        if dropout:
            inputs = dropout_layer(
                inputs, 'drop', drop_rate=drop_rate, dropout=dropout,
                training=training)
        # [batch_size, time_len, n_feats] -> [batch_size, time_len, 1, feats]
        inputs = tf.expand_dims(inputs, axis=2)
        outputs = tf.layers.conv2d(
            inputs=inputs, filters=num_units, kernel_size=1,
            padding=constants.PAD_SAME,
            kernel_initializer=kernel_init,
            name="conv1", reuse=reuse)
        outputs = outputs + context_conv1
        outputs = activation(outputs)
        # [batch_size, time_len, 1, n_units] -> [batch_size, time_len, n_units]
        outputs = tf.squeeze(outputs, axis=2, name="squeeze")
    return outputs


def sequence_output_2class_layer_with_context(
        inputs,
        context,
        training,
        batchnorm=None,
        dropout=None,
        drop_rate=0,
        activation=None,
        kernel_init=None,
        init_positive_proba=0.5,
        reuse=False,
        name=None):
    """ Builds a FC layer that can be applied directly to a sequence.

    Each time-step is passed through to the same FC layer.

    Args:
        inputs: (3d tensor) input tensor of shape
            [batch_size, time_len, n_feats].
        num_units: (int) Number of neurons for the FC layer.
        batchnorm: (Optional, {None, BN, BN_RENORM}, defaults to None) Type of
            batchnorm to be used. BN is normal batchnorm, and BN_RENORM is a
            batchnorm with renorm activated. If None, batchnorm is not applied.
            The batchnorm layer is applied before the fc layer.
        dropout: (Optional, {None REGULAR_DROP, SEQUENCE_DROP}, defaults to
            None) Type of dropout to be used. REGULAR_DROP is regular
            dropout, and SEQUENCE_DROP is a dropout with the same noise shape
            for each time_step. If None, dropout is not applied. The
            dropout layer is applied before the fc layer, after the batchnorm.
        drop_rate: (Optional, float, defaults to 0.5) Dropout rate. Fraction of
            units to be dropped. If dropout is None, this is ignored.
        activation: (Optional, function, defaults to None) Type of activation
            to be used at the output. If None, activation is linear.
        training: (Optional, boolean, defaults to False) Indicates if it is the
            training phase or not
        reuse: (Optional, boolean, defaults to False) Whether to reuse the layer
            variables.
        name: (Optional, string, defaults to None) A name for the operation.
    """
    with tf.variable_scope(name):

        context_conv1 = tf.keras.layers.Dense(
            2, use_bias=False, name="context1")(context)
        context_conv1 = tf.expand_dims(context_conv1, axis=1)
        context_conv1 = tf.expand_dims(context_conv1, axis=1)

        if batchnorm:
            inputs = batchnorm_layer(
                inputs, 'bn', batchnorm=batchnorm, reuse=reuse,
                training=training)
        if dropout:
            inputs = dropout_layer(
                inputs, 'drop', drop_rate=drop_rate, dropout=dropout,
                training=training)
        # [batch_size, time_len, n_feats] -> [batch_size, time_len, 1, feats]
        inputs = tf.expand_dims(inputs, axis=2)

        outputs_0 = tf.layers.conv2d(
            inputs=inputs, filters=1, kernel_size=1,
            padding=constants.PAD_SAME,
            kernel_initializer=kernel_init,
            name="conv1_0", reuse=reuse)

        bias_init = - np.log((1 - init_positive_proba) / init_positive_proba)
        print('Initializing bias as %1.4f, to have init positive proba of %1.4f'
              % (bias_init, init_positive_proba))

        outputs_1 = tf.layers.conv2d(
            inputs=inputs, filters=1, kernel_size=1,
            padding=constants.PAD_SAME,
            kernel_initializer=kernel_init,
            bias_initializer=tf.constant_initializer(value=bias_init),
            name="conv1_1", reuse=reuse)

        outputs = tf.concat([outputs_0, outputs_1], axis=3)
        outputs = outputs + context_conv1
        if activation is not None:
            outputs = activation(outputs)

        # [batch_size, time_len, 1, n_units] -> [batch_size, time_len, n_units]
        outputs = tf.squeeze(outputs, axis=2, name="squeeze")
    return outputs


def conv1d_prebn_block_unet_down(
        inputs,
        filters,
        training,
        n_layers=2,
        kernel_size=3,
        batchnorm=None,
        downsampling=constants.MAXPOOL,
        reuse=False,
        kernel_init=None,
        name=None
):
    checks.check_valid_value(
        downsampling, 'downsampling',
        [constants.AVGPOOL, constants.MAXPOOL, None])

    if batchnorm:
        use_bias = False
    else:
        use_bias = True

    with tf.variable_scope(name):

        # [batch_size, time_len, n_feats] -> [batch_size, time_len, 1, feats]
        outputs = tf.expand_dims(inputs, axis=2)

        for i in range(n_layers):
            outputs = tf.layers.conv2d(
                inputs=outputs, filters=filters, kernel_size=(kernel_size, 1),
                padding=constants.PAD_SAME,
                strides=1, name='conv%d_%d' % (kernel_size, i), reuse=reuse,
                kernel_initializer=kernel_init,
                use_bias=use_bias)
            outputs = batchnorm_layer(
                outputs, 'bn_%d' % i, batchnorm=batchnorm,
                reuse=reuse, training=training, scale=False)
            outputs = tf.nn.relu(outputs)

        outputs_prepool = outputs
        outputs = pooling1d(outputs_prepool, downsampling)

        # [batch_size, time_len, 1, n_units] -> [batch_size, time_len, n_units]
        outputs_prepool = tf.squeeze(outputs_prepool, axis=2, name="squeeze")
        outputs = tf.squeeze(outputs, axis=2, name="squeeze")
    return outputs, outputs_prepool


def conv1d_prebn_block_unet_up(
        inputs,
        inputs_skip_prepool,
        filters,
        training,
        n_layers=2,
        kernel_size=3,
        batchnorm=None,
        reuse=False,
        kernel_init=None,
        name=None
):

    if batchnorm:
        use_bias = False
    else:
        use_bias = True

    with tf.variable_scope(name):
        # Up-sample feature map before concatenation
        inputs = time_upsampling_layer(
            inputs, filters, name='up')
        # Concatenate skip connection with up-sampled current input
        outputs = tf.concat([inputs, inputs_skip_prepool], axis=2)
        # [batch_size, time_len, n_feats] -> [batch_size, time_len, 1, feats]
        outputs = tf.expand_dims(outputs, axis=2)

        for i in range(n_layers):
            outputs = tf.layers.conv2d(
                inputs=outputs, filters=filters, kernel_size=(kernel_size, 1),
                padding=constants.PAD_SAME,
                strides=1, name='conv%d_%d' % (kernel_size, i), reuse=reuse,
                kernel_initializer=kernel_init,
                use_bias=use_bias)
            outputs = batchnorm_layer(
                outputs, 'bn_%d' % i, batchnorm=batchnorm,
                reuse=reuse, training=training, scale=False)
            outputs = tf.nn.relu(outputs)

        # [batch_size, time_len, 1, n_units] -> [batch_size, time_len, n_units]
        outputs = tf.squeeze(outputs, axis=2, name="squeeze")
    return outputs


def upconv1d_prebn(
        inputs,
        filters,
        training,
        kernel_size=3,
        stride=2,
        batchnorm=None,
        reuse=False,
        kernel_init=None,
        name=None
):
    """Performs an upsampling by a factor of stride, using UpConv.
    """
    with tf.variable_scope(name):
        # [batch_size, time_len, n_feats] -> [batch_size, time_len, 1, feats]
        inputs = tf.expand_dims(inputs, axis=2)

        if batchnorm:
            outputs = tf.layers.conv2d_transpose(
                inputs=inputs, filters=filters, kernel_size=(kernel_size, 1),
                strides=(stride, 1), padding=constants.PAD_SAME,
                name='upconv%d' % kernel_size, reuse=reuse,
                kernel_initializer=kernel_init,
                use_bias=False)
            outputs = batchnorm_layer(
                outputs, 'bn_1', batchnorm=batchnorm,
                reuse=reuse, training=training, scale=False)
            outputs = tf.nn.relu(outputs)

        else:
            outputs = tf.layers.conv2d_transpose(
                inputs=inputs, filters=filters, kernel_size=(kernel_size, 1),
                strides=(stride, 1), padding=constants.PAD_SAME,
                name='upconv%d' % kernel_size, reuse=reuse,
                kernel_initializer=kernel_init)
            outputs = tf.nn.relu(outputs)
        outputs = tf.squeeze(outputs, axis=2, name="squeeze")
    return outputs


# def bn_conv3_block(
#         inputs,
#         filters,
#         batchnorm=constants.BN,
#         pooling=constants.MAXPOOL,
#         residual=False,
#         training=False,
#         reuse=False,
#         name=None):
#     """Builds a convolutional block.
#      The block consists of BN-CONV-ReLU-BN-CONV-ReLU-POOL with 3x3 kernels and
#      same number of filters. Please see the documentation of conv2d_layer
#      for details on input parameters.
#      """
#     with tf.variable_scope(name):
#         outputs = conv2d_layer(
#             inputs, filters, batchnorm=batchnorm, activation=tf.nn.relu,
#             training=training, reuse=reuse,
#             name='conv3_1')
#         outputs = conv2d_layer(
#             outputs, filters, batchnorm=batchnorm, activation=None,
#             pooling=pooling, training=training, reuse=reuse,
#             name='conv3_2')
#         if residual:
#             projected_inputs = conv2d_layer(
#                 inputs, filters, kernel_size=1, strides=2,
#                 training=training, reuse=reuse, name='conv1x1')
#             outputs = outputs + projected_inputs
#
#         outputs = tf.nn.relu(outputs)
#
#     return outputs


def flatten(inputs, name=None):
    """ Flattens [batch_size, d0, ..., dn] to [batch_size, d0*...*dn]"""
    with tf.name_scope(name):
        dims = inputs.get_shape().as_list()
        feat_dim = np.prod(dims[1:])
        outputs = tf.reshape(inputs, shape=(-1, feat_dim))
    return outputs


def sequence_flatten(inputs, name=None):
    """ Flattens [batch_size, time_len, d0, ..., dn] to
    [batch_size, time_len, d0*...*dn]"""
    with tf.name_scope(name):
        dims = inputs.get_shape().as_list()
        feat_dim = np.prod(dims[2:])
        outputs = tf.reshape(inputs, shape=(-1, dims[1], feat_dim))
    return outputs


# def sequence_unflatten(inputs, n_channels, name=None):
#     """ Unflattens [batch_size, time_len, width*channels] to
#     [batch_size, time_len, width, channels]"""
#     with tf.name_scope(name):
#         dims = inputs.get_shape().as_list()
#         width_dim = dims[2] // n_channels
#         outputs = tf.reshape(inputs, shape=(-1, dims[1], width_dim, n_channels))
#     return outputs


def swap_batch_time(inputs, name=None):
    """Interchange batch axis with time axis of a 3D tensor, which are assumed
    to be on the first and second axis."""
    with tf.name_scope(name):
        outputs = tf.transpose(inputs, (1, 0, 2))
    return outputs


def sequence_fc_layer_with_zscore(
        inputs,
        num_units,
        training,
        batchnorm=None,
        dropout=None,
        drop_rate=0,
        kernel_init=None,
        reuse=False,
        name=None):
    """ Builds a FC layer that can be applied directly to a sequence.

    Each time-step is passed through to the same FC layer.

    Args:
        inputs: (3d tensor) input tensor of shape
            [batch_size, time_len, n_feats].
        num_units: (int) Number of neurons for the FC layer.
        batchnorm: (Optional, {None, BN, BN_RENORM}, defaults to None) Type of
            batchnorm to be used. BN is normal batchnorm, and BN_RENORM is a
            batchnorm with renorm activated. If None, batchnorm is not applied.
            The batchnorm layer is applied before the fc layer.
        dropout: (Optional, {None REGULAR_DROP, SEQUENCE_DROP}, defaults to
            None) Type of dropout to be used. REGULAR_DROP is regular
            dropout, and SEQUENCE_DROP is a dropout with the same noise shape
            for each time_step. If None, dropout is not applied. The
            dropout layer is applied before the fc layer, after the batchnorm.
        drop_rate: (Optional, float, defaults to 0.5) Dropout rate. Fraction of
            units to be dropped. If dropout is None, this is ignored.
        activation: (Optional, function, defaults to None) Type of activation
            to be used at the output. If None, activation is linear.
        training: (Optional, boolean, defaults to False) Indicates if it is the
            training phase or not
        reuse: (Optional, boolean, defaults to False) Whether to reuse the layer
            variables.
        name: (Optional, string, defaults to None) A name for the operation.
    """
    with tf.variable_scope(name):
        if dropout:
            inputs = dropout_layer(
                inputs, 'drop', drop_rate=drop_rate, dropout=dropout,
                training=training)
        # [batch_size, time_len, n_feats] -> [batch_size, time_len, 1, feats]
        inputs = tf.expand_dims(inputs, axis=2)
        use_bias = batchnorm is None

        outputs = tf.layers.conv2d(
            inputs=inputs, filters=num_units, kernel_size=1,
            padding=constants.PAD_SAME,
            kernel_initializer=kernel_init, use_bias=use_bias,
            name="conv1", reuse=reuse)

        with tf.variable_scope("zscore"):
            outputs_1 = outputs[..., :num_units // 2]  # Keep as-is
            outputs_2 = outputs[..., num_units // 2:]  # Transform to z-score

            outputs_2_mean = tf.reduce_mean(outputs_2, keepdims=True, axis=1)
            outputs_2 = outputs_2 - outputs_2_mean
            outputs_2_var = tf.reduce_mean(outputs_2 ** 2, keepdims=True, axis=1)
            outputs_2 = outputs_2 / tf.math.sqrt(outputs_2_var + 1e-4)

            # Now join again
            outputs = tf.concat([outputs_1, outputs_2], axis=-1)

        if batchnorm:
            outputs = batchnorm_layer(
                outputs, 'bn', batchnorm=batchnorm,
                reuse=reuse, training=training, scale=False)
        outputs = tf.nn.relu(outputs)

        # [batch_size, time_len, 1, n_units] -> [batch_size, time_len, n_units]
        outputs = tf.squeeze(outputs, axis=2, name="squeeze")
    return outputs


def sequence_fc_layer(
        inputs,
        num_units,
        training,
        batchnorm=None,
        dropout=None,
        drop_rate=0,
        activation=None,
        kernel_init=None,
        use_bias=True,
        reuse=False,
        name=None):
    """ Builds a FC layer that can be applied directly to a sequence.

    Each time-step is passed through to the same FC layer.

    Args:
        inputs: (3d tensor) input tensor of shape
            [batch_size, time_len, n_feats].
        num_units: (int) Number of neurons for the FC layer.
        batchnorm: (Optional, {None, BN, BN_RENORM}, defaults to None) Type of
            batchnorm to be used. BN is normal batchnorm, and BN_RENORM is a
            batchnorm with renorm activated. If None, batchnorm is not applied.
            The batchnorm layer is applied before the fc layer.
        dropout: (Optional, {None REGULAR_DROP, SEQUENCE_DROP}, defaults to
            None) Type of dropout to be used. REGULAR_DROP is regular
            dropout, and SEQUENCE_DROP is a dropout with the same noise shape
            for each time_step. If None, dropout is not applied. The
            dropout layer is applied before the fc layer, after the batchnorm.
        drop_rate: (Optional, float, defaults to 0.5) Dropout rate. Fraction of
            units to be dropped. If dropout is None, this is ignored.
        activation: (Optional, function, defaults to None) Type of activation
            to be used at the output. If None, activation is linear.
        training: (Optional, boolean, defaults to False) Indicates if it is the
            training phase or not
        reuse: (Optional, boolean, defaults to False) Whether to reuse the layer
            variables.
        name: (Optional, string, defaults to None) A name for the operation.
    """
    with tf.variable_scope(name):
        if batchnorm:
            inputs = batchnorm_layer(
                inputs, 'bn', batchnorm=batchnorm, reuse=reuse,
                training=training)
        if dropout:
            inputs = dropout_layer(
                inputs, 'drop', drop_rate=drop_rate, dropout=dropout,
                training=training)
        # [batch_size, time_len, n_feats] -> [batch_size, time_len, 1, feats]
        inputs = tf.expand_dims(inputs, axis=2)
        outputs = tf.layers.conv2d(
            inputs=inputs, filters=num_units, kernel_size=1,
            activation=activation, padding=constants.PAD_SAME,
            kernel_initializer=kernel_init, use_bias=use_bias,
            name="conv1", reuse=reuse)
        # [batch_size, time_len, 1, n_units] -> [batch_size, time_len, n_units]
        outputs = tf.squeeze(outputs, axis=2, name="squeeze")
    return outputs


def sequence_output_2class_layer(
        inputs,
        training,
        batchnorm=None,
        dropout=None,
        drop_rate=0,
        activation=None,
        kernel_init=None,
        init_positive_proba=0.5,
        reuse=False,
        name=None):
    """ Builds a FC layer that can be applied directly to a sequence.

    Each time-step is passed through to the same FC layer.

    Args:
        inputs: (3d tensor) input tensor of shape
            [batch_size, time_len, n_feats].
        num_units: (int) Number of neurons for the FC layer.
        batchnorm: (Optional, {None, BN, BN_RENORM}, defaults to None) Type of
            batchnorm to be used. BN is normal batchnorm, and BN_RENORM is a
            batchnorm with renorm activated. If None, batchnorm is not applied.
            The batchnorm layer is applied before the fc layer.
        dropout: (Optional, {None REGULAR_DROP, SEQUENCE_DROP}, defaults to
            None) Type of dropout to be used. REGULAR_DROP is regular
            dropout, and SEQUENCE_DROP is a dropout with the same noise shape
            for each time_step. If None, dropout is not applied. The
            dropout layer is applied before the fc layer, after the batchnorm.
        drop_rate: (Optional, float, defaults to 0.5) Dropout rate. Fraction of
            units to be dropped. If dropout is None, this is ignored.
        activation: (Optional, function, defaults to None) Type of activation
            to be used at the output. If None, activation is linear.
        training: (Optional, boolean, defaults to False) Indicates if it is the
            training phase or not
        reuse: (Optional, boolean, defaults to False) Whether to reuse the layer
            variables.
        name: (Optional, string, defaults to None) A name for the operation.
    """
    with tf.variable_scope(name):
        if batchnorm:
            inputs = batchnorm_layer(
                inputs, 'bn', batchnorm=batchnorm, reuse=reuse,
                training=training)
        if dropout:
            inputs = dropout_layer(
                inputs, 'drop', drop_rate=drop_rate, dropout=dropout,
                training=training)
        # [batch_size, time_len, n_feats] -> [batch_size, time_len, 1, feats]
        inputs = tf.expand_dims(inputs, axis=2)

        outputs_0 = tf.layers.conv2d(
            inputs=inputs, filters=1, kernel_size=1,
            activation=activation, padding=constants.PAD_SAME,
            kernel_initializer=kernel_init,
            name="conv1_0", reuse=reuse)

        bias_init = - np.log((1 - init_positive_proba) / init_positive_proba)
        print('Initializing bias as %1.4f, to have init positive proba of %1.4f'
              % (bias_init, init_positive_proba))

        outputs_1 = tf.layers.conv2d(
            inputs=inputs, filters=1, kernel_size=1,
            activation=activation, padding=constants.PAD_SAME,
            kernel_initializer=kernel_init,
            bias_initializer=tf.constant_initializer(value=bias_init),
            name="conv1_1", reuse=reuse)

        outputs = tf.concat([outputs_0, outputs_1], axis=3)
        # [batch_size, time_len, 1, n_units] -> [batch_size, time_len, n_units]
        outputs = tf.squeeze(outputs, axis=2, name="squeeze")
    return outputs


def gru_layer(
        inputs,
        num_units,
        training,
        num_dirs=constants.UNIDIRECTIONAL,
        batchnorm=None,
        dropout=None,
        drop_rate=0.5,
        reuse=False,
        name=None):
    """ Builds a GRU layer that can be applied directly to a sequence.

    Args:
        inputs: (3d tensor) input tensor of shape
            [batch_size, time_len, n_feats].
        num_units: (int) Number of neurons for the layers in the GRU.
        num_dirs: (Optional, {UNIDIRECTIONAL, BIDIRECTIONAL}, defaults to
            UNIDIRECTIONAL). Number of directions for the GRU. If
            UNIDIRECTIONAL, a single GRU layer is applied in the forward time
            direction. If BIDIRECTIONAL, another GRU layer is applied in the
            backward time direction, and the output is concatenated with the
            forward time direction layer. In the latter case, the output
            has ndirs*num_units dimensions in the feature axis.
        batchnorm: (Optional, {None, BN, BN_RENORM}, defaults to None) Type of
            batchnorm to be used. BN is normal batchnorm, and BN_RENORM is a
            batchnorm with renorm activated. If None, batchnorm is not applied.
            The batchnorm layer is applied before the fc layer.
        dropout: (Optional, {None REGULAR_DROP, SEQUENCE_DROP}, defaults to
            None) Type of dropout to be used. REGULAR_DROP is regular
            dropout, and SEQUENCE_DROP is a dropout with the same noise shape
            for each time_step. If None, dropout is not applied. The
            dropout layer is applied before the fc layer, after the batchnorm.
        drop_rate: (Optional, float, defaults to 0.5) Dropout rate. Fraction of
            units to be dropped. If dropout is None, this is ignored.
        training: (Optional, boolean, defaults to False) Indicates if it is the
            training phase or not
        reuse: (Optional, boolean, defaults to False) Whether to reuse the layer
            variables.
        name: (Optional, string, defaults to None) A name for the operation.
    """
    checks.check_valid_value(
        num_dirs, 'num_dirs',
        [constants.UNIDIRECTIONAL, constants.BIDIRECTIONAL])

    with tf.variable_scope(name):
        if batchnorm:
            inputs = batchnorm_layer(
                inputs, 'bn', batchnorm=batchnorm, reuse=reuse,
                training=training)
        if dropout:
            inputs = dropout_layer(
                inputs, 'drop', drop_rate=drop_rate, dropout=dropout,
                training=training)

        if num_dirs == constants.UNIDIRECTIONAL:
            gru_name = 'gru'
        else:  # BIDIRECTIONAL
            gru_name = 'bigru'

        use_cudnn = tf.test.is_gpu_available(cuda_only=True)

        # Whether we use CUDNN implementation or CPU implementation, we will
        # use Fused implementations, which are the most efficient. Notice that
        # the inputs to any FusedRNNCell instance should be time-major, this can
        # be done by just transposing the tensor before calling the cell.

        # Turn batch_major into time_major
        inputs = swap_batch_time(inputs, name='to_time_major')

        if use_cudnn:  # GPU is available
            # Apply CUDNN GRU cell
            rnn_cell = tf.contrib.cudnn_rnn.CudnnGRU(
                num_layers=1, num_units=num_units, direction=num_dirs,
                name='cudnn_%s' % gru_name)
            outputs, _ = rnn_cell(inputs)
        else:  # Only CPU is available
            raise NotImplementedError(
                'CPU implementation of GRU not implemented.')

        # Return to batch_major
        outputs = swap_batch_time(outputs, name='to_batch_major')
    return outputs


def lstm_layer(
        inputs,
        num_units,
        training,
        num_dirs=constants.UNIDIRECTIONAL,
        batchnorm=None,
        dropout=None,
        drop_rate=0.5,
        reuse=False,
        name=None):
    """ Builds an LSTM layer that can be applied directly to a sequence.

    Args:
        inputs: (3d tensor) input tensor of shape
            [batch_size, time_len, n_feats].
        num_units: (int) Number of neurons for the layers inside the LSTM cell.
        num_dirs: (Optional, {UNIDIRECTIONAL, BIDIRECTIONAL}, defaults to
            UNIDIRECTIONAL). Number of directions for the LSTM cell. If
            UNIDIRECTIONAL, a single LSTM layer is applied in the forward time
            direction. If BIDIRECTIONAL, another LSTM layer is applied in the
            backward time direction, and the output is concatenated with the
            forward time direction layer. In the latter case, the output
            has ndirs*num_units dimensions in the feature axis.
        batchnorm: (Optional, {None, BN, BN_RENORM}, defaults to None) Type of
            batchnorm to be used. BN is normal batchnorm, and BN_RENORM is a
            batchnorm with renorm activated. If None, batchnorm is not applied.
            The batchnorm layer is applied before the fc layer.
        dropout: (Optional, {None REGULAR_DROP, SEQUENCE_DROP}, defaults to
            None) Type of dropout to be used. REGULAR_DROP is regular
            dropout, and SEQUENCE_DROP is a dropout with the same noise shape
            for each time_step. If None, dropout is not applied. The
            dropout layer is applied before the fc layer, after the batchnorm.
        drop_rate: (Optional, float, defaults to 0.5) Dropout rate. Fraction of
            units to be dropped. If dropout is None, this is ignored.
        training: (Optional, boolean, defaults to False) Indicates if it is the
            training phase or not
        reuse: (Optional, boolean, defaults to False) Whether to reuse the layer
            variables.
        name: (Optional, string, defaults to None) A name for the operation.
    """
    checks.check_valid_value(
        num_dirs, 'num_dirs',
        [constants.UNIDIRECTIONAL, constants.BIDIRECTIONAL])

    with tf.variable_scope(name):
        if batchnorm:
            inputs = batchnorm_layer(
                inputs, 'bn', batchnorm=batchnorm, reuse=reuse,
                training=training)
        if dropout:
            inputs = dropout_layer(
                inputs, 'drop', drop_rate=drop_rate, dropout=dropout,
                training=training)

        if num_dirs == constants.UNIDIRECTIONAL:
            lstm_name = 'lstm'
        else:  # BIDIRECTIONAL
            lstm_name = 'blstm'

        common_args = dict(units=num_units, return_sequences=True, return_state=False)

        if num_dirs == constants.BIDIRECTIONAL:
            with tf.variable_scope(lstm_name):
                forward_rnn_cell = tf.keras.layers.LSTM(**common_args, name='forward')
                backward_rnn_cell = tf.keras.layers.LSTM(**common_args, name='backward')

                forward_outputs = forward_rnn_cell(inputs)

                inputs_reversed = reverse_time(inputs)
                backward_outputs_reversed = backward_rnn_cell(inputs_reversed)
                backward_outputs = reverse_time(backward_outputs_reversed)

                outputs = tf.concat([forward_outputs, backward_outputs], -1)
        else:  # It's UNIDIRECTIONAL
            rnn_cell = tf.keras.layers.LSTM(**common_args, name=lstm_name)
            outputs = rnn_cell(inputs)
    return outputs


def reverse_time(inputs, axis=0):
    """Time reverse the provided 3D tensor. Assumes time major."""
    reversed_inputs = array_ops.reverse_v2(inputs, [axis])
    return reversed_inputs


def time_downsampling_layer(inputs, pooling=constants.AVGPOOL, name=None):
    """Performs a pooling operation on the time dimension by a factor of 2.

    Args:
        inputs: (3d tensor) input tensor with shape [batch, time, feats]
        pooling: (Optional, {AVGPOOL, MAXPOOL}, defaults to AVGPOOL) Specifies
            the type of pooling to be performed along the time axis.
        name: (Optional, string, defaults to None) A name for the operation.
    """
    checks.check_valid_value(
        pooling, 'pooling', [constants.AVGPOOL, constants.MAXPOOL])

    with tf.variable_scope(name):
        # [batch_size, time_len, n_feats] -> [batch_size, time_len, 1, feats]
        inputs = tf.expand_dims(inputs, axis=2)
        if pooling == constants.AVGPOOL:
            outputs = tf.layers.average_pooling2d(
                inputs=inputs, pool_size=(2, 1), strides=(2, 1))
        else:  # MAXPOOL
            outputs = tf.layers.max_pooling2d(
                inputs=inputs, pool_size=(2, 1), strides=(2, 1))

        # [batch_size, time_len/2, 1, n_feats]
        # -> [batch_size, time_len/2, n_feats]
        outputs = tf.squeeze(outputs, axis=2, name="squeeze")
    return outputs


def time_upsampling_layer(inputs, out_feats, name=None):
    """Performs a time upsampling by a factor of 2, using UpConv.

    Args:
        inputs: (3d tensor) input tensor with shape [batch, time, feats]
        out_feats: (int) number of features of the output.
        name: (Optional, string, defaults to None) A name for the operation.
    """
    with tf.variable_scope(name):
        # [batch_size, time_len, n_feats] -> [batch_size, time_len, 1, feats]
        inputs = tf.expand_dims(inputs, axis=2)
        outputs = tf.layers.conv2d_transpose(
            inputs, filters=out_feats, kernel_size=(2, 1),
            strides=(2, 1), padding=constants.PAD_SAME)
        outputs = tf.squeeze(outputs, axis=2, name="squeeze")
    return outputs


def multilayer_gru_block(
        inputs,
        num_units,
        n_layers,
        training,
        num_dirs=constants.UNIDIRECTIONAL,
        batchnorm_first_gru=None,
        dropout_first_gru=None,
        drop_rate_first_gru=0.5,
        batchnorm_rest_gru=None,
        dropout_rest_gru=None,
        drop_rate_rest_gru=0.5,
        name=None):
    """Builds a multi-layer gru block.

    The block consists of BN-GRU-...GRU, with every layer using the same
    specifications. A particular dropout and batchnorm specification can be
    set for the first layer. n_layers defines the number of layers
    to be stacked.
    Please see the documentation of gru_layer for details on input parameters.
    """
    with tf.variable_scope(name):
        outputs = inputs
        for i in range(n_layers):
            if i == 0:
                batchnorm = batchnorm_first_gru
                dropout = dropout_first_gru
                drop_rate = drop_rate_first_gru
            else:
                batchnorm = batchnorm_rest_gru
                dropout = dropout_rest_gru
                drop_rate = drop_rate_rest_gru
            outputs = gru_layer(
                outputs,
                num_units=num_units,
                num_dirs=num_dirs,
                batchnorm=batchnorm,
                dropout=dropout,
                drop_rate=drop_rate,
                training=training,
                name='gru_%d' % (i+1))
    return outputs


def multilayer_lstm_block(
        inputs,
        num_units,
        n_layers,
        training,
        num_dirs=constants.UNIDIRECTIONAL,
        batchnorm_first_lstm=None,
        dropout_first_lstm=None,
        drop_rate_first_lstm=0.5,
        batchnorm_rest_lstm=None,
        dropout_rest_lstm=None,
        drop_rate_rest_lstm=0.5,
        name=None):
    """Builds a multi-layer lstm block.

    The block consists of BN-LSTM-...LSTM, with every layer using the same
    specifications. A particular dropout and batchnorm specification can be
    set for the first layer. n_layers defines the number of layers
    to be stacked.
    Please see the documentation of lstm_layer for details on input parameters.
    """
    with tf.variable_scope(name):
        outputs = inputs
        for i in range(n_layers):
            if i == 0:
                batchnorm = batchnorm_first_lstm
                dropout = dropout_first_lstm
                drop_rate = drop_rate_first_lstm
            else:
                batchnorm = batchnorm_rest_lstm
                dropout = dropout_rest_lstm
                drop_rate = drop_rate_rest_lstm
            outputs = lstm_layer(
                outputs,
                num_units=num_units,
                num_dirs=num_dirs,
                batchnorm=batchnorm,
                dropout=dropout,
                drop_rate=drop_rate,
                training=training,
                name='lstm_%d' % (i+1))
    return outputs


def multistage_lstm_block(
        inputs,
        num_units,
        n_time_levels,
        training,
        duplicate_after_downsampling=True,
        num_dirs=constants.UNIDIRECTIONAL,
        batchnorm_first_lstm=constants.BN,
        dropout_first_lstm=None,
        batchnorm_rest_lstm=None,
        dropout_rest_lstm=None,
        time_pooling=constants.AVGPOOL,
        drop_rate=0.5,
        name=None):
    """Builds a multi-stage lstm block.

    The block consists of a recursive stage structure:

    LSTM ------------------------------------------------------- LSTM
            |                                               |
            downsampling - (another LSTM stage) - upsampling

    Where (another LSTM stage) repeats the same pattern. The number of
    stages is specified with 'n_time_levels', and the last stage is a single
    LSTM layer. If 'n_time_levels' is 1, then a single LSTM layer is returned.
    Every layer uses the same specifications, but a particular dropout and
    batchnorm specification can be set for the first layer. Upsampling is
    performed using an 1D upconv along the time dimension, while downsampling
    is performed using 1D pooling along the time dimension. The number of
    units used in (another LSTM stage) is doubled.
    Please see the documentation of lstm_layer and time downsampling_layer
    for details on input parameters.
    """

    with tf.variable_scope(name):
        if num_dirs == constants.BIDIRECTIONAL:
            stage_channels = 2 * num_units
        else:
            stage_channels = num_units
        if n_time_levels == 1:  # Last stage
            outputs = lstm_layer(
                inputs,
                num_units=num_units,
                num_dirs=num_dirs,
                batchnorm=batchnorm_first_lstm,
                dropout=dropout_first_lstm,
                drop_rate=drop_rate,
                training=training,
                name='lstm')
        else:  # Make a new block
            if duplicate_after_downsampling:
                next_num_units = 2 * num_units
            else:
                next_num_units = num_units
            stage_outputs = lstm_layer(
                inputs,
                num_units=num_units,
                num_dirs=num_dirs,
                batchnorm=batchnorm_first_lstm,
                dropout=dropout_first_lstm,
                drop_rate=drop_rate,
                training=training,
                name='lstm_enc')
            outputs = time_downsampling_layer(
                stage_outputs, pooling=time_pooling, name='down')
            # Nested block
            outputs = multistage_lstm_block(
                outputs,
                next_num_units,
                n_time_levels-1,
                num_dirs=num_dirs,
                batchnorm_first_lstm=batchnorm_rest_lstm,
                dropout_first_lstm=dropout_rest_lstm,
                batchnorm_rest_lstm=batchnorm_rest_lstm,
                dropout_rest_lstm=dropout_rest_lstm,
                time_pooling=time_pooling,
                drop_rate=drop_rate,
                training=training,
                name='next_stage')
            outputs = time_upsampling_layer(
                outputs, stage_channels, name='up')
            outputs = lstm_layer(
                tf.concat([outputs, stage_outputs], axis=-1),
                num_units=num_units,
                num_dirs=num_dirs,
                batchnorm=batchnorm_rest_lstm,
                dropout=dropout_rest_lstm,
                drop_rate=drop_rate,
                training=training,
                name='lstm_dec')
    return outputs


def get_positional_encoding(seq_len, dims, pe_factor, name=None):
    with tf.variable_scope(name):
        positional_encoding = np.zeros((seq_len, dims))
        if pe_factor is not None:
            print('Using Positional Encoding with factor %d' % pe_factor)
            positions = np.arange(seq_len)
            positions = np.reshape(positions, (-1, 1))
            even_dims = 2 * np.arange(dims / 2)
            denominators = 1 / (pe_factor ** (even_dims / dims))
            denominators = np.reshape(denominators, (1, -1))
            sin_arguments = np.dot(positions, denominators)
            # positional_encoding = np.zeros((seq_len, dims))
            positional_encoding[:, ::2] = np.sin(sin_arguments)
            positional_encoding[:, 1::2] = np.cos(sin_arguments)
        else:
            print('Not using Positional Encoding')
        positional_encoding = tf.cast(positional_encoding, dtype=tf.float32)
        # Returns shape [seq_len, dims]
    return positional_encoding


def attention_layer(queries, keys, values, name=None):
    with tf.variable_scope(name):
        # input shapes are [batch, time_len, dims]
        dim_keys = tf.shape(keys)
        dim_keys = tf.cast(dim_keys[2], dtype=tf.float32)
        scores = tf.matmul(queries, keys, transpose_b=True)
        # scores have shape [batch, q_time_len, k_time_len]
        scaled_scores = scores / tf.sqrt(dim_keys)
        att_weights = tf.nn.softmax(scaled_scores, axis=-1)
        outputs = tf.matmul(att_weights, values)
        # outputs have shape [batch, q_time_len, v_dims]
        return outputs, att_weights


def naive_multihead_attention_layer(queries, keys, values, n_heads, name=None):
    with tf.variable_scope(name):
        # Divide into heads
        heads_q = tf.split(queries, n_heads, 2)
        heads_k = tf.split(keys, n_heads, 2)
        heads_v = tf.split(values, n_heads, 2)

        outputs = []
        for idx_head in range(n_heads):
            # scores have shape [batch, q_time_len, k_time_len]
            # outputs have shape [batch, q_time_len, v_dims]
            head_o, _ = attention_layer(
                heads_q[idx_head], heads_k[idx_head], heads_v[idx_head],
                name='head_%d' % idx_head)
            outputs.append(head_o)

        # Concatenate heads
        outputs = tf.concat(outputs, axis=-1)
        return outputs


def multihead_attention_layer(queries, keys, values, n_heads, name=None):
    with tf.variable_scope(name):
        # inputs shape [batch, seq_len, n_feats]
        seq_len = queries.get_shape().as_list()[1]
        d_model = queries.get_shape().as_list()[2]
        depth = d_model // n_heads

        # Divide into heads: (batch_size, num_heads, seq_len, depth)
        queries = tf.reshape(queries, (-1, seq_len, n_heads, depth))
        q = tf.transpose(queries, perm=[0, 2, 1, 3])

        keys = tf.reshape(keys, (-1, seq_len, n_heads, depth))
        k = tf.transpose(keys, perm=[0, 2, 1, 3])

        values = tf.reshape(values, (-1, seq_len, n_heads, depth))
        v = tf.transpose(values, perm=[0, 2, 1, 3])

        # Compute scaled attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
        outputs = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
        # outputs.shape == (batch_size, num_heads, seq_len_q, depth)

        # Concatenate heads
        scaled_attention = tf.transpose(outputs, perm=[0, 2, 1, 3])
        # (batch_size, seq_len, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (-1, seq_len, d_model))
        # (batch_size, seq_len, d_model)
    return concat_attention, attention_weights


def tcn_block(
        inputs,
        filters,
        kernel_size,
        dilation,
        drop_rate,
        training,
        bottleneck=True,
        is_first_unit=False,
        batchnorm=constants.BN,
        reuse=False,
        kernel_init=None,
        name=None
):
    with tf.variable_scope(name):

        # [batch_size, time_len, n_feats] -> [batch_size, time_len, 1, feats]
        inputs = tf.expand_dims(inputs, axis=2)

        if is_first_unit:
            inputs = spatial_dropout_2d(inputs, drop_rate, training, "drop_1")

        shortcut = inputs
        # Projection if necessary
        input_filters = shortcut.get_shape().as_list()[-1]
        if input_filters != filters:
            shortcut = tf.layers.conv2d(
                inputs=shortcut, filters=filters, kernel_size=1,
                padding=constants.PAD_SAME, use_bias=False,
                kernel_initializer=kernel_init, name='projection', reuse=reuse)

        outputs = inputs

        if not is_first_unit:
            if batchnorm:
                outputs = batchnorm_layer(
                    outputs, 'bn_1', batchnorm=batchnorm,
                    reuse=reuse, training=training, scale=False)
            outputs = tf.nn.relu(outputs)
            outputs = spatial_dropout_2d(outputs, drop_rate, training, "drop_1")

        conv_use_bias = (batchnorm is None)

        if bottleneck:
            down_factor = 4

            outputs = tf.layers.conv2d(
                inputs=outputs, filters=filters//down_factor, kernel_size=1,
                padding=constants.PAD_SAME,
                name='conv_1', reuse=reuse,
                use_bias=conv_use_bias, kernel_initializer=kernel_init)

            if batchnorm:
                outputs = batchnorm_layer(
                    outputs, 'bn_2', batchnorm=batchnorm,
                    reuse=reuse, training=training, scale=False)
            outputs = tf.nn.relu(outputs)
            outputs = spatial_dropout_2d(outputs, drop_rate, training, "drop_2")

            outputs = tf.layers.conv2d(
                inputs=outputs, filters=filters//down_factor,
                kernel_size=(kernel_size, 1),
                padding=constants.PAD_SAME, dilation_rate=dilation,
                name='conv_2', reuse=reuse,
                use_bias=conv_use_bias, kernel_initializer=kernel_init)

            if batchnorm:
                outputs = batchnorm_layer(
                    outputs, 'bn_3', batchnorm=batchnorm,
                    reuse=reuse, training=training, scale=False)
            outputs = tf.nn.relu(outputs)
            outputs = spatial_dropout_2d(outputs, drop_rate, training, "drop_3")

            outputs = tf.layers.conv2d(
                inputs=outputs, filters=filters, kernel_size=1,
                padding=constants.PAD_SAME,
                name='conv_3', reuse=reuse,
                use_bias=conv_use_bias, kernel_initializer=kernel_init)
        else:
            outputs = tf.layers.conv2d(
                inputs=outputs, filters=filters, kernel_size=(kernel_size, 1),
                padding=constants.PAD_SAME, dilation_rate=dilation,
                name='conv_1', reuse=reuse,
                use_bias=conv_use_bias, kernel_initializer=kernel_init)

            if batchnorm:
                outputs = batchnorm_layer(
                    outputs, 'bn_2', batchnorm=batchnorm,
                    reuse=reuse, training=training, scale=False)
            outputs = tf.nn.relu(outputs)
            outputs = spatial_dropout_2d(outputs, drop_rate, training, "drop_2")

            outputs = tf.layers.conv2d(
                inputs=outputs, filters=filters, kernel_size=(kernel_size, 1),
                padding=constants.PAD_SAME, dilation_rate=dilation,
                name='conv_2', reuse=reuse,
                use_bias=conv_use_bias, kernel_initializer=kernel_init)

        outputs = outputs + shortcut

        # [batch_size, time_len, 1, n_units] -> [batch_size, time_len, n_units]
        outputs = tf.squeeze(outputs, axis=2, name="squeeze")

    return outputs


def tcn_block_simple(
        inputs,
        filters,
        kernel_size,
        dilation,
        drop_rate,
        training,
        batchnorm=constants.BN,
        reuse=False,
        kernel_init=None,
        name=None
):
    with tf.variable_scope(name):
        # [batch_size, time_len, n_feats] -> [batch_size, time_len, 1, feats]
        outputs = tf.expand_dims(inputs, axis=2)
        outputs = spatial_dropout_2d(outputs, drop_rate, training, "drop")
        conv_use_bias = (batchnorm is None)
        outputs = tf.layers.conv2d(
            inputs=outputs, filters=filters, kernel_size=(kernel_size, 1),
            padding=constants.PAD_SAME, dilation_rate=dilation,
            name='conv', reuse=reuse,
            use_bias=conv_use_bias, kernel_initializer=kernel_init)
        if batchnorm:
            outputs = batchnorm_layer(
                outputs, 'bn', batchnorm=batchnorm,
                reuse=reuse, training=training, scale=False)
        outputs = tf.nn.relu(outputs)
        # [batch_size, time_len, 1, n_units] -> [batch_size, time_len, n_units]
        outputs = tf.squeeze(outputs, axis=2, name="squeeze")
    return outputs


def conv1d_prebn(
        inputs,
        filters,
        kernel_size,
        training,
        batchnorm=constants.BN,
        reuse=False,
        kernel_init=None,
        name=None
):
    with tf.variable_scope(name):

        # [batch_size, time_len, n_feats] -> [batch_size, time_len, 1, feats]
        inputs = tf.expand_dims(inputs, axis=2)

        conv_use_bias = (batchnorm is None)
        outputs = tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=(kernel_size, 1),
            padding=constants.PAD_SAME,
            name='conv', reuse=reuse,
            use_bias=conv_use_bias, kernel_initializer=kernel_init)

        if batchnorm:
            outputs = batchnorm_layer(
                outputs, 'bn', batchnorm=batchnorm,
                reuse=reuse, training=training, scale=False)
        outputs = tf.nn.relu(outputs)

        # [batch_size, time_len, 1, n_units] -> [batch_size, time_len, n_units]
        outputs = tf.squeeze(outputs, axis=2, name="squeeze")

    return outputs


def spatial_dropout_2d(inputs, drop_rate, training, name):
    # Input shape [batch, h, w, channels]
    with tf.variable_scope(name):
        # Dropout Spatial (drop entire feature map)
        in_shape = tf.shape(inputs)
        noise_shape = [in_shape[0], 1, 1, in_shape[3]]
        outputs = tf.layers.dropout(
            inputs, training=training, rate=drop_rate,
            noise_shape=noise_shape)
    return outputs


def signal_decomposition_bandpass(inputs, fs, name):
    outputs_00_32 = inputs
    with tf.variable_scope(name):
        # Extract 0-4 Hz component
        outputs_00_04 = lowpass_tf_batch(outputs_00_32, fs, 4)
        # Extract 4-8 Hz component
        outputs_04_32 = outputs_00_32 - outputs_00_04
        outputs_04_08 = lowpass_tf_batch(outputs_04_32, fs, 8)
        # Extract 8-16 Hz component
        outputs_08_32 = outputs_04_32 - outputs_04_08
        outputs_08_16 = lowpass_tf_batch(outputs_08_32, fs, 16)
        # Extract 16-32 Hz component
        outputs_16_32 = outputs_08_32 - outputs_08_16
        bands = {
            '0-4Hz': outputs_00_04,
            '4-8Hz': outputs_04_08,
            '8-16Hz': outputs_08_16,
            '16-32Hz': outputs_16_32
        }
    return bands
