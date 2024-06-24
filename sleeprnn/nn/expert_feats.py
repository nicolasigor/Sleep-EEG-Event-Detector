from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.signal import firwin
import tensorflow as tf

from sleeprnn.common import constants


def apply_fir_filter_tf_batch(signals, kernel):
    """For batch of signals"""
    signals = signals[:, tf.newaxis, :, tf.newaxis]
    kernel = tf.reshape(kernel, shape=[1, -1, 1, 1])
    new_signals = tf.nn.conv2d(
        input=signals, filter=kernel, strides=[1, 1, 1, 1], padding="SAME")
    new_signals = new_signals[:, 0, :, 0]
    return new_signals


def lowpass_tf_batch(signals, fs, cutoff, filter_duration_ref=6, wave_expansion_factor=0.5):
    print("Applying lowpass filter with cutoff %s Hz" % cutoff)
    numtaps = fs * filter_duration_ref / (cutoff ** wave_expansion_factor)
    numtaps = int(2 * (numtaps // 2) + 1)  # ensure odd numtaps
    lp_kernel = firwin(numtaps, cutoff=cutoff, window="hamming", fs=fs).astype(np.float32)
    lp_kernel /= lp_kernel.sum()
    new_signals = apply_fir_filter_tf_batch(signals, lp_kernel)
    return new_signals


def highpass_tf_batch(signals, fs, cutoff, filter_duration_ref=6, wave_expansion_factor=0.5):
    print("Applying highpass filter with cutoff %s Hz" % cutoff)
    numtaps = fs * filter_duration_ref / (cutoff ** wave_expansion_factor)
    numtaps = int(2 * (numtaps // 2) + 1)  # ensure odd numtaps
    lp_kernel = firwin(numtaps, cutoff=cutoff, window="hamming", fs=fs).astype(np.float32)
    lp_kernel /= lp_kernel.sum()
    # HP = delta - LP
    hp_kernel = -lp_kernel
    hp_kernel[numtaps // 2] += 1
    new_signals = apply_fir_filter_tf_batch(signals, hp_kernel)
    return new_signals


def bandpass_tf_batch(signals, fs, lowcut, highcut, filter_duration_ref=6, wave_expansion_factor=0.5):
    new_signals = signals
    if lowcut is not None:
        new_signals = highpass_tf_batch(
            new_signals, fs, lowcut, filter_duration_ref, wave_expansion_factor)
    if highcut is not None:
        new_signals = lowpass_tf_batch(
            new_signals, fs, highcut, filter_duration_ref, wave_expansion_factor)
    return new_signals


def get_static_lowpass_kernel(filter_size):
    print("Using static lowpass kernel")
    lp_filter = np.hanning(filter_size).astype(np.float32)
    lp_filter /= lp_filter.sum()
    return lp_filter


def get_learnable_lowpass_kernel(initial_filter_size, size_factor=3.0, trainable=True):
    with tf.variable_scope("learnable_kernel"):
        print("Window duration: theta trainable = %s" % trainable)
        theta = tf.Variable(initial_value=0.0, trainable=trainable, name='window_factor', dtype=tf.float32)
        one_side = int(size_factor * initial_filter_size / 2)
        kernel_size = 2 * one_side + 1
        k_array = np.arange(kernel_size, dtype=np.float32) - one_side
        sigma = initial_filter_size / 6
        lp_filter = tf.math.exp(- tf.math.exp(theta) * k_array ** 2 / (2 * sigma ** 2))
        lp_filter = lp_filter / tf.reduce_sum(lp_filter)

        # Summary to track theta
        tf.summary.scalar('window_factor', theta)

    return lp_filter


def moving_average_tf(signals, lp_filter_size):
    lp_filter = np.hanning(lp_filter_size).astype(np.float32)
    lp_filter /= lp_filter.sum()
    results = apply_fir_filter_tf_batch(signals, lp_filter)
    return results


def zscore_tf(signals, dispersion_mode=constants.DISPERSION_STD_ROBUST):
    mean_signals = tf.reduce_mean(signals, axis=1, keepdims=True)
    signals = signals - mean_signals
    if dispersion_mode == constants.DISPERSION_MADE:
        std_signals = tf.reduce_mean(tf.math.abs(signals), axis=1, keepdims=True)
    elif dispersion_mode == constants.DISPERSION_STD:
        std_signals = tf.math.sqrt(tf.reduce_mean(signals ** 2, axis=1, keepdims=True))
    elif dispersion_mode == constants.DISPERSION_STD_ROBUST:
        prc_range = [10, 90]  # valid percentile range to avoid extreme values
        signal_size = signals.get_shape().as_list()[1]
        loc_prc_start = int(signal_size * prc_range[0] / 100)
        loc_prc_end = int(signal_size * prc_range[1] / 100)
        sorted_values = tf.sort(signals, axis=1)
        sorted_values_valid = sorted_values[:, loc_prc_start:loc_prc_end]
        std_signals = tf.math.sqrt(tf.reduce_mean(sorted_values_valid ** 2, axis=1, keepdims=True))
    else:
        raise ValueError()
    signals = signals / std_signals
    return signals


def log10_tf(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def a7_layer_tf(
        signals,
        fs,
        window_duration,
        window_duration_relSigPow,
        sigma_lowcut=11,
        sigma_highcut=16,
        use_log_absSigPow=True,
        use_log_relSigPow=True,
        use_log_sigCov=True,
        use_log_sigCorr=False,
        use_zscore_relSigPow=True,
        use_zscore_sigCov=True,
        use_zscore_sigCorr=False,
        remove_delta_in_cov=False,
        dispersion_mode=constants.DISPERSION_STD_ROBUST,
        return_raw_values=False
):
    with tf.variable_scope("a7_layer"):
        lp_filter_size = int(fs * window_duration)
        if window_duration_relSigPow is None:
            window_duration_relSigPow = window_duration
        lp_filter_size_relSigPow = int(fs * window_duration_relSigPow)
        print("Moving window: Using %1.2f s (%d samples)" % (window_duration, lp_filter_size))
        print("Moving window in relSigPow: Using %1.2f s (%d samples)" % (
            window_duration_relSigPow, lp_filter_size_relSigPow))
        print("Z-score: Using '%s'" % dispersion_mode)

        signal_sigma = bandpass_tf_batch(signals, fs, sigma_lowcut, sigma_highcut)
        signal_no_delta = bandpass_tf_batch(signals, fs, 4.5, None)

        # absolute sigma power
        signal_sigma_squared = signal_sigma ** 2
        abs_sig_pow_raw = moving_average_tf(signal_sigma_squared, lp_filter_size)
        if use_log_absSigPow:
            abs_sig_pow = log10_tf(abs_sig_pow_raw + 1e-4)
            print("absSigPow: Using log10.")
        else:
            abs_sig_pow = tf.math.sqrt(abs_sig_pow_raw)
            print("absSigPow: Using sqrt.")

        # relative sigma power
        signal_sigma_squared = signal_sigma ** 2
        abs_sig_pow_raw_2 = moving_average_tf(signal_sigma_squared, lp_filter_size_relSigPow)
        signal_no_delta_squared = signal_no_delta ** 2
        abs_no_delta_pow_raw = moving_average_tf(signal_no_delta_squared, lp_filter_size_relSigPow)
        rel_sig_pow_raw = abs_sig_pow_raw_2 / (abs_no_delta_pow_raw + 1e-6)
        if use_log_relSigPow:
            rel_sig_pow = log10_tf(rel_sig_pow_raw + 1e-4)
            print("relSigPow: Using log10.")
        else:
            rel_sig_pow = tf.math.sqrt(rel_sig_pow_raw)
            print("relSigPow: Using sqrt.")
        if use_zscore_relSigPow:
            rel_sig_pow = zscore_tf(rel_sig_pow, dispersion_mode)
            print("relSigPow: Using z-score.")

        # sigma covariance
        sigma_centered = signal_sigma - tf.reduce_mean(signal_sigma, axis=1, keepdims=True)
        if remove_delta_in_cov:
            broad_centered = signal_no_delta - tf.reduce_mean(signal_no_delta, axis=1, keepdims=True)
            print("Removing delta band in covariance.")
        else:
            broad_centered = signals - tf.reduce_mean(signals, axis=1, keepdims=True)
        sig_cov_raw = moving_average_tf(sigma_centered * broad_centered, lp_filter_size)
        sig_cov = tf.nn.relu(sig_cov_raw)  # no negatives
        if use_log_sigCov:
            sig_cov = log10_tf(sig_cov + 1e-4)
            print("sigCov: Using log10.")
        else:
            sig_cov = tf.math.sqrt(sig_cov)
            print("sigCov: Using sqrt.")
        if use_zscore_sigCov:
            sig_cov = zscore_tf(sig_cov, dispersion_mode)
            print("sigCov: Using z-score.")

        # sigma correlation
        sig_var_raw = moving_average_tf(sigma_centered ** 2, lp_filter_size)
        broad_var_raw = moving_average_tf(broad_centered ** 2, lp_filter_size)
        sig_std_raw = tf.math.sqrt(sig_var_raw)
        broad_stf_raw = tf.math.sqrt(broad_var_raw)
        sig_corr_raw = sig_cov_raw / (sig_std_raw * broad_stf_raw + 1e-6)
        if use_log_sigCorr:
            sig_corr = log10_tf(tf.nn.relu(sig_corr_raw) + 1e-4)
            print("sigCorr: Using log10.")
        else:
            sig_corr = sig_corr_raw
            print("sigCorr: Using raw.")
        if use_zscore_sigCorr:
            sig_corr = zscore_tf(sig_corr, dispersion_mode)
            print("sigCorr: Using z-score.")

        a7_parameters = tf.stack([abs_sig_pow, rel_sig_pow, sig_cov, sig_corr], axis=2)
        if return_raw_values:
            a7_parameters_raw = tf.stack([abs_sig_pow_raw, rel_sig_pow_raw, sig_cov_raw, sig_corr_raw], axis=2)
            return a7_parameters, a7_parameters_raw
        else:
            return a7_parameters


def a7_layer_v2_tf(
        signals,
        fs,
        window_duration,
        sigma_frequencies=(11, 16),
        rel_power_broad_lowcut=4.5,
        covariance_broad_lowcut=None,
        abs_power_transformation='log',
        rel_power_transformation='log',
        covariance_transformation='log',
        correlation_transformation=None,
        rel_power_use_zscore=True,
        covariance_use_zscore=True,
        correlation_use_zscore=False,
        zscore_dispersion_mode=constants.DISPERSION_STD_ROBUST,
        trainable_window_duration=True,
):
    transformations_dict = {
        'abs_power': abs_power_transformation,
        'rel_power': rel_power_transformation,
        'covariance': covariance_transformation,
        'correlation': correlation_transformation}
    zscore_dict = {
        'abs_power': False,
        'rel_power': rel_power_use_zscore,
        'covariance': covariance_use_zscore,
        'correlation': correlation_use_zscore}
    feats_dict = {}
    with tf.variable_scope("a7_dense_feats"):
        initial_lp_filter_size = int(fs * window_duration)
        print("Moving window: Using %1.2f s initial duration (%d samples)" % (window_duration, initial_lp_filter_size))
        print("Z-score: Using '%s'" % zscore_dispersion_mode)
        print("Broad band low cutoff for relative power:", rel_power_broad_lowcut)
        print("Broad band low cutoff for covariance/correlation:", covariance_broad_lowcut)

        signal_sigma = bandpass_tf_batch(signals, fs, sigma_frequencies[0], sigma_frequencies[1])
        signal_rel_power_broad = bandpass_tf_batch(signals, fs, rel_power_broad_lowcut, None)
        signal_covariance_broad = bandpass_tf_batch(signals, fs, covariance_broad_lowcut, None)

        # Learnable kernel
        if trainable_window_duration:
            lp_filter_kernel = get_learnable_lowpass_kernel(initial_lp_filter_size)
        else:
            lp_filter_kernel = get_static_lowpass_kernel(initial_lp_filter_size)

        # Raw parameters
        # --------------
        # Absolute sigma power
        sigma_power = apply_fir_filter_tf_batch(signal_sigma ** 2, lp_filter_kernel)
        feats_dict['abs_power'] = sigma_power
        # Relative sigma power
        broad_power = apply_fir_filter_tf_batch(signal_rel_power_broad ** 2, lp_filter_kernel)
        relative_power = sigma_power / (broad_power + 1e-8)
        feats_dict['rel_power'] = relative_power
        # Sigma-Broad covariance
        # cov(x, y) = E(x * y) - E(x) * E(y)
        sigma_mean = apply_fir_filter_tf_batch(signal_sigma, lp_filter_kernel)
        broad_mean = apply_fir_filter_tf_batch(signal_covariance_broad, lp_filter_kernel)
        sigma_broad_mean_product = apply_fir_filter_tf_batch(signal_sigma * signal_covariance_broad, lp_filter_kernel)
        covariance = sigma_broad_mean_product - sigma_mean * broad_mean
        feats_dict['covariance'] = covariance
        # Sigma-Broad correlation factor
        # var(x) = E(x ** 2) - E(x) ** 2
        sigma_squared = sigma_power
        broad_squared = apply_fir_filter_tf_batch(signal_covariance_broad ** 2, lp_filter_kernel)
        sigma_variance = sigma_squared - sigma_mean ** 2
        broad_variance = broad_squared - broad_mean ** 2
        correlation = covariance / tf.math.sqrt(sigma_variance * broad_variance + 1e-8)
        feats_dict['correlation'] = correlation

        # Transformations
        for feat_name in feats_dict.keys():
            feat_signal = feats_dict[feat_name]
            # Apply transformation
            if transformations_dict[feat_name] is None:
                print("Not applying transformation to feature '%s'" % feat_name)
            elif transformations_dict[feat_name] == 'log':
                print("Applying logarithm to feature '%s'" % feat_name)
                feat_signal = tf.nn.relu(feat_signal) if feat_name in ['covariance', 'correlation'] else feat_signal
                feat_signal = log10_tf(feat_signal + 1e-4)
            elif transformations_dict[feat_name] == 'sqrt':
                print("Applying sqrt to feature '%s'" % feat_name)
                feat_signal = tf.nn.relu(feat_signal) if feat_name in ['covariance', 'correlation'] else feat_signal
                feat_signal = tf.math.sqrt(feat_signal)
            else:
                raise ValueError("Transformation %s not supported" % transformations_dict[feat_name])
            feats_dict[feat_name] = feat_signal

        # Z-score
        for feat_name in feats_dict.keys():
            feat_signal = feats_dict[feat_name]
            if zscore_dict[feat_name]:
                print("Applying z-score to feature '%s'" % feat_name)
                feat_signal = zscore_tf(feat_signal, zscore_dispersion_mode)
            else:
                print("Not applying z-score to feature '%s'" % feat_name)
            feats_dict[feat_name] = feat_signal

        features_stacked = tf.stack([
            feats_dict["abs_power"], feats_dict["rel_power"], feats_dict["covariance"], feats_dict["correlation"]
        ], axis=2)
        return features_stacked
