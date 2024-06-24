from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.signal import firwin
import tensorflow as tf


def apply_fir_filter_tf(signal, kernel):
    """For single signal, not batch"""
    signal = tf.reshape(signal, shape=[1, 1, -1, 1])
    kernel = tf.reshape(kernel, shape=[1, -1, 1, 1])
    with tf.device("/cpu:0"):
        new_signal = tf.nn.conv2d(
            input=signal, filter=kernel, strides=[1, 1, 1, 1], padding="SAME"
        )
    new_signal = new_signal[0, 0, :, 0]
    return new_signal


def random_window_tf(signal_size, window_min_size, window_max_size):
    window_size = tf.random.uniform([], minval=window_min_size, maxval=window_max_size)
    start_sample = tf.random.uniform(
        [], minval=0, maxval=(signal_size - window_size - 1)
    )
    k_array = np.arange(signal_size)
    offset_1 = start_sample + 0.1 * window_size
    offset_2 = start_sample + 0.9 * window_size
    scaling = 0.1 * window_size / 4
    window_onset = tf.math.sigmoid((k_array - offset_1) / scaling)
    window_offset = tf.math.sigmoid((k_array - offset_2) / scaling)
    window = window_onset - window_offset
    return window


def random_smooth_function_tf(
    signal_size, function_min_val, function_max_val, lp_filter_size
):
    lp_filter = np.hanning(lp_filter_size).astype(np.float32)
    lp_filter /= lp_filter.sum()
    noise_vector = tf.random.uniform([signal_size], minval=-1, maxval=1)
    noise_vector = apply_fir_filter_tf(noise_vector, lp_filter)
    # Set noise to [0, 1] range
    min_val = tf.reduce_min(noise_vector)
    max_val = tf.reduce_max(noise_vector)
    noise_vector = (noise_vector - min_val) / (max_val - min_val)
    # Set to [function_min_val, function_max_val] range
    noise_vector = function_min_val + noise_vector * (
        function_max_val - function_min_val
    )
    return noise_vector


def lowpass_tf(signal, fs, cutoff, filter_duration_ref=6, wave_expansion_factor=0.5):
    numtaps = fs * filter_duration_ref / (cutoff**wave_expansion_factor)
    numtaps = int(2 * (numtaps // 2) + 1)  # ensure odd numtaps
    lp_kernel = firwin(numtaps, cutoff=cutoff, window="hann", fs=fs).astype(np.float32)
    lp_kernel /= lp_kernel.sum()
    new_signal = apply_fir_filter_tf(signal, lp_kernel)
    return new_signal


def highpass_tf(signal, fs, cutoff, filter_duration_ref=6, wave_expansion_factor=0.5):
    numtaps = fs * filter_duration_ref / (cutoff**wave_expansion_factor)
    numtaps = int(2 * (numtaps // 2) + 1)  # ensure odd numtaps
    lp_kernel = firwin(numtaps, cutoff=cutoff, window="hann", fs=fs).astype(np.float32)
    lp_kernel /= lp_kernel.sum()
    # HP = delta - LP
    hp_kernel = -lp_kernel
    hp_kernel[numtaps // 2] += 1
    new_signal = apply_fir_filter_tf(signal, hp_kernel)
    return new_signal


def bandpass_tf(
    signal, fs, lowcut, highcut, filter_duration_ref=6, wave_expansion_factor=0.5
):
    new_signal = signal
    if lowcut is not None:
        new_signal = highpass_tf(
            new_signal, fs, lowcut, filter_duration_ref, wave_expansion_factor
        )
    if highcut is not None:
        new_signal = lowpass_tf(
            new_signal, fs, highcut, filter_duration_ref, wave_expansion_factor
        )
    return new_signal


def generate_soft_mask_from_labels_tf(
    labels, fs, mask_lp_filter_duration=0.2, use_background=True
):
    lp_filter_size = int(fs * mask_lp_filter_duration)
    labels = tf.cast(labels, tf.float32)
    # Enlarge labels
    expand_filter = np.ones(lp_filter_size).astype(np.float32)
    expanded_labels = apply_fir_filter_tf(labels, expand_filter)
    expanded_labels = tf.clip_by_value(expanded_labels, 0, 1)
    # Now filter
    lp_filter = np.hanning(lp_filter_size).astype(np.float32)
    lp_filter /= lp_filter.sum()
    smooth_labels = apply_fir_filter_tf(expanded_labels, lp_filter)
    if use_background:
        soft_mask = 1 - smooth_labels
    else:
        soft_mask = smooth_labels
    return soft_mask


def generate_wave_tf(
    signal_size,  # Number of samples
    fs,  # [Hz]
    max_amplitude,  # signal units
    min_frequency,  # [Hz]
    max_frequency,  # [Hz]
    frequency_bandwidth,  # [Hz]
    min_duration,  # [s]
    max_duration,  # [s]
    mask,  # [0, 1]
    frequency_lp_filter_duration=0.5,  # [s]
    amplitude_lp_filter_duration=0.5,  # [s]
    return_intermediate_steps=False,
):
    # This is ok to be numpy
    window_min_size = int(fs * min_duration)
    window_max_size = int(fs * max_duration)
    frequency_lp_filter_size = int(fs * frequency_lp_filter_duration)
    amplitude_lp_filter_size = int(fs * amplitude_lp_filter_duration)
    # Oscillation
    lower_freq = tf.random.uniform(
        [], minval=min_frequency, maxval=max_frequency - frequency_bandwidth
    )
    upper_freq = lower_freq + frequency_bandwidth
    wave_freq = random_smooth_function_tf(
        signal_size, lower_freq, upper_freq, frequency_lp_filter_size
    )
    wave_phase = 2 * np.pi * tf.math.cumsum(wave_freq) / fs
    oscillation = tf.math.cos(wave_phase)
    # Amplitude
    amplitude_high = tf.random.uniform([], minval=0, maxval=max_amplitude)
    amplitude_low = tf.random.uniform([], minval=0, maxval=amplitude_high)
    amplitude = random_smooth_function_tf(
        signal_size, amplitude_low, amplitude_high, amplitude_lp_filter_size
    )
    # Window
    window = random_window_tf(signal_size, window_min_size, window_max_size)
    # Total wave
    generated_wave = window * amplitude * oscillation
    # Optional masking
    if mask is not None:
        generated_wave = generated_wave * mask
    if return_intermediate_steps:
        intermediate_steps = {
            "oscillation": oscillation,
            "amplitude": amplitude,
            "window": window,
            "mask": mask,
        }
        return generated_wave, intermediate_steps
    return generated_wave


def generate_anti_wave_tf(
    signal,
    signal_size,  # number of samples
    fs,  # [Hz]
    lowcut,  # [Hz]
    highcut,  # [Hz]
    min_duration,  # [s]
    max_duration,  # [s]
    max_attenuation,  # [0, 1]
    mask,  # [0, 1]
    return_intermediate_steps=False,
):
    # This is ok to be numpy
    window_min_size = int(fs * min_duration)
    window_max_size = int(fs * max_duration)
    # Oscillation (opposite sign of band signal) and attenuation factor
    oscillation = -bandpass_tf(signal, fs, lowcut, highcut)
    attenuation_factor = tf.random.uniform([], minval=0, maxval=max_attenuation)
    # Window
    window = random_window_tf(signal_size, window_min_size, window_max_size)
    # Total wave
    generated_wave = window * attenuation_factor * oscillation
    # Optional masking
    if mask is not None:
        generated_wave = generated_wave * mask
    if return_intermediate_steps:
        intermediate_steps = {
            "oscillation": -oscillation,
            "attenuation": -attenuation_factor,
            "window": window,
            "mask": mask,
        }
        return generated_wave, intermediate_steps
    return generated_wave


def generate_base_oscillation(
    signal_size,  # Number of samples
    fs,  # [Hz]
    min_frequency,  # [Hz]
    max_frequency,  # [Hz]
    frequency_variation_width,  # [Hz]
    min_amplitude,  # signal units
    max_amplitude,  # signal units
    amplitude_relative_variation_width,  # relative
    frequency_lp_filter_duration=0.5,  # [s]
    amplitude_lp_filter_duration=0.5,  # [s]
):
    frequency_lp_filter_size = int(fs * frequency_lp_filter_duration)
    amplitude_lp_filter_size = int(fs * amplitude_lp_filter_duration)
    # Oscillation
    central_freq = tf.random.uniform([], minval=min_frequency, maxval=max_frequency)
    lower_freq = central_freq - 0.5 * frequency_variation_width
    upper_freq = central_freq + 0.5 * frequency_variation_width
    wave_freq = random_smooth_function_tf(
        signal_size, lower_freq, upper_freq, frequency_lp_filter_size
    )
    wave_phase = 2 * np.pi * tf.math.cumsum(wave_freq) / fs
    oscillation = tf.math.cos(wave_phase)
    # Amplitude
    central_amplitude = tf.random.uniform(
        [], minval=min_amplitude, maxval=max_amplitude
    )
    amplitude_high = central_amplitude * (1 + 0.5 * amplitude_relative_variation_width)
    amplitude_low = central_amplitude * (1 - 0.5 * amplitude_relative_variation_width)
    amplitude = random_smooth_function_tf(
        signal_size, amplitude_low, amplitude_high, amplitude_lp_filter_size
    )
    return amplitude * oscillation, central_amplitude, central_freq


def generate_false_spindle_single_contamination(
    signal,
    signal_size,  # Number of samples
    fs,  # [Hz]
    duration_range,  # [s]
    bandstop_cutoff,  # [Hz]
    spindle_frequency_range,  # [Hz]
    spindle_frequency_variation_width,  # [Hz]
    spindle_amplitude_absolute_range,  # signal units
    spindle_amplitude_relative_variation_width,
    contamination_frequency_range,  # [Hz]  IMPORTANT
    contamination_frequency_variation_width,  # [Hz]
    contamination_amplitude_relative_range,  # IMPORTANT
    contamination_amplitude_relative_variation_width,
    mask,
    min_distance_between_frequencies=1.5,  # Hz
    frequency_lp_filter_duration=0.5,  # [s]
    amplitude_lp_filter_duration=0.5,  # [s]
):

    if (
        contamination_frequency_range[0]
        > spindle_frequency_range[0] - min_distance_between_frequencies
    ):
        raise ValueError(
            "Contamination interval %s Hz incompatible with spindle interval %s Hz and min distance %s Hz"
            % (
                contamination_frequency_range,
                spindle_frequency_range,
                min_distance_between_frequencies,
            )
        )

    # Prepare window
    window_min_size = int(fs * duration_range[0])
    window_max_size = int(fs * duration_range[1])
    window = random_window_tf(signal_size, window_min_size, window_max_size)
    window = mask * window if (mask is not None) else window

    part_to_remove = bandpass_tf(signal, fs, bandstop_cutoff[0], bandstop_cutoff[1])

    base_sigma_wave, sigma_central_amp, sigma_central_freq = generate_base_oscillation(
        signal_size,
        fs,
        spindle_frequency_range[0],
        spindle_frequency_range[1],
        spindle_frequency_variation_width,
        spindle_amplitude_absolute_range[0],
        spindle_amplitude_absolute_range[1],
        spindle_amplitude_relative_variation_width,
        frequency_lp_filter_duration,
        amplitude_lp_filter_duration,
    )

    contamination_upper_freq = tf.math.minimum(
        float(contamination_frequency_range[1]),
        sigma_central_freq - min_distance_between_frequencies,
    )
    base_contamination_wave, _, _ = generate_base_oscillation(
        signal_size,
        fs,
        contamination_frequency_range[0],
        contamination_upper_freq,
        contamination_frequency_variation_width,
        contamination_amplitude_relative_range[0] * sigma_central_amp,
        contamination_amplitude_relative_range[1] * sigma_central_amp,
        contamination_amplitude_relative_variation_width,
        frequency_lp_filter_duration,
        amplitude_lp_filter_duration,
    )

    # Total wave
    generated_wave = window * (
        -part_to_remove + base_sigma_wave + base_contamination_wave
    )
    return generated_wave
