"""Module that implements the CWT using a trainable complex morlet wavelet"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


from src.data.utils import get_kernel


def compute_cwt(
        inputs,
        fb_list,
        fs,
        lower_freq,
        upper_freq,
        n_scales,
        size_factor=1.0,
        flattening=False,
        border_crop=0,
        stride=1,
        trainable=False):
    """Computes the CWT of a batch of signals with the complex Morlet wavelet.
    Please refer to the documentation of compute_wavelets and apply_wavelets to
    see the description of the parameters.
    """
    wavelets, _ = compute_wavelets(
        fb_list=fb_list,
        fs=fs,
        lower_freq=lower_freq,
        upper_freq=upper_freq,
        n_scales=n_scales,
        size_factor=size_factor,
        flattening=flattening,
        trainable=trainable,
        name='cmorlet')
    cwt = apply_wavelets(
        inputs=inputs,
        wavelets=wavelets,
        border_crop=border_crop,
        stride=stride,
        name='cwt')
    return cwt, wavelets


def compute_cwt_rectangular(
        inputs,
        fb_list,
        fs,
        lower_freq,
        upper_freq,
        n_scales,
        size_factor=1.0,
        flattening=False,
        border_crop=0,
        stride=1,
        trainable=False):
    """Computes the CWT of a batch of signals with the complex Morlet wavelet.
    Please refer to the documentation of compute_wavelets and apply_wavelets to
    see the description of the parameters.
    """
    wavelets, _ = compute_wavelets(
        fb_list=fb_list,
        fs=fs,
        lower_freq=lower_freq,
        upper_freq=upper_freq,
        n_scales=n_scales,
        size_factor=size_factor,
        flattening=flattening,
        trainable=trainable,
        name='cmorlet')
    cwt = apply_wavelets_rectangular(
        inputs=inputs,
        wavelets=wavelets,
        border_crop=border_crop,
        stride=stride,
        name='cwt')
    return cwt, wavelets


def compute_wavelets_noisy(
        fb_list,
        fs,
        lower_freq,
        upper_freq,
        n_scales,
        training_flag,
        noise_intensity,
        size_factor=1.0,
        flattening=False,
        trainable=False,
        expansion_factor=1.0,
        trainable_expansion_factor=False,
        name=None):
    """
    Computes the complex morlet wavelets

    This function computes the complex morlet wavelet defined as:
    PSI(k) = (pi*Fb)^(-0.5) * exp(i*2*pi*Fc*k) * exp(-(k^2)/Fb)
    It supports several values of Fb at once, while Fc is fixed to 1 since we
    can change the frequency of the wavelets by changing the scale. Note that
    greater Fb values will lead to more duration of the wavelet in time,
    leading to better frequency resolution but worse time resolution.
    Scales will be automatically computed from the given frequency range and the
    number of desired scales. The scales  will increase exponentially.

    Args:
        fb_list: (list of floats) list of values for Fb (one for each scalogram)
        fs: (float) Sampling frequency of the signals of interest.
        lower_freq: (float) Lower frequency to be considered for the scalogram.
        upper_freq: (float) Upper frequency to be considered for the scalogram.
        n_scales: (int) Number of scales to cover the frequency range.
        training_flag: (boolean) whether it is training phase.
        noise_intensity: (float) intensity of the noise in the scales during training.
        flattening: (Optional, boolean, defaults to False) If True, each wavelet
            will be multiplied by its corresponding frequency, to avoid having
            too large coefficients for low frequency ranges, since it is
            common for natural signals to have a spectrum whose power decays
            roughly like 1/f.
        size_factor: (Optional, float, defaults to 1.0) Factor by which the
            size of the kernels will be increased with respect to the original
            size.
        trainable: (Optional, boolean, defaults to False) If True, the fb params
            will be trained with backprop.
        expansion_factor: (Optional, float between 0 and 1, defaults to 1)
            interpolates between STFT (0) and CWT (1) behavior of the kernel width.
        trainable_expansion_factor: (Optional, boolean, default to False)
            If true, the expansion factor will be trained with backprop.
        name: (Optional, string, defaults to None) A name for the operation.

    Returns:
        wavelets: (list of tuples of arrays) A list of computed wavelet banks.
        frequencies: (1D array) Array of frequencies for each scale.
    """

    # Checking
    if lower_freq > upper_freq:
        raise ValueError("lower_freq should be lower than upper_freq")
    if lower_freq < 0:
        raise ValueError("Expected positive lower_freq.")

    # Generate initial and last scale
    s_0 = 1 / upper_freq
    s_n = 1 / lower_freq

    # Generate the array of scales
    base = np.power(s_n / s_0, 1 / (n_scales - 1))
    scales = s_0 * np.power(base, np.arange(n_scales))
    scales = scales.astype(np.float32)

    # Generate the frequency range
    frequencies = 1 / scales
    noise_intensity = 0.0 if (noise_intensity is None) else noise_intensity
    print("CWT noise intensity:", noise_intensity)
    with tf.variable_scope(name):
        print("CWT expansion factor:", expansion_factor)
        if trainable_expansion_factor:
            print("Expansion factor trainable")
            expansion_factor = np.clip(expansion_factor, a_min=0.02, a_max=0.98)
            q_logit = np.log(expansion_factor / (1.0 - expansion_factor))
            q_logit_tensor = tf.Variable(
                initial_value=q_logit, trainable=True, name='q_logit', dtype=tf.float32)
            q = tf.nn.sigmoid(q_logit_tensor)
            tf.summary.scalar('expansion_factor', q)
        else:
            print("Expansion factor NOT trainable")
            q = tf.cast(expansion_factor, tf.float32)

        # Generate the wavelets
        wavelets = []
        for j, fb in enumerate(fb_list):
            # Trainable fb value
            # (we enforce positive number and avoids zero division)
            print("Using initial wavelet width %s" % fb)
            fb_tensor = tf.Variable(
                initial_value=fb, trainable=trainable, name='fb_%d' % j, dtype=tf.float32)
            fb_tensor = tf.math.abs(fb_tensor) + 1e-4  # Ensure positivity
            tf.summary.scalar('fb_%d' % j, fb_tensor)
            # We will make a bigger wavelet in case fb grows
            # Note that for the size of the wavelet we use the initial fb value.
            one_side = int(size_factor * s_n * fs * np.sqrt(4.5 * fb))
            kernel_size = 2 * one_side + 1
            k_array = np.arange(kernel_size, dtype=np.float32) - one_side
            k_array = k_array / fs  # Time units
            # Wavelet bank shape: 1, kernel_size, 1, n_scales
            wavelet_bank_real = []
            wavelet_bank_imag = []
            # wavelet_bank_real = np.zeros((1, kernel_size, 1, n_scales))
            # wavelet_bank_imag = np.zeros((1, kernel_size, 1, n_scales))
            print("Applying scale noise at CWT (%d scales)" % n_scales)
            for i in range(n_scales):
                scale_original = scales[i]
                scale = tf.cond(
                    training_flag,
                    lambda: scale_original / tf.random.uniform([], 1.0 - noise_intensity, 1.0 + noise_intensity),
                    lambda: scale_original
                )
                scale_expanded = q * scale + (1.0 - q) * s_n
                norm_constant = tf.sqrt(np.pi * fb_tensor) * scale_expanded * fs / 2.0
                exp_term = tf.exp(-((k_array / scale_expanded) ** 2) / fb_tensor)
                kernel_base = exp_term / norm_constant
                kernel_real = kernel_base * tf.cos(2 * np.pi * k_array / scale)
                kernel_imag = kernel_base * tf.sin(2 * np.pi * k_array / scale)
                if flattening:
                    kernel_real = kernel_real * frequencies[i]
                    kernel_imag = kernel_imag * frequencies[i]
                # wavelet_bank_real[0, :, 0, i] = kernel_real
                # wavelet_bank_imag[0, :, 0, i] = kernel_imag
                wavelet_bank_real.append(kernel_real)
                wavelet_bank_imag.append(kernel_imag)
            # Stack wavelets (shape = kernel_size, n_scales)
            wavelet_bank_real = tf.stack(wavelet_bank_real, axis=-1)
            wavelet_bank_imag = tf.stack(wavelet_bank_imag, axis=-1)
            # Give it proper shape for convolutions
            # -> shape: 1, kernel_size, n_scales
            wavelet_bank_real = tf.expand_dims(wavelet_bank_real, axis=0)
            wavelet_bank_imag = tf.expand_dims(wavelet_bank_imag, axis=0)
            # -> shape: 1, kernel_size, 1, n_scales
            wavelet_bank_real = tf.expand_dims(wavelet_bank_real, axis=2)
            wavelet_bank_imag = tf.expand_dims(wavelet_bank_imag, axis=2)
            wavelets.append((wavelet_bank_real, wavelet_bank_imag))
    return wavelets, frequencies


def compute_wavelets(
        fb_list,
        fs,
        lower_freq,
        upper_freq,
        n_scales,
        size_factor=1.0,
        flattening=False,
        trainable=False,
        expansion_factor=1.0,
        trainable_expansion_factor=False,
        name=None):
    """
    Computes the complex morlet wavelets

    This function computes the complex morlet wavelet defined as:
    PSI(k) = (pi*Fb)^(-0.5) * exp(i*2*pi*Fc*k) * exp(-(k^2)/Fb)
    It supports several values of Fb at once, while Fc is fixed to 1 since we
    can change the frequency of the wavelets by changing the scale. Note that
    greater Fb values will lead to more duration of the wavelet in time,
    leading to better frequency resolution but worse time resolution.
    Scales will be automatically computed from the given frequency range and the
    number of desired scales. The scales  will increase exponentially.

    Args:
        fb_list: (list of floats) list of values for Fb (one for each scalogram)
        fs: (float) Sampling frequency of the signals of interest.
        lower_freq: (float) Lower frequency to be considered for the scalogram.
        upper_freq: (float) Upper frequency to be considered for the scalogram.
        n_scales: (int) Number of scales to cover the frequency range.
        flattening: (Optional, boolean, defaults to False) If True, each wavelet
            will be multiplied by its corresponding frequency, to avoid having
            too large coefficients for low frequency ranges, since it is
            common for natural signals to have a spectrum whose power decays
            roughly like 1/f.
        size_factor: (Optional, float, defaults to 1.0) Factor by which the
            size of the kernels will be increased with respect to the original
            size.
        trainable: (Optional, boolean, defaults to False) If True, the fb params
            will be trained with backprop.
        expansion_factor: (Optional, float between 0 and 1, defaults to 1)
            interpolates between STFT (0) and CWT (1) behavior of the kernel width.
        trainable_expansion_factor: (Optional, boolean, default to False)
            If true, the expansion factor will be trained with backprop.
        name: (Optional, string, defaults to None) A name for the operation.

    Returns:
        wavelets: (list of tuples of arrays) A list of computed wavelet banks.
        frequencies: (1D array) Array of frequencies for each scale.
    """

    # Checking
    if lower_freq > upper_freq:
        raise ValueError("lower_freq should be lower than upper_freq")
    if lower_freq < 0:
        raise ValueError("Expected positive lower_freq.")

    # Generate initial and last scale
    s_0 = 1 / upper_freq
    s_n = 1 / lower_freq

    # Generate the array of scales
    base = np.power(s_n / s_0, 1 / (n_scales - 1))
    scales = s_0 * np.power(base, np.arange(n_scales))

    # Generate the frequency range
    frequencies = 1 / scales

    with tf.variable_scope(name):
        print("CWT expansion factor:", expansion_factor)
        if trainable_expansion_factor:
            expansion_factor = np.clip(expansion_factor, a_min=0.02, a_max=0.98)
            q_logit = np.log(expansion_factor / (1.0 - expansion_factor))
            q_logit_tensor = tf.Variable(
                initial_value=q_logit, trainable=True, name='q_logit', dtype=tf.float32)
            q = tf.nn.sigmoid(q_logit_tensor)
            tf.summary.scalar('expansion_factor', q)
        else:
            q = tf.cast(expansion_factor, tf.float32)

        # Generate the wavelets
        wavelets = []
        for j, fb in enumerate(fb_list):
            # Trainable fb value
            # (we enforce positive number and avoids zero division)
            fb_tensor = tf.Variable(
                initial_value=fb, trainable=trainable, name='fb_%d' % j, dtype=tf.float32)
            fb_tensor = tf.math.abs(fb_tensor)  # Ensure positivity
            tf.summary.scalar('fb_%d' % j, fb_tensor)
            # We will make a bigger wavelet in case fb grows
            # Note that for the size of the wavelet we use the initial fb value.
            one_side = int(size_factor * s_n * fs * np.sqrt(4.5 * fb))
            kernel_size = 2 * one_side + 1
            k_array = np.arange(kernel_size, dtype=np.float32) - one_side
            k_array = k_array / fs  # Time units
            # Wavelet bank shape: 1, kernel_size, 1, n_scales
            wavelet_bank_real = []
            wavelet_bank_imag = []
            # wavelet_bank_real = np.zeros((1, kernel_size, 1, n_scales))
            # wavelet_bank_imag = np.zeros((1, kernel_size, 1, n_scales))
            for i in range(n_scales):
                scale = scales[i]
                scale_expanded = q * scale + (1.0 - q) * s_n
                norm_constant = tf.sqrt(np.pi * fb_tensor) * scale_expanded * fs / 2.0
                exp_term = tf.exp(-((k_array / scale_expanded) ** 2) / fb_tensor)
                kernel_base = exp_term / norm_constant
                kernel_real = kernel_base * np.cos(2 * np.pi * k_array / scale)
                kernel_imag = kernel_base * np.sin(2 * np.pi * k_array / scale)
                if flattening:
                    kernel_real = kernel_real * frequencies[i]
                    kernel_imag = kernel_imag * frequencies[i]
                # wavelet_bank_real[0, :, 0, i] = kernel_real
                # wavelet_bank_imag[0, :, 0, i] = kernel_imag
                wavelet_bank_real.append(kernel_real)
                wavelet_bank_imag.append(kernel_imag)
            # Stack wavelets (shape = kernel_size, n_scales)
            wavelet_bank_real = tf.stack(wavelet_bank_real, axis=-1)
            wavelet_bank_imag = tf.stack(wavelet_bank_imag, axis=-1)
            # Give it proper shape for convolutions
            # -> shape: 1, kernel_size, n_scales
            wavelet_bank_real = tf.expand_dims(wavelet_bank_real, axis=0)
            wavelet_bank_imag = tf.expand_dims(wavelet_bank_imag, axis=0)
            # -> shape: 1, kernel_size, 1, n_scales
            wavelet_bank_real = tf.expand_dims(wavelet_bank_real, axis=2)
            wavelet_bank_imag = tf.expand_dims(wavelet_bank_imag, axis=2)
            wavelets.append((wavelet_bank_real, wavelet_bank_imag))
    return wavelets, frequencies


def apply_wavelets(
        inputs,
        wavelets,
        border_crop=0,
        stride=1,
        name=None):
    """
    CWT layer implementation in Tensorflow that returns the scalograms tensor.

    Implementation of CWT in Tensorflow, aimed at providing GPU acceleration.
    This layer use computed wavelets. It supports the computation of several
    scalograms. Different scalograms will be stacked along the channel axis.

    Args:
        inputs: (tensor) A batch of 1D tensors of shape [batch_size, time_len].
        wavelets: (list of tuples of arrays) A list of computed wavelet banks.
        border_crop: (Optional, int, defaults to 0) Non-negative integer that
            specifies the number of samples to be removed at each border at the
            end. This parameter allows to input a longer signal than the final
            desired size to remove border effects of the CWT.
        stride: (Optional, int, defaults to 1) The stride of the sliding window
            across the input. Default is 1.
        name: (Optional, string, defaults to None) A name for the operation.

    Returns:
        Scalogram tensor.
    """

    n_scalograms = len(wavelets)

    # Generate the scalograms
    border_crop = int(border_crop/stride)
    start = border_crop
    if border_crop <= 0:
        end = None
    else:
        end = -border_crop

    if name is None:
        name = "cwt"
    with tf.variable_scope(name):
        # Reshape input [batch, time_len] -> [batch, 1, time_len, 1]
        inputs_expand = tf.expand_dims(inputs, axis=1)
        inputs_expand = tf.expand_dims(inputs_expand, axis=3)
        scalograms_list = []
        for j in range(n_scalograms):
            with tf.name_scope('%s_%d' % (name, j)):
                bank_real, bank_imag = wavelets[j]  # n_scales filters each
                bank_imag = -bank_imag  # Conjugation
                out_real = tf.nn.conv2d(
                    input=inputs_expand, filter=bank_real,
                    strides=[1, 1, stride, 1], padding="SAME")
                out_imag = tf.nn.conv2d(
                    input=inputs_expand, filter=bank_imag,
                    strides=[1, 1, stride, 1], padding="SAME")
                out_real_crop = out_real[:, :, start:end, :]
                out_imag_crop = out_imag[:, :, start:end, :]
                out_power = tf.sqrt(tf.square(out_real_crop)
                                    + tf.square(out_imag_crop))
                out_angle = tf.atan2(out_imag_crop, out_real_crop)
                out_concat = tf.concat([out_power, out_angle], axis=1)
                # [batch, 2, time_len, n_scales]->[batch, time_len, n_scales, 2]
                single_scalogram = tf.transpose(out_concat, perm=[0, 2, 3, 1])
                scalograms_list.append(single_scalogram)
        # Get all scalograms in shape [batch, time_len, n_scales,2*n_scalograms]
        scalograms = tf.concat(scalograms_list, -1)
    return scalograms


def apply_wavelets_rectangular(
        inputs,
        wavelets,
        border_crop=0,
        stride=1,
        name=None):
    """
    CWT layer implementation in Tensorflow that returns the scalograms tensor.

    Implementation of CWT in Tensorflow, aimed at providing GPU acceleration.
    This layer use computed wavelets. It supports the computation of several
    scalograms. Different scalograms will be stacked along the channel axis.

    Args:
        inputs: (tensor) A batch of 1D tensors of shape [batch_size, time_len].
        wavelets: (list of tuples of arrays) A list of computed wavelet banks.
        border_crop: (Optional, int, defaults to 0) Non-negative integer that
            specifies the number of samples to be removed at each border at the
            end. This parameter allows to input a longer signal than the final
            desired size to remove border effects of the CWT.
        stride: (Optional, int, defaults to 1) The stride of the sliding window
            across the input. Default is 1.
        name: (Optional, string, defaults to None) A name for the operation.

    Returns:
        Scalogram tensor.
    """

    n_scalograms = len(wavelets)

    # Generate the scalograms
    border_crop = int(border_crop/stride)
    start = border_crop
    if border_crop <= 0:
        end = None
    else:
        end = -border_crop

    if name is None:
        name = "cwt"
    with tf.variable_scope(name):
        # Reshape input [batch, time_len] -> [batch, 1, time_len, 1]
        inputs_expand = tf.expand_dims(inputs, axis=1)
        inputs_expand = tf.expand_dims(inputs_expand, axis=3)
        scalograms_list = []
        for j in range(n_scalograms):
            with tf.name_scope('%s_%d' % (name, j)):
                bank_real, bank_imag = wavelets[j]  # n_scales filters each
                bank_imag = -bank_imag  # Conjugation
                out_real = tf.nn.conv2d(
                    input=inputs_expand, filter=bank_real,
                    strides=[1, 1, stride, 1], padding="SAME")
                out_imag = tf.nn.conv2d(
                    input=inputs_expand, filter=bank_imag,
                    strides=[1, 1, stride, 1], padding="SAME")
                out_real_crop = out_real[:, :, start:end, :]
                out_imag_crop = out_imag[:, :, start:end, :]
                out_concat = tf.concat([out_real_crop, out_imag_crop], axis=1)
                # [batch, 2, time_len, n_scales]->[batch, time_len, n_scales, 2]
                single_scalogram = tf.transpose(out_concat, perm=[0, 2, 3, 1])
                scalograms_list.append(single_scalogram)
        # Get all scalograms in shape [batch, time_len, n_scales,2*n_scalograms]
        scalograms = tf.concat(scalograms_list, -1)
    return scalograms


def compute_sigma_band(
        inputs,
        fs,
        ntaps=41,
        central_freq=13,
        border_crop=0,
        stride=1):
    kernel = get_kernel(ntaps, central_freq, fs)
    sigma_inputs = apply_kernel(inputs, kernel, border_crop, stride, 'sigma')
    return sigma_inputs


def apply_kernel(
        inputs,
        kernel,
        border_crop=0,
        stride=1,
        name=None):

    border_crop = int(border_crop / stride)
    start = border_crop
    if border_crop <= 0:
        end = None
    else:
        end = -border_crop

    if name is None:
        name = "sigma"
    with tf.variable_scope(name):
        # Reshape input [batch, time_len] -> [batch, 1, time_len, 1]
        inputs_expand = tf.expand_dims(inputs, axis=1)
        inputs_expand = tf.expand_dims(inputs_expand, axis=3)
        with tf.name_scope('filtering'):
            # Reshape kernel [kernel_size] -> [1, kernel_size, 1, 1]
            kernel_expand = np.reshape(kernel, newshape=(1, -1, 1, 1))
            out_filter = tf.nn.conv2d(
                input=inputs_expand, filter=kernel_expand,
                strides=[1, 1, stride, 1], padding="SAME")
            out_filter_crop = out_filter[:, :, start:end, :]
        # Remove extra dim
        outputs = tf.squeeze(out_filter_crop, axis=1, name="squeeze")
    return outputs
