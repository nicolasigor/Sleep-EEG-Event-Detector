from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pprint import pprint

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import tensorflow_probability as tfp

from src.common import checks, constants
from src.nn import wave_augment


def random_waves_wrapper(
        feat,
        label,
        probability,
        fs,
        params_dict_list
):
    """Addition of waves randomly generated in given frequency bands"""
    checks.check_valid_range(probability, 'probability', [0, 1])
    with tf.variable_scope('random_waves'):
        uniform_random = tf.random.uniform([], 0.0, 1.0)
        aug_condition = tf.less(uniform_random, probability)
        new_feat = tf.cond(
            aug_condition,
            lambda: random_waves(feat, label, fs, params_dict_list),
            lambda: feat
        )
    return new_feat


def random_waves(
        feat,
        label,
        fs,
        params_dict_list
):
    with tf.variable_scope('random_waves_generator'):
        mask_keep_events = wave_augment.generate_soft_mask_from_labels_tf(label, fs, use_background=False)
        mask_keep_background = 1.0 - mask_keep_events
        mask_map_dict = {
            constants.MASK_KEEP_EVENTS: mask_keep_events,
            constants.MASK_KEEP_BACKGROUND: mask_keep_background,
            constants.MASK_NONE: None,
        }
        feat_size = feat.get_shape().as_list()[0]
        for params_dict in params_dict_list:
            my_params = params_dict.copy()
            my_params["mask"] = mask_map_dict[params_dict["mask"]]
            random_wave = wave_augment.generate_wave_tf(feat_size, fs, **my_params)
            feat = random_wave + feat
    return feat


def random_anti_waves_wrapper(
        feat,
        label,
        probability,
        fs,
        params_dict_list
):
    """Random attenuation of waves in given frequency bands"""
    checks.check_valid_range(probability, 'probability', [0, 1])
    with tf.variable_scope('random_anti_waves'):
        uniform_random = tf.random.uniform([], 0.0, 1.0)
        aug_condition = tf.less(uniform_random, probability)
        new_feat = tf.cond(
            aug_condition,
            lambda: random_anti_waves(feat, label, fs, params_dict_list),
            lambda: feat
        )
    return new_feat


def random_anti_waves(
        feat,
        label,
        fs,
        params_dict_list
):
    with tf.variable_scope('random_anti_waves_generator'):
        mask_keep_events = wave_augment.generate_soft_mask_from_labels_tf(label, fs, use_background=False)
        mask_keep_background = 1.0 - mask_keep_events
        mask_map_dict = {
            constants.MASK_KEEP_EVENTS: mask_keep_events,
            constants.MASK_KEEP_BACKGROUND: mask_keep_background,
            constants.MASK_NONE: None,
        }
        feat_size = feat.get_shape().as_list()[0]
        for params_dict in params_dict_list:
            my_params = params_dict.copy()
            my_params["mask"] = mask_map_dict[params_dict["mask"]]
            random_anti_wave = wave_augment.generate_anti_wave_tf(feat, feat_size, fs, **my_params)
            feat = feat + random_anti_wave
    return feat


def false_spindles_single_contamination_wrapper(
        feat,
        label,
        probability,
        fs,
        params_dict
):
    """Addition of false spindles randomly generated with contamination in given frequency band"""
    checks.check_valid_range(probability, 'probability', [0, 1])
    with tf.variable_scope('false_spindles_single_cont'):
        uniform_random = tf.random.uniform([], 0.0, 1.0)
        aug_condition = tf.less(uniform_random, probability)
        new_feat = tf.cond(
            aug_condition,
            lambda: false_spindles_single_contamination(feat, label, fs, params_dict),
            lambda: feat
        )
    return new_feat


def false_spindles_single_contamination(
        feat,
        label,
        fs,
        params_dict
):
    with tf.variable_scope('false_spindles_single_cont_generator'):
        feat_size = feat.get_shape().as_list()[0]
        my_params = params_dict.copy()
        mask_keep_background = wave_augment.generate_soft_mask_from_labels_tf(label, fs, use_background=True)
        my_params["mask"] = mask_keep_background
        false_spindle = wave_augment.generate_false_spindle_single_contamination(feat, feat_size, fs, **my_params)
        new_feat = false_spindle + feat
    return new_feat


def gaussian_noise(feat, probability, std):
    """Noise is relative to each value"""
    checks.check_valid_range(probability, 'probability', [0, 1])
    with tf.variable_scope('gaussian_noise'):
        uniform_random = tf.random.uniform([], 0.0, 1.0)
        aug_condition = tf.less(uniform_random, probability)
        new_feat = tf.cond(
            aug_condition,
            lambda: feat * (1.0 + tf.random.normal(
                tf.shape(feat), mean=0.0, stddev=std)),
            lambda: feat
        )
    return new_feat


def independent_gaussian_noise(feat, probability, std):
    """Noise is AWGN"""
    checks.check_valid_range(probability, 'probability', [0, 1])
    with tf.variable_scope('independent_gaussian_noise'):
        print("Using AWGN std %1.6f with probability %1.2f" % (std, probability))
        uniform_random = tf.random.uniform([], 0.0, 1.0)
        aug_condition = tf.less(uniform_random, probability)
        new_feat = tf.cond(
            aug_condition,
            lambda: feat + tf.random.normal(
                tf.shape(feat), mean=0.0, stddev=std),
            lambda: feat
        )
    return new_feat


def independent_uniform_noise(feat, probability, intensity):
    """Noise is an independent uniform and additive noise"""
    checks.check_valid_range(probability, 'probability', [0, 1])
    if intensity is None:
        raise ValueError("Independent uniform noise requires setting its intensity, but it is 'None'.")
    with tf.variable_scope('independent_gaussian_noise'):
        print("Using uniform add noise intensity %1.6f with probability %1.2f" % (intensity, probability))
        uniform_random = tf.random.uniform([], 0.0, 1.0)
        aug_condition = tf.less(uniform_random, probability)
        new_feat = tf.cond(
            aug_condition,
            lambda: feat + tf.random.uniform(tf.shape(feat), -intensity, intensity),
            lambda: feat
        )
    return new_feat


def rescale_normal(feat, probability, std):
    checks.check_valid_range(probability, 'probability', [0, 1])
    with tf.variable_scope('rescale_normal'):
        print("Using RandomScalingNorma std %1.6f and probability %1.2f" % (std, probability))
        uniform_random = tf.random.uniform([], 0.0, 1.0)
        aug_condition = tf.less(uniform_random, probability)
        new_feat = tf.cond(
            aug_condition,
            lambda: feat * tf.random.normal([], mean=1.0, stddev=std),
            lambda: feat
        )
    return new_feat


def rescale_uniform(feat, probability, intensity):
    # Intensity is a float number representing fraction.
    checks.check_valid_range(probability, 'probability', [0, 1])
    with tf.variable_scope('rescale_uniform'):
        print("Using RandomScalingUniform intensity %1.6f and probability %1.2f" % (intensity, probability))
        uniform_random = tf.random.uniform([], 0.0, 1.0)
        aug_condition = tf.less(uniform_random, probability)
        new_feat = tf.cond(
            aug_condition,
            lambda: feat * tf.random.uniform([], 1.0 - intensity, 1.0 + intensity),
            lambda: feat
        )
    return new_feat


def elastic_1d_deformation_wrapper(feat, label, probability, fs, alpha, sigma):
    checks.check_valid_range(probability, 'probability', [0, 1])
    with tf.variable_scope('elastic_deform'):
        uniform_random = tf.random.uniform([], 0.0, 1.0)
        aug_condition = tf.less(uniform_random, probability)
        new_feat, new_label = tf.cond(
            aug_condition,
            lambda: elastic_1d_deformation(feat, label, fs, alpha, sigma),
            lambda: (feat, label)
        )
    return new_feat, new_label


def elastic_1d_deformation(feat, label, fs, alpha=0.2, sigma=0.05):
    """Transforms the given feat and label using elastic deformation.
    This implementation is intended to be used on a single example,
    and follows the description in:
        Simard, Steinkraus and Platt, "Best Practices for
           Convolutional Neural Networks applied to Visual Document
           Analysis", in Proc. of the International Conference on Document
           Analysis and Recognition, 2003.
    In this paper, the process is as follows:
        "The image deformations were created by first generating random
        displacement fields, that is ∆x(x,y) = rand(-1,+1) and
        ∆y(x,y)=rand(-1,+1), where rand(-1,+1) is a random
        number between -1 and +1, generated with a uniform
        distribution. The fields ∆x and ∆y are then convolved
        with a Gaussian of standard deviation sigma (in pixels). If sigma
        is large, the resulting values are very small because the
        random values average 0. If we normalize the displacement field
        (to a norm of 1), the field is then close to constant, with a random
        direction. If sigma is small, the field looks like a completely
        random field after normalization (as depicted in Figure 2, top
        right). For intermediate sigma values, the displacement fields look
        like elastic deformation, where sigma is the elasticity coefficient.
        The displacement fields are then multiplied by a scaling
        factor alpha that controls the intensity of the deformation."
    Args:
        feat: (tensor) 1D tensor of shape
            [time_len] corresponding to a single signal.
        label: (tensor) 1D tensor of the same shape as feat,
            corresponding to the class.
        fs: (float) sampling frequency
        alpha: (float) Scaling factor of the transformation (in seconds)
        sigma: (float) Elasticity coefficient of the transformation (in seconds)
    Returns:
        new_feat: (tensor) 1D tensor of the same shape as feat,
            corresponding to the transformed signal.
        new_label: (tensor) 1D tensor of the same shape as label,
            corresponding to the transformed label.
    """
    # Transform to number of samples
    alpha = alpha * fs
    sigma = sigma * fs

    with tf.device('/cpu:0'):
        with tf.variable_scope('elastic'):
            # Input shape
            input_dim = tf.shape(feat)

            # Random fields
            dx_random_fields = tf.random_uniform(shape=input_dim, minval=-1.0,
                                                 maxval=1.0)
            # Gaussian filtration and scaling
            kernel = gaussian_kernel(sigma, truncate=4.0)
            flow = alpha * apply_kernel1d(dx_random_fields, kernel)
            # Elastic deformation coordinates
            flow = tf.expand_dims(flow, axis=0)  # [1, time_len]

            # Apply transformation
            # Stack inputs along channel dimension, and add dummy batch dim
            feat_tensor = tf.cast(feat, tf.float32)[tf.newaxis, :]
            label_tensor = tf.cast(label, tf.float32)[tf.newaxis, :]
            stacked_input = tf.stack([feat_tensor, label_tensor], axis=2)
            stacked_output = warp_1d(stacked_input, flow)

            # Unstack and remove dummy batch dimension
            new_feat = tf.squeeze(stacked_output[..., 0])
            new_label = tf.squeeze(stacked_output[..., 1])
            # Make the marks integers
            new_label = tf.cast(tf.round(new_label), tf.int32)
    return new_feat, new_label


def gaussian_kernel(sigma, truncate=4.0):
    """Returns a gaussian kernel of shape [height, width]."""
    with tf.device('/cpu:0'):
        with tf.variable_scope('get_gaussian_kernel'):
            d = tfp.distributions.Normal(0.0, 1.0 * sigma)
            size = int(truncate * sigma)
            gauss_kernel = d.prob(
                tf.range(start=-size, limit=size + 1, dtype=tf.float32))
            gauss_kernel = gauss_kernel / tf.reduce_sum(gauss_kernel)  # Normalize
    return gauss_kernel


def apply_kernel1d(signal, kernel):
    """Applies a 1d kernel over a signal."""
    with tf.device('/cpu:0'):
        with tf.variable_scope('apply_gaussian_kernel'):
            kernel = kernel[
                ..., tf.newaxis, tf.newaxis, tf.newaxis]  # Proper shape for conv2d
            signal = signal[
                tf.newaxis, ..., tf.newaxis, tf.newaxis]  # Proper shape for conv2d

            new_signal = tf.nn.conv2d(
                signal, kernel, strides=[1, 1, 1, 1], padding='SAME')
            # Return to 2d tensor
            new_signal = tf.squeeze(new_signal)
    return new_signal


def warp_1d(signal, flow):
    """
    Per-sample warping

    Apply a non-linear warp to the signal, where the warp is specified by a dense
    flow field of offset vectors that define the correspondences of pixel values
    in the logits signal back to locations in the source signal. Specifically, the
    pixel value at logits[b, i, c] is images[b, i - flow[b, i], c].

    The locations specified by this formula do not necessarily map to an int
    index. Therefore, the pixel value is obtained by linear
    interpolation of the nearest samples around
    (b, i - flow[b, i]). For locations outside
    of the signal, we use the nearest pixel values at the signal boundary.

    Args:
        signal: 3-D float `Tensor` with shape `[batch, time_len, channels]`.
        flow: A 2-D float `Tensor` with shape `[batch, time_len]`.

      Returns:
        A 3-D float `Tensor` with shape`[batch, time_len, channels]`
          and same type as input signal.
    """
    with tf.device('/cpu:0'):
        batch_size, time_len, channels = signal.get_shape().as_list()
        grid_x = math_ops.cast(math_ops.range(time_len), flow.dtype)
        batched_grid = array_ops.expand_dims(grid_x, axis=0)
        query_points_on_grid = batched_grid - flow

        # Now we retrieve query values using interpolation
        interpolated_signal = tfp.math.interp_regular_1d_grid(
            query_points_on_grid, grid_x[0], grid_x[time_len - 1], signal, axis=1)

    return interpolated_signal
