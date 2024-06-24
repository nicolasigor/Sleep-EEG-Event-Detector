"""Module that defines losses to train a model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from sleeprnn.common import constants


def get_border_weights_quad(labels, amplitude, half_width):
    print(
        "Border weights QUADRATIC with A=%1.2f and half_s=%1.2f"
        % (amplitude, half_width)
    )
    with tf.variable_scope("loss_border_weights"):
        # Edge detector definition
        kernel_edge = [-0.5, 0, 0.5]
        kernel_edge = tf.constant(kernel_edge, dtype=tf.float32)
        # Quadratic window definition (1st order approx of Gauss)
        kernel_steps = np.arange(2 * half_width + 1) - half_width
        kernel_quad = (amplitude - 1) * (1 - (kernel_steps / half_width) ** 2)
        print("Border kernel", kernel_quad)
        kernel_quad = tf.constant(kernel_quad, dtype=tf.float32)
        # Prepare labels
        first_label = labels[:, 0:1]
        last_label = labels[:, -1:]
        labels_extended = tf.concat([first_label, labels, last_label], axis=1)
        labels_extended = tf.cast(labels_extended, dtype=tf.float32)
        # Prepare shapes for convolution
        kernel_edge = kernel_edge[:, tf.newaxis, tf.newaxis, tf.newaxis]
        kernel_quad = kernel_quad[:, tf.newaxis, tf.newaxis, tf.newaxis]
        labels_extended = labels_extended[:, :, tf.newaxis, tf.newaxis]
        # Filter
        output = labels_extended
        output = tf.nn.conv2d(
            output, kernel_edge, strides=[1, 1, 1, 1], padding="VALID"
        )
        output = tf.abs(output)
        output = tf.nn.conv2d(output, kernel_quad, strides=[1, 1, 1, 1], padding="SAME")
        output = 1.0 + output
        weights = output[:, :, 0, 0]
    return weights


def get_border_weights(labels, amplitude, half_width):
    print("Border weights with A=%1.2f and half_s=%1.2f" % (amplitude, half_width))
    with tf.variable_scope("loss_border_weights"):
        # Edge detector definition
        kernel_edge = [-0.5, 0, 0.5]
        kernel_edge = tf.constant(kernel_edge, dtype=tf.float32)
        # Gaussian window definition
        std_kernel = (2 * half_width + 1) / 6
        kernel_steps = np.arange(2 * half_width + 1) - half_width
        exp_term = -(kernel_steps**2) / (2 * (std_kernel**2))
        kernel_gauss = (amplitude - 1) * np.exp(exp_term)
        kernel_gauss = tf.constant(kernel_gauss, dtype=tf.float32)
        # Prepare labels
        first_label = labels[:, 0:1]
        last_label = labels[:, -1:]
        labels_extended = tf.concat([first_label, labels, last_label], axis=1)
        labels_extended = tf.cast(labels_extended, dtype=tf.float32)
        # Prepare shapes for convolution
        kernel_edge = kernel_edge[:, tf.newaxis, tf.newaxis, tf.newaxis]
        kernel_gauss = kernel_gauss[:, tf.newaxis, tf.newaxis, tf.newaxis]
        labels_extended = labels_extended[:, :, tf.newaxis, tf.newaxis]
        # Filter
        output = labels_extended
        output = tf.nn.conv2d(
            output, kernel_edge, strides=[1, 1, 1, 1], padding="VALID"
        )
        output = tf.abs(output)
        output = tf.nn.conv2d(
            output, kernel_gauss, strides=[1, 1, 1, 1], padding="SAME"
        )
        output = 1.0 + output
        weights = output[:, :, 0, 0]
    return weights


def get_anti_border_weights(labels, amplitude, half_width):
    print("Anti border weights with A=%1.2f and half_s=%1.2f" % (amplitude, half_width))
    if amplitude == 0:
        return 1.0
    with tf.variable_scope("loss__anti_border_weights"):
        # Edge detector definition
        kernel_edge = [-0.5, 0, 0.5]
        kernel_edge = tf.constant(kernel_edge, dtype=tf.float32)
        # Gaussian window definition
        std_kernel = (2 * half_width + 1) / 6
        kernel_steps = np.arange(2 * half_width + 1) - half_width
        exp_term = -(kernel_steps**2) / (2 * (std_kernel**2))
        kernel_gauss = np.exp(exp_term)
        kernel_gauss = tf.constant(kernel_gauss, dtype=tf.float32)
        # Prepare labels
        first_label = labels[:, 0:1]
        last_label = labels[:, -1:]
        labels_extended = tf.concat([first_label, labels, last_label], axis=1)
        labels_extended = tf.cast(labels_extended, dtype=tf.float32)
        # Prepare shapes for convolution
        kernel_edge = kernel_edge[:, tf.newaxis, tf.newaxis, tf.newaxis]
        kernel_gauss = kernel_gauss[:, tf.newaxis, tf.newaxis, tf.newaxis]
        labels_extended = labels_extended[:, :, tf.newaxis, tf.newaxis]
        # Filter
        output = labels_extended
        output = tf.nn.conv2d(
            output, kernel_edge, strides=[1, 1, 1, 1], padding="VALID"
        )
        output = tf.abs(output)
        output = tf.nn.conv2d(
            output, kernel_gauss, strides=[1, 1, 1, 1], padding="SAME"
        )
        # Prepare proper formula
        output = 1.0 - amplitude * output
        weights = output[:, :, 0, 0]
    return weights


def get_weights(logits, labels, class_weights):
    with tf.variable_scope("loss_weights"):
        if class_weights is None:
            print("No weight balancing")
            class_weights = tf.constant([1.0, 1.0], dtype=tf.float32)
            weights = tf.gather(class_weights, labels)
        elif class_weights == constants.BALANCED:
            print("Class freq as weight to balance")
            n_negative = tf.cast(tf.reduce_sum(1 - labels), tf.float32)
            n_positive = tf.cast(tf.reduce_sum(labels), tf.float32)
            total = n_negative + n_positive
            weight_negative = n_positive / total
            weight_positive = n_negative / total
            class_weights = tf.stack([weight_negative, weight_positive], axis=0)
            weights = tf.gather(class_weights, labels)
        elif class_weights == constants.BALANCED_DROP:
            print("Random negative class dropping to balance")
            n_negative = tf.cast(tf.reduce_sum(1 - labels), tf.float32)
            n_positive = tf.cast(tf.reduce_sum(labels), tf.float32)
            p_drop_negative = (n_negative - n_positive) / n_negative
            random_mask = tf.random.uniform(
                tf.shape(labels), minval=0.0, maxval=1.0, dtype=tf.float32
            )
            random_mask = random_mask + tf.cast(labels, tf.float32)
            weights = tf.cast(tf.math.greater(random_mask, p_drop_negative), tf.float32)
        elif class_weights == constants.BALANCED_DROP_V2:
            print("Random negative class dropping to balance (V2)")
            n_negative = tf.cast(tf.reduce_sum(1 - labels), tf.float32)
            n_positive = tf.cast(tf.reduce_sum(labels), tf.float32)

            # Add negative labels on the neighborhood of positive ones
            neighbor_radius = 4

            labels_2d = tf.expand_dims(tf.cast(labels, tf.float32), 1)
            labels_2d = tf.expand_dims(labels_2d, 3)
            spread_labels = tf.nn.conv2d(
                labels_2d,
                filter=np.ones((1, 2 * neighbor_radius + 1, 1, 1)),
                strides=[1, 1, 1, 1],
                padding="SAME",
            )
            spread_labels = tf.squeeze(spread_labels, [1, 3])
            spread_labels = tf.cast(tf.math.greater(spread_labels, 0), tf.float32)
            n_negative_already_added = tf.reduce_sum(spread_labels) - n_positive
            n_negative_to_add = tf.nn.relu(n_positive - n_negative_already_added)
            p_keep_negative = n_negative_to_add / (n_negative + 1e-3)  # in (0, 1)
            p_drop_negative = 1 - p_keep_negative

            random_mask = tf.random.uniform(
                tf.shape(labels), minval=0.0, maxval=1.0, dtype=tf.float32
            )
            random_mask = random_mask + spread_labels
            weights = tf.cast(tf.math.greater(random_mask, p_drop_negative), tf.float32)
        else:
            print("Balancing with weights", class_weights)
            class_weights = tf.constant(class_weights)
            weights = tf.gather(class_weights, labels)
    return weights


def weighted_cross_entropy_loss_hard(
    logits,
    labels,
    border_amplitude,
    border_half_width,
    class_weights,
    mix_weights_strategy,
):
    print("Using Weighted Cross Entropy Loss - with regular Cross-entropy")
    print("By-segment mean loss")
    print("Class weights:", class_weights)
    print("Borders A: %1.2f HW: %1.2f" % (border_amplitude, border_half_width))
    print("Mix weights strategy:", mix_weights_strategy)

    with tf.variable_scope(constants.WEIGHTED_CROSS_ENTROPY_LOSS_HARD):
        # Border weights
        weight_border = get_border_weights(labels, border_amplitude, border_half_width)

        # Class weights
        weight_label = get_weights(logits, labels, class_weights)

        # Combine weights
        if mix_weights_strategy == constants.MIX_WEIGHTS_MAX:
            weights = tf.maximum(weight_border, weight_label)
        elif mix_weights_strategy == constants.MIX_WEIGHTS_SUM:
            weights = weight_border + weight_label - 1.0
        elif mix_weights_strategy == constants.MIX_WEIGHTS_PRODUCT:
            weights = weight_border * weight_label
        else:
            raise ValueError('mix_strategy "%s" invalid' % mix_weights_strategy)

        # Weighted loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        )
        loss = weights * loss
        # First, we compute the weighted average for each segment independently
        loss = tf.reduce_sum(loss, axis=1) / tf.reduce_sum(weights, axis=1)
        # Now we average the segments
        loss = tf.reduce_mean(loss)
        # Summaries
        loss_summ = tf.summary.scalar("loss", loss)
    return loss, loss_summ


def weighted_cross_entropy_loss_soft(
    logits,
    labels,
    border_amplitude,
    border_half_width,
    soft_epsilon,
    class_weights,
    mix_weights_strategy,
):
    print("Using Weighted Cross Entropy Loss - with Soft-Clip Cross-entropy")
    print("By-segment mean loss")
    print("Class weights:", class_weights)
    print("Borders A: %1.2f HW: %1.2f" % (border_amplitude, border_half_width))
    print("Soft epsilon:", soft_epsilon)
    print("Mix weights strategy:", mix_weights_strategy)

    with tf.variable_scope(constants.WEIGHTED_CROSS_ENTROPY_LOSS_SOFT):
        # Border weights
        weight_border = get_border_weights(labels, border_amplitude, border_half_width)

        # Class weights
        weight_label = get_weights(logits, labels, class_weights)

        # Combine weights
        if mix_weights_strategy == constants.MIX_WEIGHTS_MAX:
            weights = tf.maximum(weight_border, weight_label)
        elif mix_weights_strategy == constants.MIX_WEIGHTS_SUM:
            weights = weight_border + weight_label - 1.0
        elif mix_weights_strategy == constants.MIX_WEIGHTS_PRODUCT:
            weights = weight_border * weight_label
        else:
            raise ValueError('mix_strategy "%s" invalid' % mix_weights_strategy)

        # Cross-entropy loss with soft labels
        labels_onehot = tf.cast(tf.one_hot(labels, 2), dtype=tf.float32)
        smooth_labels = labels_onehot * (1.0 - 2.0 * soft_epsilon) + soft_epsilon
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=smooth_labels, logits=logits
        )
        offset = -tf.reduce_sum(smooth_labels * tf.log(smooth_labels + 1e-6), axis=2)
        loss = loss - offset
        # Set loss to zero when prediction is beyond soft_epsilon
        probabilities = tf.nn.softmax(logits)
        distance_to_hard_label = tf.reduce_sum(
            (probabilities - labels_onehot) ** 2, axis=2
        )
        distance_thr = 2 * (soft_epsilon**2)
        mask = tf.cast(
            tf.math.greater(distance_to_hard_label, distance_thr), tf.float32
        )
        loss = loss * mask
        # Weighted loss
        loss = weights * loss
        # First, we compute the weighted average for each segment independently
        loss = tf.reduce_sum(loss, axis=1) / tf.reduce_sum(weights, axis=1)
        # Now we average the segments
        loss = tf.reduce_mean(loss)
        # Summaries
        loss_summ = tf.summary.scalar("loss", loss)
    return loss, loss_summ


def masked_soft_focal_loss(
    logits, labels, masks, class_weights, focal_gamma, focal_eps, return_weights=False
):
    print("Using Masked Soft Focal Loss")
    print("By-segment mean loss")
    print("Class weights:", class_weights)
    print("Soft focal G: %s EPS: %s" % (focal_gamma, focal_eps))

    with tf.variable_scope(constants.MASKED_SOFT_FOCAL_LOSS):
        weight_label = get_weights(logits, labels, class_weights)
        if focal_eps == 1.0:
            # soft focal weight is disabled
            print("Soft focal behaviour reduced to regular cross-entropy (eps = 1)")
            weight_focal = 1.0
        else:
            # soft focal weights
            probabilities = tf.nn.softmax(logits)
            labels_onehot = tf.cast(tf.one_hot(labels, 2), dtype=tf.float32)
            proba_correct_class = tf.reduce_sum(
                probabilities * labels_onehot, axis=2
            )  # output shape [batch, time]
            focal_term = (1.0 - proba_correct_class) ** focal_gamma
            weight_focal = focal_eps + (1.0 - focal_eps) * focal_term

        weight_mask = tf.cast(masks, tf.float32)  # Shape [batch, time]
        # Samples with mask = 0 do not contribute to the final loss
        # This is equivalent to assign a weight of zero to those samples

        # Combine weights
        weights = weight_label * weight_focal * weight_mask

        # Weighted loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        )
        loss = weights * loss
        # First, we compute the weighted average for each segment independently
        loss = tf.reduce_sum(loss, axis=1) / tf.reduce_sum(weights, axis=1)
        # Now we average the segments
        loss = tf.reduce_mean(loss)
        # Summaries
        loss_summ = tf.summary.scalar("loss", loss)

        if return_weights:
            weights_dict = {
                "w_class": weight_label,
                "w_focal": weight_focal,
                "w_mask": weight_mask,
                "w_total": weights,
            }
            return loss, loss_summ, weights_dict
        else:
            return loss, loss_summ


def weighted_cross_entropy_loss_v5(
    logits,
    labels,
    class_weights,
    focal_gamma,
    focal_eps,
    anti_border_amplitude,
    anti_border_half_width,
    return_weights=False,
):
    print("Using Weighted Cross Entropy Loss v5 (anti-borders)")
    print("By-segment mean loss")
    print("Class weights:", class_weights)
    print("Soft focal G: %1.2f EPS: %1.2f" % (focal_gamma, focal_eps))
    print(
        "Anti Borders A: %1.2f HW: %1.2f"
        % (anti_border_amplitude, anti_border_half_width)
    )

    with tf.variable_scope(constants.WEIGHTED_CROSS_ENTROPY_LOSS_V5):
        # Border weights
        weight_anti_border = get_anti_border_weights(
            labels, anti_border_amplitude, anti_border_half_width
        )

        # Modified focal weights with class weights
        probabilities = tf.nn.softmax(logits)
        labels_onehot = tf.cast(tf.one_hot(labels, 2), dtype=tf.float32)
        proba_correct_class = tf.reduce_sum(
            probabilities * labels_onehot, axis=2
        )  # output shape [batch, time]
        focal_term = (1.0 - proba_correct_class) ** focal_gamma
        weight_focal = focal_eps + (1.0 - focal_eps) * focal_term
        weight_label = get_weights(logits, labels, class_weights)

        # Combine weights
        weights = weight_label * weight_focal * weight_anti_border

        # Weighted loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        )
        loss = weights * loss
        # First, we compute the weighted average for each segment independently
        loss = tf.reduce_sum(loss, axis=1) / tf.reduce_sum(weights, axis=1)
        # Now we average the segments
        loss = tf.reduce_mean(loss)
        # Summaries
        loss_summ = tf.summary.scalar("loss", loss)
        if return_weights:
            weights_dict = {
                "w_class": weight_label,
                "w_focal": weight_focal,
                "w_border": weight_anti_border,
                "w_total": weights,
            }
            return loss, loss_summ, weights_dict
        else:
            return loss, loss_summ


def weighted_cross_entropy_loss(
    logits,
    labels,
    border_amplitude,
    border_half_width,
    focal_gamma,
    focal_weight,
    class_weights,
    mix_weights_strategy,
):
    print("Using Weighted Cross Entropy Loss")
    print("By-segment mean loss")
    print("Class weights:", class_weights)
    print("Borders A: %1.2f HW: %1.2f" % (border_amplitude, border_half_width))
    print("Mod focal G: %1.2f W0: %1.2f" % (focal_gamma, focal_weight))
    print("Mix weights strategy:", mix_weights_strategy)

    with tf.variable_scope(constants.WEIGHTED_CROSS_ENTROPY_LOSS):
        # Border weights
        weight_border = get_border_weights(labels, border_amplitude, border_half_width)

        # Modified focal weights with class weights
        probabilities = tf.nn.softmax(logits)
        labels_onehot = tf.cast(tf.one_hot(labels, 2), dtype=tf.float32)
        proba_correct_class = tf.reduce_sum(
            probabilities * labels_onehot, axis=2
        )  # output shape [batch, time]
        focal_term = (1.0 - proba_correct_class) ** focal_gamma
        weight_focal_raw = 1.0 + (focal_weight - 1.0) * focal_term
        weight_label = get_weights(logits, labels, class_weights)
        weight_focal = weight_focal_raw * weight_label

        # Combine weights
        if mix_weights_strategy == constants.MIX_WEIGHTS_MAX:
            weights = tf.maximum(weight_border, weight_focal)
        elif mix_weights_strategy == constants.MIX_WEIGHTS_SUM:
            weights = weight_border + weight_focal - 1.0
        elif mix_weights_strategy == constants.MIX_WEIGHTS_PRODUCT:
            weights = weight_border * weight_focal
        else:
            raise ValueError('mix_strategy "%s" invalid' % mix_weights_strategy)

        # Weighted loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        )
        loss = weights * loss
        # First, we compute the weighted average for each segment independently
        loss = tf.reduce_sum(loss, axis=1) / tf.reduce_sum(weights, axis=1)
        # Now we average the segments
        loss = tf.reduce_mean(loss)
        # Summaries
        loss_summ = tf.summary.scalar("loss", loss)
    return loss, loss_summ


def weighted_cross_entropy_loss_v2(
    logits,
    labels,
    border_amplitude,
    border_half_width,
    focal_gamma,
    focal_weight,
    class_weights,
    mix_weights_strategy,
    prediction_variability_regularizer,
):
    print("Using Weighted Cross Entropy Loss V2")
    print("By-segment mean loss")
    print("Class weights:", class_weights)
    print("Borders A: %1.2f HW: %1.2f" % (border_amplitude, border_half_width))
    print("Mod focal G: %1.2f W0: %1.2f" % (focal_gamma, focal_weight))
    print("Mix weights strategy:", mix_weights_strategy)
    print("Pred. variability regularizer:", prediction_variability_regularizer)

    with tf.variable_scope(constants.WEIGHTED_CROSS_ENTROPY_LOSS_V2):
        # Border weights
        weight_border = get_border_weights(labels, border_amplitude, border_half_width)

        # Modified focal weights with class weights
        probabilities = tf.nn.softmax(logits)
        labels_onehot = tf.cast(tf.one_hot(labels, 2), dtype=tf.float32)
        proba_correct_class = tf.reduce_sum(
            probabilities * labels_onehot, axis=2
        )  # output shape [batch, time]
        focal_term = (1.0 - proba_correct_class) ** focal_gamma
        weight_focal_raw = 1.0 + (focal_weight - 1.0) * focal_term
        weight_label = get_weights(logits, labels, class_weights)
        weight_focal = weight_focal_raw * weight_label

        # Combine weights
        if mix_weights_strategy == constants.MIX_WEIGHTS_MAX:
            weights = tf.maximum(weight_border, weight_focal)
        elif mix_weights_strategy == constants.MIX_WEIGHTS_SUM:
            weights = weight_border + weight_focal - 1.0
        elif mix_weights_strategy == constants.MIX_WEIGHTS_PRODUCT:
            weights = weight_border * weight_focal
        else:
            raise ValueError('mix_strategy "%s" invalid' % mix_weights_strategy)

        # Weighted loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        )
        loss = weights * loss
        # First, we compute the weighted average for each segment independently
        loss = tf.reduce_sum(loss, axis=1) / tf.reduce_sum(weights, axis=1)
        # Now we average the segments
        loss = tf.reduce_mean(loss)

        # Add regularization of variability in predictions
        probabilities_1 = probabilities[:, :, 1]
        probabilities_1_right = probabilities_1[:, 1:]
        probabilities_1_left = probabilities_1[:, :-1]
        proba_difference = (probabilities_1_right - probabilities_1_left) ** 2
        penalization_term = tf.reduce_mean(proba_difference)
        tf.summary.scalar("variability", penalization_term)
        tf.summary.scalar("unregularized_loss", loss)

        loss = loss + (prediction_variability_regularizer * penalization_term)

        # Summaries
        loss_summ = tf.summary.scalar("loss", loss)

    return loss, loss_summ


def weighted_cross_entropy_loss_v3(
    logits,
    labels,
    border_amplitude,
    border_half_width,
    focal_gamma,
    focal_weight,
    class_weights,
    mix_weights_strategy,
    prediction_variability_regularizer,
    prediction_variability_lag,
):
    print("Using Weighted Cross Entropy Loss V3 (max at 0.5, L2)")
    print("By-segment mean loss")
    print("Class weights:", class_weights)
    print("Borders A: %1.2f HW: %1.2f" % (border_amplitude, border_half_width))
    print("Mod focal G: %1.2f W0: %1.2f" % (focal_gamma, focal_weight))
    print("Mix weights strategy:", mix_weights_strategy)
    print("Pred. variability regularizer:", prediction_variability_regularizer)
    print("Pred. variability lag:", prediction_variability_lag)

    with tf.variable_scope(constants.WEIGHTED_CROSS_ENTROPY_LOSS_V3):
        # Border weights
        weight_border = get_border_weights(labels, border_amplitude, border_half_width)

        # Modified focal weights with class weights
        probabilities = tf.nn.softmax(logits)
        labels_onehot = tf.cast(tf.one_hot(labels, 2), dtype=tf.float32)
        proba_correct_class = tf.reduce_sum(
            probabilities * labels_onehot, axis=2
        )  # output shape [batch, time]
        focal_term = (1.0 - proba_correct_class) ** focal_gamma
        weight_focal_raw = 1.0 + (focal_weight - 1.0) * focal_term
        weight_label = get_weights(logits, labels, class_weights)
        weight_focal = weight_focal_raw * weight_label

        # Combine weights
        if mix_weights_strategy == constants.MIX_WEIGHTS_MAX:
            weights = tf.maximum(weight_border, weight_focal)
        elif mix_weights_strategy == constants.MIX_WEIGHTS_SUM:
            weights = weight_border + weight_focal - 1.0
        elif mix_weights_strategy == constants.MIX_WEIGHTS_PRODUCT:
            weights = weight_border * weight_focal
        else:
            raise ValueError('mix_strategy "%s" invalid' % mix_weights_strategy)

        # Weighted loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        )
        loss = weights * loss
        # First, we compute the weighted average for each segment independently
        loss = tf.reduce_sum(loss, axis=1) / tf.reduce_sum(weights, axis=1)
        # Now we average the segments
        loss = tf.reduce_mean(loss)

        # Add regularization of variability in predictions
        lag = prediction_variability_lag
        probabilities_1 = probabilities[:, :, 1]
        probabilities_1_right = probabilities_1[:, lag:]
        probabilities_1_left = probabilities_1[:, :-lag]
        proba_diff = tf.abs(probabilities_1_right - probabilities_1_left)
        proba_diff_cost = 1 - (2 * proba_diff - 1) ** 2
        # proba_diff_cost = 1.0 - 4.0 * ((proba_diff - 0.5) ** 2)

        penalization_term = tf.reduce_mean(proba_diff_cost)
        tf.summary.scalar("variability", penalization_term)
        tf.summary.scalar("unregularized_loss", loss)

        loss = loss + (prediction_variability_regularizer * penalization_term)

        # Summaries
        loss_summ = tf.summary.scalar("loss", loss)

    return loss, loss_summ


def weighted_cross_entropy_loss_v4(
    logits,
    labels,
    border_amplitude,
    border_half_width,
    focal_gamma,
    focal_weight,
    class_weights,
    mix_weights_strategy,
    prediction_variability_regularizer,
    prediction_variability_lag,
):
    print("Using Weighted Cross Entropy Loss V4 (max at 0.5, L1)")
    print("By-segment mean loss")
    print("Class weights:", class_weights)
    print("Borders A: %1.2f HW: %1.2f" % (border_amplitude, border_half_width))
    print("Mod focal G: %1.2f W0: %1.2f" % (focal_gamma, focal_weight))
    print("Mix weights strategy:", mix_weights_strategy)
    print("Pred. variability regularizer:", prediction_variability_regularizer)
    print("Pred. variability lag:", prediction_variability_lag)

    with tf.variable_scope(constants.WEIGHTED_CROSS_ENTROPY_LOSS_V4):
        # Border weights
        weight_border = get_border_weights(labels, border_amplitude, border_half_width)

        # Modified focal weights with class weights
        probabilities = tf.nn.softmax(logits)
        labels_onehot = tf.cast(tf.one_hot(labels, 2), dtype=tf.float32)
        proba_correct_class = tf.reduce_sum(
            probabilities * labels_onehot, axis=2
        )  # output shape [batch, time]
        focal_term = (1.0 - proba_correct_class) ** focal_gamma
        weight_focal_raw = 1.0 + (focal_weight - 1.0) * focal_term
        weight_label = get_weights(logits, labels, class_weights)
        weight_focal = weight_focal_raw * weight_label

        # Combine weights
        if mix_weights_strategy == constants.MIX_WEIGHTS_MAX:
            weights = tf.maximum(weight_border, weight_focal)
        elif mix_weights_strategy == constants.MIX_WEIGHTS_SUM:
            weights = weight_border + weight_focal - 1.0
        elif mix_weights_strategy == constants.MIX_WEIGHTS_PRODUCT:
            weights = weight_border * weight_focal
        else:
            raise ValueError('mix_strategy "%s" invalid' % mix_weights_strategy)

        # Weighted loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        )
        loss = weights * loss
        # First, we compute the weighted average for each segment independently
        loss = tf.reduce_sum(loss, axis=1) / tf.reduce_sum(weights, axis=1)
        # Now we average the segments
        loss = tf.reduce_mean(loss)

        # Add regularization of variability in predictions
        lag = prediction_variability_lag
        probabilities_1 = probabilities[:, :, 1]
        probabilities_1_right = probabilities_1[:, lag:]
        probabilities_1_left = probabilities_1[:, :-lag]
        proba_diff = tf.abs(probabilities_1_right - probabilities_1_left)
        proba_diff_cost_a = 2 * proba_diff
        proba_diff_cost_b = 2 * (1 - proba_diff)
        proba_diff_cost = tf.minimum(proba_diff_cost_a, proba_diff_cost_b)

        penalization_term = tf.reduce_mean(proba_diff_cost)
        tf.summary.scalar("variability", penalization_term)
        tf.summary.scalar("unregularized_loss", loss)

        loss = loss + (prediction_variability_regularizer * penalization_term)

        # Summaries
        loss_summ = tf.summary.scalar("loss", loss)

    return loss, loss_summ


def cross_entropy_loss_borders_fn(logits, labels, amplitude, half_width):
    """Returns the cross-entropy loss to be minimized.

    It computes border weights with a gaussian window of maximum amplitude
    edge_weight and half size of half_width, which is 3*sigma.

    Args:
        logits: (3d tensor) logits tensor of shape [batch, timelen, 2]
        labels: (2d tensor) binary tensor of shape [batch, timelen]
        amplitude: (float) Maximum amplitude of gaussian.
        half_width: (float) Number of samples on one half of the window, which
            is used to compute sigma as half_width = 3 * sigma.
    """
    print("Using Cross Entropy Loss BORDERS")
    with tf.variable_scope(constants.CROSS_ENTROPY_BORDERS_LOSS):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        )
        # Weighted loss
        weights = get_border_weights(labels, amplitude, half_width)
        loss = tf.reduce_sum(weights * loss) / tf.reduce_sum(weights)
        # Summaries
        loss_summ = tf.summary.scalar("loss", loss)
    return loss, loss_summ


def cross_entropy_loss_borders_ind_fn(logits, labels, amplitude, half_width):
    """Returns the cross-entropy loss to be minimized.

    It computes border weights with a gaussian window of maximum amplitude
    edge_weight and half size of half_width, which is 3*sigma.

    Individual version

    Args:
        logits: (3d tensor) logits tensor of shape [batch, timelen, 2]
        labels: (2d tensor) binary tensor of shape [batch, timelen]
        amplitude: (float) Maximum amplitude of gaussian.
        half_width: (float) Number of samples on one half of the window, which
            is used to compute sigma as half_width = 3 * sigma.
    """
    print("Using Cross Entropy Loss BORDERS INDIVIDUAL")
    with tf.variable_scope(constants.CROSS_ENTROPY_BORDERS_IND_LOSS):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        )
        # Weighted loss
        weights = get_border_weights(labels, amplitude, half_width)
        loss = weights * loss
        # First, we compute the weighted average for each segment independently
        loss = tf.reduce_sum(loss, axis=1) / tf.reduce_sum(weights, axis=1)
        # Now we average the segments
        loss = tf.reduce_mean(loss)
        # Summaries
        loss_summ = tf.summary.scalar("loss", loss)
    return loss, loss_summ


def cross_entropy_loss_fn(logits, labels, class_weights):
    """Returns the cross-entropy loss to be minimized.

    Args:
        logits: (3d tensor) logits tensor of shape [batch, timelen, 2]
        labels: (2d tensor) binary tensor of shape [batch, timelen]
        class_weights: ({None, BALANCED, array_like}) Determines the class
            weights to be applied when computing the loss. If None, no weights
            are applied. If BALANCED, the weights balance the class
            frequencies. If is an array of shape [2,], class_weights[i]
            is the weight applied to class i. If BALANCED_DROP, then
            outputs related to the negative class are randomly dropped
            so that their number approx equals that of the positive class.
    """
    print("Using Cross Entropy Loss")
    with tf.variable_scope(constants.CROSS_ENTROPY_LOSS):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        )
        # Weighted loss
        weights = get_weights(logits, labels, class_weights)
        loss = tf.reduce_sum(weights * loss) / tf.reduce_sum(weights)
        # Summaries
        loss_summ = tf.summary.scalar("loss", loss)
    return loss, loss_summ


def cross_entropy_loss_with_self_supervision_fn(
    logits, labels, class_weights, self_supervised_loss, self_supervised_coefficient
):
    """Returns the cross-entropy loss to be minimized.

    Args:
        logits: (3d tensor) logits tensor of shape [batch, timelen, 2]
        labels: (2d tensor) binary tensor of shape [batch, timelen]
        class_weights: ({None, BALANCED, array_like}) Determines the class
            weights to be applied when computing the loss. If None, no weights
            are applied. If BALANCED, the weights balance the class
            frequencies. If is an array of shape [2,], class_weights[i]
            is the weight applied to class i. If BALANCED_DROP, then
            outputs related to the negative class are randomly dropped
            so that their number approx equals that of the positive class.
    """
    print("Using Cross Entropy Loss with Self-supervision")
    with tf.variable_scope(constants.CROSS_ENTROPY_LOSS_WITH_SELF_SUPERVISION):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        )
        # Weighted loss
        weights = get_weights(logits, labels, class_weights)
        main_loss = tf.reduce_sum(weights * loss) / tf.reduce_sum(weights)

        print("Using self supervision with coefficient", self_supervised_coefficient)
        loss = main_loss + self_supervised_coefficient * self_supervised_loss

        # Summaries
        tf.summary.scalar("xentropy_loss", main_loss)
        tf.summary.scalar("self_supervised_loss", self_supervised_loss)

        loss_summ = tf.summary.scalar("loss", loss)
    return loss, loss_summ


def cross_entropy_loss_with_logits_reg_fn(
    logits, labels, class_weights, regularization_type, regularization_weight
):
    print("Using Cross Entropy Loss with Logits Regularization")
    print("Using regularization type %s" % regularization_type)
    print("Using regularization weight %s" % regularization_weight)
    with tf.variable_scope(constants.CROSS_ENTROPY_LOSS):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        )
        # Weighted loss
        weights = get_weights(logits, labels, class_weights)
        loss = tf.reduce_sum(weights * loss) / tf.reduce_sum(weights)

        # Regularization
        print("logits", logits)
        if regularization_type == constants.LOGITS_REG_NORM:
            norm = tf.reduce_sum(logits**2, axis=-1)
            reg_term = norm / 4.0
        elif regularization_type == constants.LOGITS_REG_NORM_SQRT:
            norm = tf.reduce_sum(logits**2, axis=-1)
            reg_term = tf.math.sqrt(norm)
        elif regularization_type == constants.LOGITS_REG_ATTRACTOR:
            p1 = np.array([-1.0, 1.0])
            p2 = np.array([1.0, -1.0])
            norm_1 = tf.reduce_sum((logits - p1) ** 2, axis=-1)
            norm_2 = tf.reduce_sum((logits - p2) ** 2, axis=-1)
            attractor = tf.math.sqrt(norm_1 * norm_2)
            reg_term = attractor / 4.0
        elif regularization_type == constants.LOGITS_REG_ATTRACTOR_SQRT:
            p1 = np.array([-1.0, 1.0])
            p2 = np.array([1.0, -1.0])
            norm_1 = tf.reduce_sum((logits - p1) ** 2, axis=-1)
            norm_2 = tf.reduce_sum((logits - p2) ** 2, axis=-1)
            attractor = tf.math.sqrt(norm_1 * norm_2)
            reg_term = tf.math.sqrt(attractor)
        else:
            raise ValueError(
                "regularization type %s not supported" % regularization_type
            )
        print("reg_term", reg_term)
        reg_term = tf.reduce_mean(reg_term)
        print("reg_term", reg_term)

        tf.summary.scalar("logits_reg", reg_term)
        tf.summary.scalar("unregularized_loss", loss)

        loss = loss + regularization_weight * reg_term

        # Summaries
        loss_summ = tf.summary.scalar("loss", loss)

    return loss, loss_summ


def hinge_loss_fn(logits, labels, class_weights):
    """Returns the hinge loss to be minimized.

    Args:
        logits: (3d tensor) logits tensor of shape [batch, timelen, 2]
        labels: (2d tensor) binary tensor of shape [batch, timelen]
        class_weights: ({None, BALANCED, array_like}) Determines the class
            weights to be applied when computing the loss. If None, no weights
            are applied. If BALANCED, the weights balance the class
            frequencies. If is an array of shape [2,], class_weights[i]
            is the weight applied to class i. If BALANCED_DROP, then
            outputs related to the negative class are randomly dropped
            so that their number approx equals that of the positive class.
    """
    print("Using Hinge Loss")
    with tf.variable_scope(constants.HINGE_LOSS):
        # A softmax with 2 logits is equivalent to a sigmoid with logit l1-l0
        # We satisfy that the boundary of the classes is l=0 (p=0.5).
        logits_sigmoid = logits[:, :, 1] - logits[:, :, 0]

        loss = tf.losses.hinge_loss(
            labels=labels, logits=logits_sigmoid, reduction="none"
        )

        # Weighted loss
        weights = get_weights(logits, labels, class_weights)
        loss = tf.reduce_sum(weights * loss) / tf.reduce_sum(weights)
        # Summaries
        loss_summ = tf.summary.scalar("loss", loss)
    return loss, loss_summ


def dice_loss_fn(probabilities, labels):
    """Returns 1-DICE loss to be minimized.

    Args:
        probabilities: (2d tensor) tensor of the probabilities of class 1
            with shape [batch, timelen]
        labels: (2d tensor) binary tensor of shape [batch, timelen]
    """
    print("Using DICE Loss")
    with tf.variable_scope(constants.DICE_LOSS):
        labels = tf.to_float(labels)
        intersection = tf.reduce_sum(tf.multiply(probabilities, labels))
        size_prob = tf.reduce_sum(probabilities)
        size_labels = tf.reduce_sum(labels)
        dice = 2 * intersection / (size_prob + size_labels)
        loss = 1 - dice
        # Summaries
        loss_summ = tf.summary.scalar("loss", loss)
    return loss, loss_summ


def mod_focal_loss_fn(logits, labels, class_weights, gamma, mis_weight):
    """Returns the MODIFIED focal loss to be minimized.

    "Focal Loss for Dense Object Detection"
    https://arxiv.org/abs/1708.02002

    If p is probability of correct class, the original Focal loss weights the
    cross-entropy loss with:
    w = (1-p) ^ gamma

    The modified version weights the cross-entropy loss with:
    w = 1 + (mis_weight - 1) * (1-p) ^ gamma

    If mis_weight = 1 or gamma = 0, this reduces to w = 1 for all samples
    (regular cross-entropy)

    Args:
        logits: (3d tensor) logits tensor of shape [batch, timelen, 2]
        labels: (2d tensor) binary tensor of shape [batch, timelen]
        class_weights: ({None, BALANCED, array_like}) Determines the class
            weights to be applied when computing the loss. If None, no weights
            are applied. If BALANCED, the weights balance the class
            frequencies. If is an array of shape [2,], class_weights[i]
            is the weight applied to class i. If BALANCED_DROP, then
            outputs related to the negative class are randomly dropped
            so that their number approx equals that of the positive class.
        gamma: Focusing parameter (Non-negative)
        mis_weight: Weight for misclassification (greater or equal to 1).
    """
    print(
        "Using MODIFIED Focal Loss INDIVIDUAL (gamma = %1.4f, mis_weight = %1.4f)"
        % (gamma, mis_weight)
    )
    with tf.variable_scope(constants.MOD_FOCAL_LOSS):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        )
        # Apply focusing parameter
        probabilities = tf.nn.softmax(logits)
        labels_onehot = tf.cast(tf.one_hot(labels, 2), dtype=tf.float32)
        proba_correct_class = tf.reduce_sum(
            probabilities * labels_onehot, axis=2
        )  # output shape [batch, time]
        focal_term = (1.0 - proba_correct_class) ** gamma

        # Compute weights
        weight_focal = 1.0 + (mis_weight - 1.0) * focal_term
        weight_c = get_weights(logits, labels, class_weights)
        weights = weight_focal * weight_c

        # Apply weights
        loss = weights * loss  # weighted loss

        # First, we compute the weighted average for each segment independently
        loss = tf.reduce_sum(loss, axis=1) / tf.reduce_sum(weights, axis=1)

        # Now we average the segments
        loss = tf.reduce_mean(loss)

        # Summaries
        loss_summ = tf.summary.scalar("loss", loss)
    return loss, loss_summ


def focal_loss_fn(logits, labels, class_weights, gamma):
    """Returns the focal loss to be minimized.

    "Focal Loss for Dense Object Detection"
    https://arxiv.org/abs/1708.02002

    Args:
        logits: (3d tensor) logits tensor of shape [batch, timelen, 2]
        labels: (2d tensor) binary tensor of shape [batch, timelen]
        class_weights: ({None, BALANCED, array_like}) Determines the class
            weights to be applied when computing the loss. If None, no weights
            are applied. If BALANCED, the weights balance the class
            frequencies. If is an array of shape [2,], class_weights[i]
            is the weight applied to class i. If BALANCED_DROP, then
            outputs related to the negative class are randomly dropped
            so that their number approx equals that of the positive class.
        gamma: Focusing parameter (Non-negative)
    """
    print("Using Focal Loss (gamma = %1.4f)" % gamma)
    with tf.variable_scope(constants.FOCAL_LOSS):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        )
        # Apply focusing parameter
        probabilities = tf.nn.softmax(logits)
        labels_onehot = tf.cast(tf.one_hot(labels, 2), dtype=tf.float32)
        proba_correct_class = tf.reduce_sum(
            probabilities * labels_onehot, axis=2
        )  # output shape [batch, time]
        loss = ((1.0 - proba_correct_class) ** gamma) * loss
        # Weighted loss
        weights = get_weights(logits, labels, class_weights)
        loss = tf.reduce_sum(weights * loss) / tf.reduce_sum(weights)
        # Summaries
        loss_summ = tf.summary.scalar("loss", loss)
    return loss, loss_summ


def cross_entropy_negentropy_loss_fn(logits, labels, class_weights, beta):
    print("Using Cross Entropy Loss with Entropy Regularization (beta = %1.4f)" % beta)
    with tf.variable_scope(constants.CROSS_ENTROPY_NEG_ENTROPY_LOSS):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        )
        # Regularization
        probabilities = tf.nn.softmax(logits)
        neg_entropy = tf.reduce_sum(
            probabilities * tf.log(probabilities + 1e-6), axis=2
        )  # output shape [batch, time]
        loss = loss + beta * neg_entropy
        # Weighted loss
        weights = get_weights(logits, labels, class_weights)
        loss = tf.reduce_sum(weights * loss) / tf.reduce_sum(weights)
        # Summaries
        loss_summ = tf.summary.scalar("loss", loss)
    return loss, loss_summ


def cross_entropy_smoothing_loss_fn(logits, labels, class_weights, epsilon):
    print("Using Cross Entropy Loss with Label Smoothing (eps = %1.4f)" % epsilon)
    with tf.variable_scope(constants.CROSS_ENTROPY_SMOOTHING_LOSS):
        labels_onehot = tf.cast(tf.one_hot(labels, 2), dtype=tf.float32)
        smooth_labels = labels_onehot * (1.0 - 2.0 * epsilon) + epsilon
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=smooth_labels, logits=logits
        )
        offset = -tf.reduce_sum(smooth_labels * tf.log(smooth_labels + 1e-6), axis=2)
        loss = loss - offset
        # Weighted loss
        weights = get_weights(logits, labels, class_weights)
        loss = tf.reduce_sum(weights * loss) / tf.reduce_sum(weights)
        # Summaries
        loss_summ = tf.summary.scalar("loss", loss)
    return loss, loss_summ


def cross_entropy_hard_clip_loss_fn(logits, labels, class_weights, epsilon):
    print("Using Cross Entropy Loss with Hard Clip (eps = %1.4f)" % epsilon)
    with tf.variable_scope(constants.CROSS_ENTROPY_HARD_CLIP_LOSS):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        )
        # Clip
        offset = -tf.log(1.0 - epsilon)
        loss = tf.nn.relu(loss - offset)
        # Weighted loss
        weights = get_weights(logits, labels, class_weights)
        loss = tf.reduce_sum(weights * loss) / tf.reduce_sum(weights)
        # Summaries
        loss_summ = tf.summary.scalar("loss", loss)
    return loss, loss_summ


def cross_entropy_smoothing_clip_loss_fn(logits, labels, class_weights, epsilon):
    print(
        "Using Cross Entropy Loss with Label Smoothing and Clip (eps = %1.4f)" % epsilon
    )
    with tf.variable_scope(constants.CROSS_ENTROPY_SMOOTHING_CLIP_LOSS):
        labels_onehot = tf.cast(tf.one_hot(labels, 2), dtype=tf.float32)
        smooth_labels = labels_onehot * (1.0 - 2.0 * epsilon) + epsilon
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=smooth_labels, logits=logits
        )
        offset = -tf.reduce_sum(smooth_labels * tf.log(smooth_labels + 1e-6), axis=2)
        loss = loss - offset
        # Clip
        probabilities = tf.nn.softmax(logits)
        distance_to_hard_label = tf.reduce_sum(
            (probabilities - labels_onehot) ** 2, axis=2
        )
        distance_thr = 2 * (epsilon**2)
        mask = tf.cast(
            tf.math.greater(distance_to_hard_label, distance_thr), tf.float32
        )
        loss = loss * mask
        # Weighted loss
        weights = get_weights(logits, labels, class_weights)
        loss = tf.reduce_sum(weights * loss) / tf.reduce_sum(weights)
        # Summaries
        loss_summ = tf.summary.scalar("loss", loss)
    return loss, loss_summ


def worst_mining_loss_fn(logits, labels, factor_negative, min_negative):
    """Returns the balanced cross-entropy loss to be minimized by worst
    negative mining

    Args:
        logits: (3d tensor) logits tensor of shape [batch, timelen, 2]
        labels: (2d tensor) binary tensor of shape [batch, timelen]
        factor_negative: (int) Ratio of negatives / positives
        min_negative: (int) minimum number of negatives examples that are
            kept when balancing.

    """
    print("Using Worst Mining Loss (individual version)")
    with tf.variable_scope(constants.WORST_MINING_LOSS):
        print("Min negative:", min_negative)
        print("Factor negative:", factor_negative)
        min_negative = tf.cast(min_negative, tf.float32)
        factor_negative = tf.cast(factor_negative, tf.float32)
        batch_size = 32
        print("-----------------------------------")
        print("-----------------------------------")
        print("AT LOSS, BATCH SIZE HARDCODED AS %d" % batch_size)
        print("-----------------------------------")
        print("-----------------------------------")

        # Unstack batch
        logits_list = tf.unstack(logits, axis=0, num=batch_size)
        labels_list = tf.unstack(labels, axis=0, num=batch_size)
        sum_loss_batch = 0
        for s_logits, s_labels in zip(logits_list, labels_list):
            # # First we flatten the vectors
            # logits = tf.reshape(logits, [-1, 2])  # [n_outputs, 2]
            # labels = tf.reshape(labels, [-1])  # [n_outputs]

            # Number of positives and negatives
            labels_float = tf.cast(s_labels, tf.float32)
            n_negative = tf.reduce_sum(1 - labels_float)
            n_positive = tf.reduce_sum(labels_float)

            # Compute number of negatives to keep
            feasible_min_negative = tf.minimum(min_negative, n_negative)
            n_negative_to_keep = tf.minimum(factor_negative * n_positive, n_negative)
            n_negative_to_keep = tf.maximum(n_negative_to_keep, feasible_min_negative)

            # Compute loss
            loss_raw = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=s_labels, logits=s_logits
            )
            loss_positive = loss_raw * labels_float
            loss_negative = loss_raw * (1 - labels_float)
            loss_negative_to_keep, _ = tf.nn.top_k(
                loss_negative, k=tf.cast(n_negative_to_keep, tf.int32)
            )
            sum_loss = tf.reduce_sum(loss_positive) + tf.reduce_sum(
                loss_negative_to_keep
            )
            n_examples = n_positive + n_negative_to_keep
            s_loss = sum_loss / n_examples

            # Add to batch sum
            sum_loss_batch = sum_loss_batch + s_loss

        # Loss is mean of single losses
        loss = sum_loss_batch / batch_size
        loss_summ = tf.summary.scalar("loss", loss)
    return loss, loss_summ


def worst_mining_v2_loss_fn(logits, labels, factor_negative, min_negative):
    """Returns the balanced cross-entropy loss to be minimized by worst
    negative mining. L+ and L- are averaged independently.

    Args:
        logits: (3d tensor) logits tensor of shape [batch, timelen, 2]
        labels: (2d tensor) binary tensor of shape [batch, timelen]
        factor_negative: (int) Ratio of negatives / positives
        min_negative: (int) minimum number of negatives examples that are
            kept when balancing.

    """
    print("Using Worst Mining Loss V2 (pos and neg parts averaged before sum)")
    with tf.variable_scope(constants.WORST_MINING_V2_LOSS):
        print("Min negative:", min_negative)
        print("Factor negative:", factor_negative)

        # First we flatten the vectors
        logits = tf.reshape(logits, [-1, 2])  # [n_outputs, 2]
        labels = tf.reshape(labels, [-1])  # [n_outputs]

        labels_float = tf.cast(labels, tf.float32)

        n_negative = tf.reduce_sum(1 - labels_float)
        n_positive = tf.reduce_sum(labels_float)

        n_negative_to_keep = tf.minimum(factor_negative * n_positive, n_negative)
        n_negative_to_keep = tf.maximum(n_negative_to_keep, min_negative)

        loss_raw = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        )

        loss_positive = loss_raw * labels_float

        loss_negative = loss_raw * (1 - labels_float)
        loss_negative_to_keep, _ = tf.nn.top_k(
            loss_negative, k=tf.cast(n_negative_to_keep, tf.int32)
        )

        loss_positive_norm = tf.reduce_sum(loss_positive) / (n_positive + 1e-8)
        loss_negative_norm = tf.reduce_sum(loss_negative_to_keep) / n_negative_to_keep

        loss = loss_positive_norm + loss_negative_norm

        loss_summ = tf.summary.scalar("loss", loss)
    return loss, loss_summ
