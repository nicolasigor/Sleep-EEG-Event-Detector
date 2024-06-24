"""Module that defines optimizers to train a model."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from sleeprnn.common import constants
from sleeprnn.nn import adam_w


def generic_optimizer_fn(optimizer, loss, clip_norm):
    """Applies the optimizer to the loss."""

    if type(optimizer) == adam_w.AdamW:
        train_vars = tf.trainable_variables()
        grads = optimizer.get_gradients(loss, train_vars)
        original_gvs = [(grad, var) for grad, var in zip(grads, train_vars)]
    else:
        original_gvs = optimizer.compute_gradients(loss)

    gradients = [gv[0] for gv in original_gvs]
    grad_norm = tf.global_norm(gradients, name='gradient_norm')
    grad_norm_summ = tf.summary.scalar('original_grad_norm', grad_norm)

    if clip_norm is not None:
        gradients, _ = tf.clip_by_global_norm(
            gradients,
            clip_norm,
            use_norm=grad_norm,
            name='clipping')
        clipped_grad_norm = tf.global_norm(gradients, name='new_gradient_norm')
        variables = [gv[1] for gv in original_gvs]
        new_gvs = [(grad, var) for grad, var in zip(gradients, variables)]
        clipped_grad_norm_summ = tf.summary.scalar(
            'clipped_grad_norm', clipped_grad_norm)
        grad_norm_summ = tf.summary.merge(
            [grad_norm_summ, clipped_grad_norm_summ])
    else:
        new_gvs = original_gvs

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # For BN
    with tf.control_dependencies(update_ops):
        train_step = optimizer.apply_gradients(new_gvs)
    reset_optimizer_op = tf.variables_initializer(optimizer.variables())
    return train_step, reset_optimizer_op, grad_norm_summ


def adam_optimizer_fn(
        loss, learning_rate, clip_norm):
    """Returns the optimizer operation to minimize the loss with Adam.

    Args:
        loss: (tensor) loss to be minimized
        learning_rate: (float) learning rate for the optimizer
        clip_norm: (float) Global norm to clip.
    """
    with tf.name_scope(constants.ADAM_OPTIMIZER):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    return generic_optimizer_fn(optimizer, loss, clip_norm)


def adam_w_optimizer_fn(
        loss, learning_rate, weight_decay, clip_norm):
    """Returns the optimizer operation to minimize the loss with Adam W.

    Args:
        loss: (tensor) loss to be minimized
        learning_rate: (float) learning rate for the optimizer
        weight_decay: (float) Weight decay for the optimizer.
        clip_norm: (float) Global norm to clip.
    """
    with tf.name_scope(constants.ADAM_W_OPTIMIZER):
        optimizer = adam_w.AdamW(weight_decay, learning_rate=learning_rate)
    return generic_optimizer_fn(optimizer, loss, clip_norm)


def sgd_optimizer_fn(
        loss, learning_rate, momentum, clip_norm, use_nesterov):
    """Returns the optimizer operation to minimize the loss with SGD with
    momentum.

    Args:
        loss: (tensor) loss to be minimized
        learning_rate: (float) learning rate for the optimizer
        momentum: (Optional, float) momentum for the optimizer.
        clip_norm: (float) Global norm to clip.
        use_nesterov: (bool) whether to use
            Nesterov momentum instead of regular momentum.
    """
    with tf.name_scope(constants.SGD_OPTIMIZER):
        optimizer = tf.train.MomentumOptimizer(
            learning_rate, momentum, use_nesterov=use_nesterov)
    return generic_optimizer_fn(optimizer, loss, clip_norm)


def rmsprop_optimizer_fn(
        loss, learning_rate, momentum, clip_norm):
    """Returns the optimizer operation to minimize the loss with RMSProp

    Args:
        loss: (tensor) loss to be minimized
        learning_rate: (float) learning rate for the optimizer
        momentum: (Optional, float) momentum for the optimizer.
        clip_norm: (float) Global norm to clip.
    """
    with tf.name_scope(constants.RMSPROP_OPTIMIZER):
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate, momentum=momentum)
    return generic_optimizer_fn(optimizer, loss, clip_norm)
