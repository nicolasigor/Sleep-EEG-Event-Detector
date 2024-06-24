"""Module that defines useful metrics to monitor a model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from sleeprnn.common import constants


def confusion_matrix(logits, labels, masks=None):
    """Returns TP, FP and FN"""
    with tf.variable_scope("confusion_matrix"):
        predictions_sparse = tf.argmax(logits, axis=-1)
        labels_zero = tf.equal(labels, tf.zeros_like(labels))
        labels_one = tf.equal(labels, tf.ones_like(labels))
        predictions_zero = tf.equal(
            predictions_sparse, tf.zeros_like(predictions_sparse)
        )
        predictions_one = tf.equal(predictions_sparse, tf.ones_like(predictions_sparse))

        valid_samples = 1.0 if masks is None else tf.cast(masks, tf.float32)

        tp_samples = tf.cast(tf.logical_and(labels_one, predictions_one), tf.float32)
        tp = tf.reduce_sum(valid_samples * tp_samples)

        fp_samples = tf.cast(tf.logical_and(labels_zero, predictions_one), tf.float32)
        fp = tf.reduce_sum(valid_samples * fp_samples)

        fn_samples = tf.cast(tf.logical_and(labels_one, predictions_zero), tf.float32)
        fn = tf.reduce_sum(valid_samples * fn_samples)

    return tp, fp, fn


def precision_recall_f1score(tp, fp, fn):
    """Return Precision, Recall, and F1-Score metrics."""
    with tf.variable_scope("precision"):
        # Edge case: no detections -> precision 1
        precision = tf.cond(
            pred=tf.equal((tp + fp), 0),
            true_fn=lambda: 1.0,
            false_fn=lambda: tp / (tp + fp),
        )
    with tf.variable_scope("recall"):
        # Edge case: no marks -> recall 1
        recall = tf.cond(
            pred=tf.equal((tp + fn), 0),
            true_fn=lambda: 1.0,
            false_fn=lambda: tp / (tp + fn),
        )
    with tf.variable_scope("f1_score"):
        # Edge case: precision and recall 0 -> f1 score 0
        f1_score = tf.cond(
            pred=tf.equal((2 * tp + fn + fp), 0),
            true_fn=lambda: 0.0,
            false_fn=lambda: 2 * tp / (2 * tp + fn + fp),
        )
    return precision, recall, f1_score
