"""Module that defines input pipeline operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def get_iterator_splitted(
        tensors_ph_1,
        tensors_ph_2,
        batch_size,
        repeat=True,
        shuffle_buffer_size=0,
        map_fn=None,
        prefetch_buffer_size=0,
        name=None):
    with tf.name_scope(name):
        with tf.device('/cpu:0'):

            batch_size_1 = int(batch_size / 2)
            batch_size_2 = batch_size - batch_size_1

            # First dataset
            dataset_1 = tf.data.Dataset.from_tensor_slices(tensors_ph_1)
            if shuffle_buffer_size > 0:
                dataset_1 = dataset_1.shuffle(buffer_size=shuffle_buffer_size)
            if repeat:
                dataset_1 = dataset_1.repeat()
            if map_fn is not None:
                dataset_1 = dataset_1.map(map_fn)
            dataset_1 = dataset_1.batch(batch_size=batch_size_1)
            if prefetch_buffer_size > 0:
                dataset_1 = dataset_1.prefetch(buffer_size=prefetch_buffer_size)

            # Second dataset
            dataset_2 = tf.data.Dataset.from_tensor_slices(tensors_ph_2)
            if shuffle_buffer_size > 0:
                dataset_2 = dataset_2.shuffle(buffer_size=shuffle_buffer_size)
            if repeat:
                dataset_2 = dataset_2.repeat()
            if map_fn is not None:
                dataset_2 = dataset_2.map(map_fn)
            dataset_2 = dataset_2.batch(batch_size=batch_size_2)
            if prefetch_buffer_size > 0:
                dataset_2 = dataset_2.prefetch(buffer_size=prefetch_buffer_size)

            # Zip datasets
            dataset = tf.data.Dataset.zip((dataset_1, dataset_2))
            dataset = dataset.map(_combine_batch_fn)
            if prefetch_buffer_size > 0:
                dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

            iterator = dataset.make_initializable_iterator()
    return iterator


def _combine_batch_fn(tensors_1, tensors_2):
    """Takes a tuple of tensors from two sources and concatenates them
    along the batch dimension to form a single tuple."""
    n_tensors = len(tensors_1)

    combined_tensors = []
    for k in range(n_tensors):
        tensor_from_1 = tensors_1[k]
        tensor_from_2 = tensors_2[k]
        this_combined = tf.concat([tensor_from_1, tensor_from_2], axis=0)
        combined_tensors.append(this_combined)
    combined_tensors = tuple(combined_tensors)
    return combined_tensors


def get_iterator(
        tensors_ph,
        batch_size,
        repeat=True,
        shuffle_buffer_size=0,
        map_fn=None,
        prefetch_buffer_size=0,
        name=None):
    """Builds efficient iterators for the training loop.

    Args:
        tensors_ph: (tensor) Input tensors placeholders
        batch_size: (int) Size of the minibatches
        repeat: (optional, boolean, defaults to True) whether to repeat
            ad infinitum the dataset or not.
        shuffle_buffer_size: (Optional, int, defaults to 0) Size of the buffer
            to shuffle the data. If 0, no shuffle is applied.
        map_fn: (Optional, function, defaults to None) A function that
            preprocess the features and labels before passing them to the model.
        prefetch_buffer_size: (Optional, int, defaults to 0) Size of the buffer
            to prefetch the batches. If 0, no prefetch is applied.
        name: (Optional, string, defaults to None) Name for the operation.
    """
    with tf.name_scope(name):
        with tf.device('/cpu:0'):
            dataset = tf.data.Dataset.from_tensor_slices(tensors_ph)
            if shuffle_buffer_size > 0:
                dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
            if repeat:
                dataset = dataset.repeat()
            if map_fn is not None:
                dataset = dataset.map(map_fn)
            dataset = dataset.batch(batch_size=batch_size)
            if prefetch_buffer_size > 0:
                dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
            iterator = dataset.make_initializable_iterator()
    return iterator


def get_global_iterator(handle_ph, iterators_list, name=None):
    """Builds a global iterator that can switch between two iterators.

    Args:
        handle_ph: (Tensor) Placeholder of type tf.string and shape [] that
            will be fed with the proper string_handle.
        iterators_list: (list of Iterator) List of the iterators from where we
            can obtain inputs.
        name: (Optional, string, defaults to None) Name for the operation.

    Returns:
        global_iterator: (Iterator) Iterator that will switch between iterator_1
            and iterator_2 according to the handle fed to handle_ph.
    """
    with tf.name_scope(name):
        with tf.device('/cpu:0'):
            global_iterator = tf.data.Iterator.from_string_handle(
                handle_ph, iterators_list[0].output_types,
                iterators_list[0].output_shapes)
    return global_iterator
