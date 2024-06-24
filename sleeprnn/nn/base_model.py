"""Module that defines as base model class to manage neural networks."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import numpy as np
import tensorflow as tf

from sleeprnn.common import pkeys
from sleeprnn.common import constants
from sleeprnn.detection.feeder_dataset import FeederDataset
from sleeprnn.detection.predicted_dataset import PredictedDataset
from sleeprnn.data.utils import pages2seq, extract_pages_from_centers, extract_pages
from sleeprnn.nn import feeding

PATH_THIS_DIR = os.path.dirname(__file__)
PATH_TO_PROJECT = os.path.abspath(os.path.join(PATH_THIS_DIR, '..'))

# Summaries keys
KEY_LOSS = 'loss'
KEY_GRAD_NORM = 'grad_norm'
KEY_BATCH_METRICS = 'batch_metrics'
KEY_EVAL_METRICS = 'eval_metrics'


class BaseModel(object):
    """ Base Model class to train and evaluate neural networks models.
    """
    def __init__(
            self,
            feat_train_shape,
            label_train_shape,
            feat_eval_shape,
            label_eval_shape,
            params,
            logdir='logs'
    ):
        """ Constructor.

        Args:
            feat_train_shape: (iterable) Shape of the features of a single
                example that is the input to the training iterator.
            label_train_shape: (iterable) Shape of the labels of a single
                example that is the input to the training iterator.
            feat_eval_shape: (iterable) Shape of the features of a single
                example that is the input to the evaluation iterator.
            label_eval_shape: (iterable) Shape of the labels of a single
                example that is the input to the evaluation iterator.
            params: (dict) Dictionary of parameters to configure the model.
                See common.model_keys for more details.
            logdir: (optional, string, defaults to 'logs') Directory of the
                model. This path can be absolute, or relative to project root.
        """
        # Clean computational graph
        tf.reset_default_graph()

        # Save attributes
        self.feat_train_shape = list(feat_train_shape)
        self.label_train_shape = list(label_train_shape)
        self.feat_eval_shape = list(feat_eval_shape)
        self.label_eval_shape = list(label_eval_shape)
        if os.path.isabs(logdir):
            self.logdir = logdir
        else:
            self.logdir = os.path.join(PATH_TO_PROJECT, logdir)
        self.params = pkeys.default_params.copy()
        if params is not None:
            self.params.update(params)  # Overwrite defaults

        # Create directory of logs
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        self.ckptdir = os.path.join(self.logdir, 'model', 'ckpt')

        # --- Build model

        # Input placeholders
        with tf.variable_scope('inputs_ph'):
            self.handle_ph = tf.placeholder(
                tf.string, shape=[], name='handle_ph')

            self.feats_train_1_ph = tf.placeholder(
                tf.float32, shape=[None] + self.feat_train_shape,
                name='feats_train_1_ph')
            self.labels_train_1_ph = tf.placeholder(
                tf.int8, shape=[None] + self.label_train_shape,
                name='labels_train_1_ph')
            self.masks_train_1_ph = tf.placeholder(
                tf.int8, shape=[None] + self.label_train_shape,
                name='masks_train_1_ph')

            self.feats_train_2_ph = tf.placeholder(
                tf.float32, shape=[None] + self.feat_train_shape,
                name='feats_train_2_ph')
            self.labels_train_2_ph = tf.placeholder(
                tf.int8, shape=[None] + self.label_train_shape,
                name='labels_train_2_ph')
            self.masks_train_2_ph = tf.placeholder(
                tf.int8, shape=[None] + self.label_train_shape,
                name='masks_train_2_ph')

            self.feats_eval_ph = tf.placeholder(
                tf.float32, shape=[None] + self.feat_eval_shape,
                name='feats_eval_ph')
            self.labels_eval_ph = tf.placeholder(
                tf.int8, shape=[None] + self.label_eval_shape,
                name='labels_eval_ph')
            self.masks_eval_ph = tf.placeholder(
                tf.int8, shape=[None] + self.label_eval_shape,
                name='masks_eval_ph')

            self.training_ph = tf.placeholder(tf.bool, name="training_ph")

        # Learning rate variable
        init_lr = self.params[pkeys.LEARNING_RATE]
        with tf.variable_scope('learning_rate'):
            self.learning_rate = tf.Variable(init_lr, trainable=False, name='lr')
            self.lr_summ = tf.summary.scalar('lr', self.learning_rate)
            self.lr_updates = 0

        # Weight decay variable
        if pkeys.WEIGHT_DECAY_FACTOR in self.params.keys():
            wd_factor = self.params[pkeys.WEIGHT_DECAY_FACTOR]
        else:
            wd_factor = None
        if wd_factor is not None:
            with tf.variable_scope('weight_decay'):
                # Has the same decay than learning rate
                self.weight_decay = wd_factor * (self.learning_rate / init_lr)
                self.wd_summ = tf.summary.scalar('wd', self.weight_decay)

        with tf.variable_scope('feeding'):
            # Training iterator
            self.iterator_train = feeding.get_iterator_splitted(
                (self.feats_train_1_ph, self.labels_train_1_ph, self.masks_train_1_ph),
                (self.feats_train_2_ph, self.labels_train_2_ph, self.masks_train_2_ph),
                batch_size=self.params[pkeys.BATCH_SIZE],
                shuffle_buffer_size=self.params[pkeys.SHUFFLE_BUFFER_SIZE],
                map_fn=self._train_map_fn,
                prefetch_buffer_size=self.params[pkeys.PREFETCH_BUFFER_SIZE],
                name='iter_train')

            # Evaluation iterator
            self.iterator_eval = feeding.get_iterator(
                (self.feats_eval_ph, self.labels_eval_ph, self.masks_eval_ph),
                batch_size=self.params[pkeys.BATCH_SIZE],
                shuffle_buffer_size=0,
                map_fn=self._eval_map_fn,
                prefetch_buffer_size=1,
                name='iter_eval')

            # Global iterator
            iterators_list = [self.iterator_train, self.iterator_eval]
            self.iterator = feeding.get_global_iterator(
                    self.handle_ph, iterators_list,
                    name='iters')
            self.feats, self.labels, self.masks = self.iterator.get_next()

        # Model prediction
        self.logits, self.probabilities, self.other_outputs_dict = self._model_fn()

        # Add training operations
        self.loss, self.loss_sum = self._loss_fn()
        self.train_step, self.reset_optimizer, self.grad_norm_summ = self._optimizer_fn()

        # Evaluation metrics
        self.batch_metrics_dict, self.batch_metrics_summ = self._batch_metrics_fn()
        self.eval_metrics_dict, self.eval_metrics_summ = self._eval_metrics_fn()

        # Fusion of all summaries
        self.merged = tf.summary.merge_all()

        # AF1 in validation
        with tf.variable_scope("by_event_metrics"):
            self.eval_threshold = tf.placeholder(tf.float32, shape=[], name='threshold_ph')
            self.eval_af1 = tf.placeholder(tf.float32, shape=[], name='af1_ph')
            self.eval_af1_half = tf.placeholder(tf.float32, shape=[], name='af1_half_ph')
            self.eval_f1 = tf.placeholder(tf.float32, shape=[], name='f1_ph')
            self.eval_precision = tf.placeholder(tf.float32, shape=[], name='precision_ph')
            self.eval_recall = tf.placeholder(tf.float32, shape=[], name='recall_ph')
            self.eval_miou = tf.placeholder(tf.float32, shape=[], name='miou_ph')
            eval_threshold_summ = tf.summary.scalar('threshold', self.eval_threshold)
            eval_af1_summ = tf.summary.scalar('af1', self.eval_af1)
            eval_af1_half_summ = tf.summary.scalar('af1_half', self.eval_af1_half)
            eval_f1_summ = tf.summary.scalar('f1', self.eval_f1)
            eval_precision_summ = tf.summary.scalar('precision', self.eval_precision)
            eval_recall_summ = tf.summary.scalar('recall', self.eval_recall)
            eval_miou_summ = tf.summary.scalar('miou', self.eval_miou)
            self.byevent_metrics_summ = tf.summary.merge([
                eval_threshold_summ, eval_af1_summ, eval_af1_half_summ,
                eval_f1_summ, eval_recall_summ, eval_precision_summ, eval_miou_summ])

        # Tensorflow session for graph management
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        # Get handles for iterators
        with tf.variable_scope('handles'):
            handles_list = self.sess.run(
                [iterator.string_handle() for iterator in iterators_list])
            self.handle_train = handles_list[0]
            self.handle_eval = handles_list[1]

        # Saver for checkpoints
        self.saver = tf.train.Saver()

        # Summary writers
        self.train_writer = tf.summary.FileWriter(
            os.path.join(self.logdir, 'train'))
        self.val_writer = tf.summary.FileWriter(
            os.path.join(self.logdir, 'val'))
        self.train_writer.add_graph(self.sess.graph)

        # Initialization op
        self.init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer())

        # Save the parameters used to define this model
        with open(os.path.join(self.logdir, 'params.json'), 'w') as outfile:
            json.dump(self.params, outfile)

    def predict_dataset(
            self,
            data_inference: FeederDataset,
            verbose=False,
            signal_transform_fn=None,  # For perturbation experiments
            time_reverse=False,  # For temporal inversion experiment
    ):
        with_augmented_page = self.params[pkeys.PREDICT_WITH_AUGMENTED_PAGE]
        border_size = int(np.round(self.params[pkeys.BORDER_DURATION] * self.params[pkeys.FS]))
        x_inference, _ = data_inference.get_data_for_prediction(
            border_size=border_size,
            predict_with_augmented_page=with_augmented_page,
            verbose=False)

        if signal_transform_fn is None:
            # print("Signal transform fn given is None. Defaults to identity")

            def signal_transform_fn(x):
                return x
        x_inference = [signal_transform_fn(single_x) for single_x in x_inference]

        if time_reverse:  # Reverse time
            # print("Reversing time")
            x_inference = [np.flip(single_x, axis=1) for single_x in x_inference]

        probabilities_list = self.predict_proba_with_list(
            x_inference, verbose=verbose, with_augmented_page=with_augmented_page)

        if time_reverse:  # Recover proper time ordering
            probabilities_list = [np.flip(single_p, axis=1) for single_p in probabilities_list]

        # Now create PredictedDataset object
        probabilities_dict = {}
        all_ids = data_inference.get_ids()
        for k, sub_id in enumerate(all_ids):
            this_proba = probabilities_list[k]
            # Transform to whole-night probability vector
            this_proba = pages2seq(
                this_proba,
                data_inference.get_subject_pages(sub_id, constants.WN_RECORD))
            probabilities_dict[sub_id] = this_proba
        prediction = PredictedDataset(
            dataset=data_inference,
            probabilities_dict=probabilities_dict,
            params=self.params.copy())
        return prediction

    def predict_proba_from_vector(self, x, with_augmented_page=False):
        """Vector is 1D and assumed to be normalized."""
        page_size = self.params[pkeys.PAGE_DURATION] * self.params[pkeys.FS]
        border_size = int(np.round(self.params[pkeys.BORDER_DURATION] * self.params[pkeys.FS]))

        # Compute border to be added
        if with_augmented_page:
            total_border = page_size // 2 + border_size
        else:
            total_border = border_size

        original_samples = x.size
        samples_needed = int(np.ceil(original_samples / page_size) * page_size)
        x_padded = np.zeros(samples_needed, dtype=x.dtype)
        x_padded[:original_samples] = x
        n_pages = int(x_padded.size / page_size)
        pages = np.arange(n_pages)
        x_batched = extract_pages(x_padded, pages, page_size, border_size=total_border)
        x_batched = x_batched.astype(np.float32)

        y_batched = self.predict_proba(x_batched, with_augmented_page=with_augmented_page)

        y = pages2seq(y_batched, pages)

        original_samples_downsampled = original_samples // self.params[pkeys.TOTAL_DOWNSAMPLING_FACTOR]
        y = y[:original_samples_downsampled]
        return y

    def predict_proba(self, x, with_augmented_page=False):
        """Predicts the class probabilities over the data x."""
        niters = np.ceil(x.shape[0] / self.params[pkeys.BATCH_SIZE])
        niters = int(niters)
        probabilities_list = []
        for i in range(niters):
            start_index = i*self.params[pkeys.BATCH_SIZE]
            end_index = (i+1)*self.params[pkeys.BATCH_SIZE]
            batch = x[start_index:end_index]

            if with_augmented_page:
                page_size = self.params[pkeys.PAGE_DURATION] * self.params[pkeys.FS]
                border_size = int(np.round(self.params[pkeys.BORDER_DURATION] * self.params[pkeys.FS]))
                input_size = page_size + 2 * border_size
                start_left = int(page_size / 4)
                end_left = int(start_left + input_size)
                start_right = int(3 * page_size / 4)
                end_right = int(start_right + input_size)

                batch_left = batch[:, start_left:end_left]
                batch_right = batch[:, start_right:end_right]

                proba_left = self.sess.run(
                    self.probabilities,
                    feed_dict={
                        self.feats: batch_left,
                        self.training_ph: False
                    })
                proba_right = self.sess.run(
                    self.probabilities,
                    feed_dict={
                        self.feats: batch_right,
                        self.training_ph: False
                    })
                # Keep central half of each
                length_out = proba_left.shape[1]
                start_crop = int(length_out / 4)
                end_crop = int(3 * length_out / 4)
                crop_left = proba_left[:, start_crop:end_crop, :]
                crop_right = proba_right[:, start_crop:end_crop, :]
                probabilities = np.concatenate([crop_left, crop_right], axis=1)
            else:
                probabilities = self.sess.run(
                    self.probabilities,
                    feed_dict={
                        self.feats: batch,
                        self.training_ph: False
                    })
            probabilities_list.append(probabilities)
        final_probabilities = np.concatenate(probabilities_list, axis=0)

        # Keep only probability of class 1
        final_probabilities = final_probabilities[..., 1]

        # Transform to float16 precision
        final_probabilities = final_probabilities.astype(np.float16)

        return final_probabilities

    def predict_proba_with_list(
            self, x_list, verbose=False, with_augmented_page=False):
        """Predicts the class probabilities over a list of data x."""
        probabilities_list = []
        if verbose:
            if with_augmented_page:
                print('Predicting with augmented page')
            else:
                print('Predicting with regular size page')
        for i, x in enumerate(x_list):
            if verbose:
                print('Predicting %d / %d ... '
                      % (i+1, len(x_list)), end='', flush=True)
            this_pred = self.predict_proba(
                x, with_augmented_page=with_augmented_page)
            probabilities_list.append(this_pred)
            if verbose:
                print('Done', flush=True)
        return probabilities_list

    def predict_tensor_at_samples(self, x, samples, tensor_name='last_hidden'):
        page_size = self.params[pkeys.PAGE_DURATION] * self.params[pkeys.FS]
        border_size = int(np.round(self.params[pkeys.BORDER_DURATION] * self.params[pkeys.FS]))
        # x is a whole-night signal
        x = extract_pages_from_centers(x, samples, page_size, border_size)
        niters = np.ceil(x.shape[0] / self.params[pkeys.BATCH_SIZE])
        niters = int(niters)

        result = []
        for i in range(niters):
            start_index = i * self.params[pkeys.BATCH_SIZE]
            end_index = (i + 1) * self.params[pkeys.BATCH_SIZE]
            batch = x[start_index:end_index]

            tensors = self.sess.run(
                self.other_outputs_dict[tensor_name],
                feed_dict={
                    self.feats: batch,
                    self.training_ph: False
                })
            center_loc = tensors.shape[1] // 2
            tensors_at_center = tensors[:, center_loc]
            result.append(tensors_at_center)
        result = np.concatenate(result, axis=0)
        return result

    def predict_tensor_at_samples_with_list(self, x_list, samples_list, tensor_name='last_hidden', verbose=False):
        result_list = []
        assert len(x_list) == len(samples_list)
        n = len(x_list)
        for i in range(n):
            if verbose:
                print('Predicting %d / %d ... ' % (i+1, n), end='', flush=True)
            this_result = self.predict_tensor_at_samples(x_list[i], samples_list[i], tensor_name=tensor_name)
            result_list.append(this_result)
            if verbose:
                print('Done', flush=True)
        return result_list

    def evaluate(self, x, y, m):
        """Evaluates the model, averaging evaluation metrics over batches."""
        self._init_iterator_eval(x, y, m)
        niters = np.ceil(x.shape[0] / self.params[pkeys.BATCH_SIZE])
        niters = int(niters)
        metrics_list = []
        for i in range(niters):
            eval_metrics = self.sess.run(
                self.eval_metrics_dict,
                feed_dict={self.training_ph: False,
                           self.handle_ph: self.handle_eval})
            metrics_list.append(eval_metrics)
        # Average
        mean_metrics = {}
        for key in self.eval_metrics_dict:
            value = 0
            for i in range(niters):
                value += metrics_list[i][key]
            mean_metrics[key] = value / niters
        # Create summary to write
        feed_dict = {}
        for key in self.eval_metrics_dict:
            feed_dict.update(
                {self.eval_metrics_dict[key]:  mean_metrics[key]}
            )
        mean_loss, mean_metrics, mean_summ = self.sess.run(
            [self.loss, self.batch_metrics_dict, self.eval_metrics_summ],
            feed_dict=feed_dict)
        return mean_loss, mean_metrics, mean_summ

    def load_checkpoint(self, ckptdir):
        """Loads variables from a checkpoint."""
        self.saver.restore(self.sess, ckptdir)

    def _init_iterator_train(
            self,
            x_train_1, y_train_1, m_train_1,
            x_train_2, y_train_2, m_train_2):
        """Init the train iterator."""
        self.sess.run(
            self.iterator_train.initializer,
            feed_dict={
                self.feats_train_1_ph: x_train_1,
                self.labels_train_1_ph: y_train_1,
                self.masks_train_1_ph: m_train_1,
                self.feats_train_2_ph: x_train_2,
                self.labels_train_2_ph: y_train_2,
                self.masks_train_2_ph: m_train_2
            })

    def _init_iterator_eval(
            self, x_eval, y_eval, m_eval):
        """Init the evaluation iterator."""
        self.sess.run(
            self.iterator_eval.initializer,
            feed_dict={
                self.feats_eval_ph: x_eval,
                self.labels_eval_ph: y_eval,
                self.masks_eval_ph: m_eval
            })

    def _update_learning_rate(self, update_factor, ckptdir=None):
        # Restore checkpoint
        if ckptdir:
            self.load_checkpoint(ckptdir)
        if self.params[pkeys.LR_UPDATE_RESET_OPTIMIZER]:
            # Reset optimizer variables (like moving averages)
            print("Resetting optimizer variables")
            self.sess.run(self.reset_optimizer)
        # Decrease learning rate
        self.lr_updates = self.lr_updates + 1
        # total_factor = update_factor ** self.lr_updates
        # new_lr = self.params[pkeys.LEARNING_RATE] * total_factor
        current_lr = self.sess.run(self.learning_rate)
        new_lr = current_lr * update_factor
        self.sess.run(tf.assign(self.learning_rate, new_lr))
        return new_lr

    def _single_train_iteration(self):
        self.sess.run(
            self.train_step,
            feed_dict={self.training_ph: True, self.handle_ph: self.handle_train})

    def _initialize_variables(self):
        self.sess.run(self.init_op)

    def fit(self, data_train, data_val):
        """This method has to be implemented."""
        pass

    def _train_map_fn(self, feat, label, mask):
        """This method has to be implemented."""
        return feat, label, mask

    def _eval_map_fn(self, feat, label, mask):
        """This method has to be implemented."""
        return feat, label, mask

    def _model_fn(self):
        """This method has to be implemented"""
        logits = None
        probabilities = None
        other_outputs_dict = None
        return logits, probabilities, other_outputs_dict

    def _loss_fn(self):
        """This method has to be implemented"""
        loss = None
        loss_summ = None
        return loss, loss_summ

    def _optimizer_fn(self):
        """This method has to be implemented"""
        train_step = None
        reset_optimizer_op = None
        grad_norm_summ = None
        return train_step, reset_optimizer_op, grad_norm_summ

    def _batch_metrics_fn(self):
        """This method has to be implemented"""
        metrics_dict = {}
        metrics_summ = None
        return metrics_dict, metrics_summ

    def _eval_metrics_fn(self):
        """This method has to be implemented"""
        metrics_dict = {}
        metrics_summ = None
        return metrics_dict, metrics_summ
