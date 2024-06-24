"""models.py: Module that defines trainable models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import time

import numpy as np
import tensorflow as tf

from sleeprnn.common import pkeys
from sleeprnn.common import constants
from sleeprnn.common import checks
from sleeprnn.data import utils
from sleeprnn.detection import threshold_optimization
from sleeprnn.detection.metrics import matching_with_list
from sleeprnn.detection.metrics import metric_vs_iou_macro_average, metric_vs_iou_micro_average
from sleeprnn.detection.metrics import average_metric_macro_average, average_metric_micro_average
from sleeprnn.detection.feeder_dataset import FeederDataset
from sleeprnn.nn.base_model import BaseModel
from sleeprnn.nn.base_model import KEY_LOSS
from sleeprnn.nn import networks, networks_v2, networks_v3
from sleeprnn.nn import losses, optimizers, metrics, augmentations

# Metrics dict
KEY_TP = 'tp'
KEY_FP = 'fp'
KEY_FN = 'fn'
KEY_PRECISION = 'precision'
KEY_RECALL = 'recall'
KEY_F1_SCORE = 'f1_score'
KEY_AF1 = 'af1'

# Fit dicts
KEY_ITER = 'iteration'


# TODO: Remove BaseModel class (is unnecessary)
class WaveletBLSTM(BaseModel):
    """ Model that manages the implemented network."""

    def __init__(self, params=None, logdir='logs'):
        """Constructor.

        Feat and label shapes can be obtained from params for this model.
        """
        self.params = pkeys.default_params.copy()
        if params is not None:
            self.params.update(params)  # Overwrite defaults
        border_size = self.get_border_size()
        page_size = self.get_page_size()
        augmented_input_length = 2*(page_size + border_size)
        time_stride = self.params[pkeys.TOTAL_DOWNSAMPLING_FACTOR]
        feat_train_shape = [augmented_input_length]
        label_train_shape = feat_train_shape
        feat_eval_shape = [page_size + 2 * border_size]
        label_eval_shape = [page_size / time_stride]
        super().__init__(
            feat_train_shape,
            label_train_shape,
            feat_eval_shape,
            label_eval_shape,
            params, logdir)

    def get_border_size(self):
        border_duration = self.params[pkeys.BORDER_DURATION]
        fs = self.params[pkeys.FS]
        border_size = int(np.round(fs * border_duration))
        return border_size

    def get_page_size(self):
        page_duration = self.params[pkeys.PAGE_DURATION]
        fs = self.params[pkeys.FS]
        page_size = int(np.round(fs * page_duration))
        return page_size

    def check_train_inputs(self, x_train, y_train, m_train, x_val, y_val, m_val):
        """Ensures that validation data has the proper shape."""
        time_stride = self.params[pkeys.TOTAL_DOWNSAMPLING_FACTOR]
        border_size = self.get_border_size()
        page_size = self.get_page_size()
        crop_size = page_size + 2 * border_size
        if x_train.shape[1] == x_val.shape[1]:
            # If validation has augmented pages
            x_val = x_val[:, page_size // 2:-page_size // 2]
            y_val = y_val[:, page_size // 2:-page_size // 2]
            m_val = m_val[:, page_size // 2:-page_size // 2]
        if y_val.shape[1] == crop_size:
            # We need to remove borders and downsampling for val labels.
            y_val = y_val[:, border_size:-border_size]
            m_val = m_val[:, border_size:-border_size]
            aligned_down = self.params[pkeys.ALIGNED_DOWNSAMPLING]
            if aligned_down:
                print('ALIGNED DOWNSAMPLING at checking inputs for fit')
                y_val_dtype = y_val.dtype
                y_val = y_val.reshape((-1, int(page_size/time_stride), time_stride))
                y_val = np.round(y_val.mean(axis=-1) + 1e-3).astype(y_val_dtype)

                m_val_dtype = m_val.dtype
                m_val = m_val.reshape((-1, int(page_size / time_stride), time_stride))
                m_val = np.round(m_val.mean(axis=-1) + 1e-3).astype(m_val_dtype)
            else:
                y_val = y_val[:, ::time_stride]
                m_val = m_val[:, ::time_stride]
        return x_train, y_train, m_train, x_val, y_val, m_val

    def fit_without_validation(
            self,
            data_train: FeederDataset,
            fine_tune=False,
            extra_data_train=None,
            verbose=False):
        """Fits the model to the training data."""
        border_size = self.get_border_size()
        forced_mark_separation_size = int(
            self.params[pkeys.FORCED_SEPARATION_DURATION] * self.params[pkeys.FS])

        x_train, y_train, m_train = data_train.get_data_for_training(
            border_size=border_size,
            forced_mark_separation_size=forced_mark_separation_size,
            return_page_mask=True,
            verbose=verbose)

        # Transform to numpy arrays
        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        m_train = np.concatenate(m_train, axis=0)

        # Add extra training data
        if extra_data_train is not None:
            x_extra, y_extra, m_extra = extra_data_train
            print('CHECK: Sum extra y:', y_extra.sum())
            print('Current train data x, y, m:', x_train.shape, y_train.shape, m_train.shape)
            x_train = np.concatenate([x_train, x_extra], axis=0)
            y_train = np.concatenate([y_train, y_extra], axis=0)
            m_train = np.concatenate([m_train, m_extra], axis=0)
            print('Extra data to be added x, y, m:', x_extra.shape, y_extra.shape, m_extra.shape)
            print('New train data', x_train.shape, y_train.shape, m_train.shape)
            del extra_data_train

        # Shuffle training set
        list_of_outputs = utils.shuffle_data_collection([x_train, y_train, m_train], seed=0)
        x_train, y_train, m_train = list_of_outputs[0], list_of_outputs[1], list_of_outputs[2]

        print('Training set shape', x_train.shape, y_train.shape, m_train.shape)
        print('Validation set does not exist')

        x_train_1, y_train_1, m_train_1, x_train_2, y_train_2, m_train_2 = self._split_train(
            x_train, y_train, m_train)
        del x_train, y_train, m_train

        batch_size = self.params[pkeys.BATCH_SIZE]
        iters_resolution = 10
        n_smallest = min(x_train_1.shape[0], x_train_2.shape[0])
        iter_per_epoch = int(n_smallest / (batch_size / 2))
        iter_per_epoch = int((iter_per_epoch // iters_resolution) * iters_resolution)

        nstats = iter_per_epoch // self.params[pkeys.STATS_PER_EPOCH]
        niters_init = self.params[pkeys.PRETRAIN_EPOCHS_INIT] * iter_per_epoch
        niters_anneal = self.params[pkeys.PRETRAIN_EPOCHS_ANNEAL] * iter_per_epoch
        n_lr_updates = self.params[pkeys.PRETRAIN_MAX_LR_UPDATES]
        total_iters = niters_init + n_lr_updates * niters_anneal

        print('\nBeginning training at logdir "%s"' % self.logdir)
        print('Batch size %d, Iters per epoch %d, '
              'Training examples %d, Init iters %d, Annealing iters %d, Total iters %d' %
              (self.params[pkeys.BATCH_SIZE],
               iter_per_epoch,
               x_train_1.shape[0] + x_train_2.shape[0], niters_init, niters_anneal, total_iters))
        print('Initial learning rate:', self.params[pkeys.LEARNING_RATE])
        if pkeys.WEIGHT_DECAY_FACTOR in self.params.keys():
            print('Initial weight decay:', self.params[pkeys.WEIGHT_DECAY_FACTOR])

        if fine_tune:
            init_lr = self.params[pkeys.LEARNING_RATE]
            factor_fine_tune = self.params[pkeys.FACTOR_INIT_LR_FINE_TUNE]
            init_lr_fine_tune = init_lr * factor_fine_tune
            self.sess.run(self.reset_optimizer)
            self.sess.run(tf.assign(self.learning_rate, init_lr_fine_tune))
            print('Fine tuning with lr %s' % init_lr_fine_tune)
        else:
            self._initialize_variables()

        self._init_iterator_train(x_train_1, y_train_1, m_train_1, x_train_2, y_train_2, m_train_2)
        del x_train_1, y_train_1, m_train_1, x_train_2, y_train_2, m_train_2

        # Training loop
        start_time = time.time()
        last_elapsed = 0
        last_it = 0
        iter_last_lr_update = niters_init - niters_anneal

        for it in range(1, total_iters+1):
            self._single_train_iteration()
            if it % nstats == 0 or it == 1 or it == total_iters:
                # Report stuff. Training report is batch report
                train_loss, train_metrics, train_summ = self.sess.run(
                    [self.loss, self.batch_metrics_dict, self.merged],
                    feed_dict={self.training_ph: False, self.handle_ph: self.handle_train})
                self.train_writer.add_summary(train_summ, it)
                elapsed = time.time() - start_time
                time_rate_per_100 = 100 * (elapsed - last_elapsed) / (it - last_it)
                last_it = it
                last_elapsed = elapsed
                loss_print = ('loss train %1.4f' % train_loss)
                f1_print = ('f1 train %1.4f' % train_metrics[KEY_F1_SCORE])
                print('It %6.0d/%d - %s - %s - E.T. %1.2fs (%1.2fs/100it)'
                      % (it, total_iters, loss_print, f1_print, elapsed, time_rate_per_100))
            # The last lr update is far enough
            lr_criterion = (it - iter_last_lr_update) >= niters_anneal
            if lr_criterion:
                new_lr = self._update_learning_rate(self.params[pkeys.LR_UPDATE_FACTOR])
                print('    Learning rate update (%d). New value: %s' % (self.lr_updates, new_lr))
                iter_last_lr_update = it
        # Final stats
        iter_saved_model = total_iters
        elapsed = time.time() - start_time
        print('\n\nTotal training time: %1.4f s' % elapsed)
        print('Ending at iteration %d' % iter_saved_model)
        save_path = self.saver.save(self.sess, self.ckptdir)
        print('Model saved at %s' % save_path)
        last_model = {
            KEY_ITER: iter_saved_model,
            KEY_LOSS: 0,
            KEY_F1_SCORE: 0
        }
        # Save last model quick info
        with open(os.path.join(self.logdir, 'last_model.json'), 'w') as outfile:
            json.dump(last_model, outfile)

    def fit(
            self,
            data_train: FeederDataset,
            data_val: FeederDataset,
            fine_tune=False,
            extra_data_train=None,
            verbose=False):
        """Fits the model to the training data."""
        border_size = self.get_border_size()
        forced_mark_separation_size = int(
            self.params[pkeys.FORCED_SEPARATION_DURATION] * self.params[pkeys.FS])

        x_train, y_train, m_train = data_train.get_data_for_training(
            border_size=border_size,
            forced_mark_separation_size=forced_mark_separation_size,
            return_page_mask=True,
            verbose=verbose)
        x_val, y_val, m_val = data_val.get_data_for_training(
            border_size=border_size,
            forced_mark_separation_size=forced_mark_separation_size,
            return_page_mask=True,
            verbose=verbose)

        # Transform to numpy arrays
        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        m_train = np.concatenate(m_train, axis=0)
        x_val = np.concatenate(x_val, axis=0)
        y_val = np.concatenate(y_val, axis=0)
        m_val = np.concatenate(m_val, axis=0)

        # Add extra training data
        if extra_data_train is not None:
            x_extra, y_extra, m_extra = extra_data_train
            print('CHECK: Sum extra y:', y_extra.sum())
            print('Current train data x, y, m:', x_train.shape, y_train.shape, m_train.shape)
            x_train = np.concatenate([x_train, x_extra], axis=0)
            y_train = np.concatenate([y_train, y_extra], axis=0)
            m_train = np.concatenate([m_train, m_extra], axis=0)
            print('Extra data to be added x, y, m:', x_extra.shape, y_extra.shape, m_extra.shape)
            print('New train data', x_train.shape, y_train.shape, m_train.shape)
            del extra_data_train

        # Shuffle training set
        list_of_outputs = utils.shuffle_data_collection([x_train, y_train, m_train], seed=0)
        x_train, y_train, m_train = list_of_outputs[0], list_of_outputs[1], list_of_outputs[2]

        print('Training set shape', x_train.shape, y_train.shape, m_train.shape)
        print('Validation set shape', x_val.shape, y_val.shape, m_val.shape)

        x_train, y_train, m_train, x_val, y_val, m_val = self.check_train_inputs(
            x_train, y_train, m_train, x_val, y_val, m_val)
        x_train_1, y_train_1, m_train_1, x_train_2, y_train_2, m_train_2 = self._split_train(
            x_train, y_train, m_train)
        del x_train, y_train, m_train

        batch_size = self.params[pkeys.BATCH_SIZE]
        iters_resolution = 10
        n_smallest = min(x_train_1.shape[0], x_train_2.shape[0])
        iter_per_epoch = n_smallest / (batch_size / 2)
        iter_per_epoch = int(iters_resolution * max(np.round(iter_per_epoch / iters_resolution), 1))

        niters = self.params[pkeys.MAX_EPOCHS] * iter_per_epoch
        iters_lr_update = self.params[pkeys.EPOCHS_LR_UPDATE] * iter_per_epoch
        nstats = iter_per_epoch // self.params[pkeys.STATS_PER_EPOCH]

        print('\nBeginning training at logdir "%s"' % self.logdir)
        print('Batch size %d, Iters per epoch %d, '
              'Training examples %d, Max iterations %d' %
              (self.params[pkeys.BATCH_SIZE],
               iter_per_epoch,
               x_train_1.shape[0] + x_train_2.shape[0], niters))
        print('Initial learning rate:', self.params[pkeys.LEARNING_RATE])
        if pkeys.WEIGHT_DECAY_FACTOR in self.params.keys():
            print('Initial weight decay:', self.params[pkeys.WEIGHT_DECAY_FACTOR])

        if fine_tune:
            init_lr = self.params[pkeys.LEARNING_RATE]
            factor_fine_tune = self.params[pkeys.FACTOR_INIT_LR_FINE_TUNE]
            init_lr_fine_tune = init_lr * factor_fine_tune
            self.sess.run(self.reset_optimizer)
            self.sess.run(tf.assign(self.learning_rate, init_lr_fine_tune))
            print('Fine tuning with lr %s' % init_lr_fine_tune)
        else:
            self._initialize_variables()

        self._init_iterator_train(x_train_1, y_train_1, m_train_1, x_train_2, y_train_2, m_train_2)
        del x_train_1, y_train_1, m_train_1, x_train_2, y_train_2, m_train_2

        # Improvement criterion
        model_criterion = {
            KEY_ITER: 0,
            KEY_LOSS: 1e10,
            KEY_F1_SCORE: 0,
            KEY_AF1: 0,
        }
        rel_tol_criterion = self.params[pkeys.REL_TOL_CRITERION]
        iter_last_lr_update = 0

        lr_update_criterion = self.params[pkeys.LR_UPDATE_CRITERION]
        checks.check_valid_value(
            lr_update_criterion,
            'lr_update_criterion',
            [constants.LOSS_CRITERION, constants.METRIC_CRITERION])
        print("Learning rate decay criterion: %s" % lr_update_criterion)

        # Validation events for AF1
        val_avg_mode = self.params[pkeys.VALIDATION_AVERAGE_MODE]
        if val_avg_mode is None:
            # Default values
            if 'moda' in data_val.dataset_name:
                val_avg_mode = constants.MICRO_AVERAGE
            else:
                val_avg_mode = constants.MACRO_AVERAGE
        print("Validation AF1 computed using %s and thr 0.5" % val_avg_mode)

        # Training loop
        start_time = time.time()
        last_elapsed = 0
        last_it = 0
        for it in range(1, niters+1):
            self._single_train_iteration()
            if it % nstats == 0 or it == 1 or it == niters:
                metric_msg = 'It %6.0d/%d' % (it, niters)
                # Train set report (mini-batch)
                train_loss, train_metrics, train_summ = self.sess.run(
                    [self.loss, self.batch_metrics_dict, self.merged],
                    feed_dict={self.training_ph: False, self.handle_ph: self.handle_train})
                self.train_writer.add_summary(train_summ, it)
                metric_msg += ' - train loss %1.4f f1 %1.4f' % (train_loss, train_metrics[KEY_F1_SCORE])
                if it % iter_per_epoch == 0 or it == 1 or it == niters:
                    # Val set report (whole set)
                    val_loss, val_metrics, val_summ = self.evaluate(x_val, y_val, m_val)
                    self.val_writer.add_summary(val_summ, it)
                    byevent_val_metrics, byevent_val_summ = self.evaluate_byevent(data_val, val_avg_mode)
                    self.val_writer.add_summary(byevent_val_summ, it)

                    metric_msg += ' - val loss %1.4f f1 %1.4f AF1 %1.4f (thr %1.2f)' % (
                        val_loss, val_metrics[KEY_F1_SCORE],
                        byevent_val_metrics['af1'], byevent_val_metrics['threshold'])
                    # Time passed
                    elapsed = time.time() - start_time
                    time_rate_per_100 = 100 * (elapsed - last_elapsed) / (it - last_it)
                    last_it = it
                    last_elapsed = elapsed
                    metric_msg += ' - E.T. %1.2fs (%1.2fs/100it)' % (elapsed, time_rate_per_100)
                    print(metric_msg)

                    if lr_update_criterion == constants.LOSS_CRITERION:
                        improvement_criterion = val_loss < (1.0 - rel_tol_criterion) * model_criterion[KEY_LOSS]
                    else:
                        improvement_criterion = byevent_val_metrics['af1'] > (1.0 + rel_tol_criterion) * model_criterion[KEY_AF1]
                    if improvement_criterion:
                        # Update last time the improvement criterion was met
                        model_criterion[KEY_LOSS] = val_loss
                        model_criterion[KEY_F1_SCORE] = val_metrics[KEY_F1_SCORE]
                        model_criterion[KEY_ITER] = it
                        model_criterion[KEY_AF1] = byevent_val_metrics['af1']
                        # Save best model
                        if self.params[pkeys.KEEP_BEST_VALIDATION]:
                            print("Checkpointing best model so far.")
                            self.saver.save(self.sess, self.ckptdir)

                    # Check LR update criterion

                    # The model has not improved for long time
                    lr_criterion_1 = (it - model_criterion[KEY_ITER]) >= iters_lr_update
                    # The last lr update is far enough
                    lr_criterion_2 = (it - iter_last_lr_update) >= iters_lr_update
                    lr_criterion = lr_criterion_1 and lr_criterion_2
                    if lr_criterion:
                        if self.lr_updates < self.params[pkeys.MAX_LR_UPDATES]:
                            # if self.params[pkeys.KEEP_BEST_VALIDATION]:
                            #     print('Restoring best model before lr update')
                            #     self.load_checkpoint(self.ckptdir)
                            new_lr = self._update_learning_rate(self.params[pkeys.LR_UPDATE_FACTOR])
                            print('    Learning rate update (%d). New value: %s' % (self.lr_updates, new_lr))
                            iter_last_lr_update = it
                        else:
                            print('    Maximum number (%d) of learning rate '
                                  'updates reached. Stopping training.'
                                  % self.params[pkeys.MAX_LR_UPDATES])
                            # Since we stop training, redefine number of iters
                            niters = it
                            break
                else:
                    print(metric_msg)

        if self.params[pkeys.KEEP_BEST_VALIDATION]:
            iter_saved_model = model_criterion[KEY_ITER]
            print('Restoring best model from it %d' % iter_saved_model)
            self.load_checkpoint(self.ckptdir)
        else:
            print('Keeping model from last iteration')
            iter_saved_model = niters

        val_loss, val_metrics, _ = self.evaluate(x_val, y_val, m_val)
        last_model = {
            KEY_ITER: iter_saved_model,
            KEY_LOSS: float(val_loss),
            KEY_F1_SCORE: float(val_metrics[KEY_F1_SCORE])
        }

        # Final stats
        elapsed = time.time() - start_time
        print('\n\nTotal training time: %1.4f s' % elapsed)
        print('Ending at iteration %d' % last_model[KEY_ITER])
        print('Validation loss %1.6f - f1 %1.6f'
              % (last_model[KEY_LOSS], last_model[KEY_F1_SCORE]))

        save_path = self.saver.save(self.sess, self.ckptdir)
        print('Model saved at %s' % save_path)

        # Save last model quick info
        with open(os.path.join(self.logdir, 'last_model.json'), 'w') as outfile:
            json.dump(last_model, outfile)

    def evaluate_byevent(
            self, validation_dataset, average_mode, iou_threshold_report=0.2):

        metric_vs_iou_fn_dict = {
            constants.MACRO_AVERAGE: metric_vs_iou_macro_average,
            constants.MICRO_AVERAGE: metric_vs_iou_micro_average}
        average_metric_fn_dict = {
            constants.MACRO_AVERAGE: average_metric_macro_average,
            constants.MICRO_AVERAGE: average_metric_micro_average}

        prediction_val = self.predict_dataset(validation_dataset, verbose=False)

        byevent_thr = 0.5
        prediction_val.set_probability_threshold(byevent_thr)

        val_events_list = validation_dataset.get_stamps()
        val_detections_list = prediction_val.get_stamps()

        iou_matching_list, _ = matching_with_list(val_events_list, val_detections_list)

        byevent_f1 = metric_vs_iou_fn_dict[average_mode](
            val_events_list, val_detections_list, [iou_threshold_report],
            metric_name=constants.F1_SCORE,
            iou_matching_list=iou_matching_list)[0]

        byevent_precision = metric_vs_iou_fn_dict[average_mode](
            val_events_list, val_detections_list, [iou_threshold_report],
            metric_name=constants.PRECISION,
            iou_matching_list=iou_matching_list)[0]

        byevent_recall = metric_vs_iou_fn_dict[average_mode](
            val_events_list, val_detections_list, [iou_threshold_report],
            metric_name=constants.RECALL,
            iou_matching_list=iou_matching_list)[0]

        nonzero_iou_list = [iou_matching[iou_matching > 0] for iou_matching in iou_matching_list]
        if average_mode == constants.MACRO_AVERAGE:
            miou_list = [np.mean(nonzero_iou) for nonzero_iou in nonzero_iou_list]
            byevent_miou = np.mean(miou_list)
        else:
            byevent_miou = np.concatenate(nonzero_iou_list).mean()

        byevent_af1_half = average_metric_fn_dict[average_mode](val_events_list, val_detections_list)

        byevent_metrics = {
            'threshold': byevent_thr,
            'af1': byevent_af1_half,
            'af1_half': byevent_af1_half,
            'f1': byevent_f1,
            'recall': byevent_recall,
            'precision': byevent_precision,
            'miou': byevent_miou}

        byevent_summ = self.sess.run(
            self.byevent_metrics_summ, feed_dict={
                self.eval_threshold: byevent_thr,
                self.eval_af1: byevent_af1_half,
                self.eval_af1_half: byevent_af1_half,
                self.eval_f1: byevent_f1,
                self.eval_precision: byevent_precision,
                self.eval_recall: byevent_recall,
                self.eval_miou: byevent_miou
            }
        )
        return byevent_metrics, byevent_summ

    def _eval_map_fn(self, feat, label, mask):
        label = tf.cast(label, tf.int32)
        mask = tf.cast(mask, tf.int32)
        return feat, label, mask

    def _train_map_fn(self, feat, label, mask):
        """Random cropping.

        This method is used to preprocess features and labels of single
        examples with a random cropping
        """

        # Prepare for training
        time_stride = self.params[pkeys.TOTAL_DOWNSAMPLING_FACTOR]
        border_size = self.get_border_size()
        page_size = self.get_page_size()
        crop_size = page_size + 2 * border_size
        # Random crop
        label_cast = tf.cast(label, dtype=tf.float32)
        mask_cast = tf.cast(mask, dtype=tf.float32)
        stack = tf.stack([feat, label_cast, mask_cast], axis=0)
        stack_crop = tf.random_crop(stack, [3, crop_size])
        feat = stack_crop[0, :]
        label = tf.cast(stack_crop[1, :], dtype=tf.int32)
        mask = tf.cast(stack_crop[2, :], dtype=tf.int32)

        # Apply data augmentation
        feat, label, mask = self._augmentation_fn(feat, label, mask)

        # Throw borders for labels, skipping steps
        # We need to remove borders and downsampling for val labels.
        label = label[border_size:-border_size]
        mask = mask[border_size:-border_size]
        aligned_down = self.params[pkeys.ALIGNED_DOWNSAMPLING]
        if aligned_down:
            print('ALIGNED DOWNSAMPLING at iterator')
            # Label downsampling
            label = tf.cast(label, tf.float32)
            label = tf.reshape(label, [-1, time_stride])
            label = tf.reduce_mean(label, axis=-1)
            label = tf.round(label + 1e-3)
            label = tf.cast(label, tf.int32)
            # Mask downsampling
            mask = tf.cast(mask, tf.float32)
            mask = tf.reshape(mask, [-1, time_stride])
            mask = tf.reduce_mean(mask, axis=-1)
            mask = tf.round(mask + 1e-3)
            mask = tf.cast(mask, tf.int32)
        else:
            label = label[::time_stride]
            mask = mask[::time_stride]
        return feat, label, mask

    def _augmentation_fn(self, feat, label, mask):
        indep_unif_noise_proba = self.params[pkeys.AUG_INDEP_UNIFORM_NOISE_PROBA]
        indep_unif_noise_intens = self.params[pkeys.AUG_INDEP_UNIFORM_NOISE_INTENSITY]
        random_waves_proba = self.params[pkeys.AUG_RANDOM_WAVES_PROBA]
        random_waves_params = self.params[pkeys.AUG_RANDOM_WAVES_PARAMS]
        random_anti_waves_proba = self.params[pkeys.AUG_RANDOM_ANTI_WAVES_PROBA]
        random_anti_waves_params = self.params[pkeys.AUG_RANDOM_ANTI_WAVES_PARAMS]
        print('indep uniform noise proba %s, intens %s' % (indep_unif_noise_proba, indep_unif_noise_intens))
        print('random waves proba %s, params %s' % (random_waves_proba, random_waves_params))
        print('random anti waves proba %s, params %s' % (random_anti_waves_proba, random_anti_waves_params))
        if indep_unif_noise_proba > 0:
            print('Applying INDEPENDENT UNIFORM noise augmentation')
            feat = augmentations.independent_uniform_noise(feat, indep_unif_noise_proba, indep_unif_noise_intens)
        if random_anti_waves_proba > 0:
            print("Applying random anti waves augmentation")
            feat = augmentations.random_anti_waves_wrapper(
                feat, label, random_anti_waves_proba, self.params[pkeys.FS], random_anti_waves_params)
        if random_waves_proba > 0:
            print("Applying random waves augmentation")
            feat = augmentations.random_waves_wrapper(
                feat, label, random_waves_proba, self.params[pkeys.FS], random_waves_params)
        return feat, label, mask

    def _model_fn(self):
        model_version = self.params[pkeys.MODEL_VERSION]
        checks.check_valid_value(
            model_version, 'model_version',
            [
                constants.V2_TIME,
                constants.V2_CWT1D,
             ])
        if model_version == constants.V2_TIME:
            model_fn = networks_v3.redv2_time
        else:
            model_fn = networks_v3.redv2_cwt1d
        logits, probabilities, other_outputs_dict = model_fn(self.feats, self.params, self.training_ph)
        return logits, probabilities, other_outputs_dict

    def _loss_fn(self):
        type_loss = self.params[pkeys.TYPE_LOSS]
        checks.check_valid_value(
            type_loss, 'type_loss',
            [
                constants.MASKED_SOFT_FOCAL_LOSS
            ])
        loss, loss_summ = losses.masked_soft_focal_loss(
            self.logits, self.labels, self.masks,
            self.params[pkeys.CLASS_WEIGHTS],
            self.params[pkeys.SOFT_FOCAL_GAMMA], self.params[pkeys.SOFT_FOCAL_EPSILON])
        return loss, loss_summ

    def _optimizer_fn(self):
        type_optimizer = self.params[pkeys.TYPE_OPTIMIZER]
        checks.check_valid_value(
            type_optimizer, 'type_optimizer',
            [
                constants.ADAM_OPTIMIZER,
            ])
        train_step, reset_optimizer_op, grad_norm_summ = optimizers.adam_optimizer_fn(
            self.loss, self.learning_rate,
            self.params[pkeys.CLIP_NORM])
        return train_step, reset_optimizer_op, grad_norm_summ

    def _batch_metrics_fn(self):
        with tf.variable_scope('batch_metrics'):
            tp, fp, fn = metrics.confusion_matrix(self.logits, self.labels, self.masks)
            precision, recall, f1_score = metrics.precision_recall_f1score(
                tp, fp, fn)
            prec_summ = tf.summary.scalar(KEY_PRECISION, precision)
            rec_summ = tf.summary.scalar(KEY_RECALL, recall)
            f1_summ = tf.summary.scalar(KEY_F1_SCORE, f1_score)
            batch_metrics_dict = {
                KEY_PRECISION: precision,
                KEY_RECALL: recall,
                KEY_F1_SCORE: f1_score,
                KEY_TP: tp,
                KEY_FP: fp,
                KEY_FN: fn
            }
            batch_metrics_summ = tf.summary.merge(
                [prec_summ, rec_summ, f1_summ])
        return batch_metrics_dict, batch_metrics_summ

    def _eval_metrics_fn(self):
        with tf.variable_scope('eval_metrics'):
            eval_metrics_dict = {
                KEY_TP: self.batch_metrics_dict[KEY_TP],
                KEY_FP: self.batch_metrics_dict[KEY_FP],
                KEY_FN: self.batch_metrics_dict[KEY_FN],
                KEY_LOSS: self.loss
            }
            eval_metrics_summ = [
                self.loss_sum,
                self.batch_metrics_summ
            ]
            eval_metrics_summ = tf.summary.merge(eval_metrics_summ)
        return eval_metrics_dict, eval_metrics_summ

    def _split_train(self, x_train, y_train, m_train):
        n_train = x_train.shape[0]
        border_size = self.get_border_size()
        page_size = self.get_page_size()
        # Remove to recover single page from augmented page
        remove_size = border_size + page_size // 2
        activity = y_train[:, remove_size:-remove_size]
        activity = np.sum(activity, axis=1)

        # Find pages with activity
        exists_activity_idx = np.where(activity > 0)[0]

        n_with_activity = exists_activity_idx.shape[0]

        print('Pages with activity: %d (%1.2f %% of total)'
              % (n_with_activity, 100 * n_with_activity / n_train))

        if n_with_activity < n_train/2:
            print('Balancing strategy: zero/exists activity')
            zero_activity_idx = np.where(activity == 0)[0]
            # Pages without any activity
            x_train_1 = x_train[zero_activity_idx]
            y_train_1 = y_train[zero_activity_idx]
            m_train_1 = m_train[zero_activity_idx]
            # Pages with activity
            x_train_2 = x_train[exists_activity_idx]
            y_train_2 = y_train[exists_activity_idx]
            m_train_2 = m_train[exists_activity_idx]
            print('Pages without activity:', x_train_1.shape)
            print('Pages with activity:', x_train_2.shape)
        else:
            print('Balancing strategy: low/high activity')
            sorted_idx = np.argsort(activity)
            low_activity_idx = sorted_idx[:int(n_train/2)]
            high_activity_idx = sorted_idx[int(n_train/2):]
            # Pages with low activity
            x_train_1 = x_train[low_activity_idx]
            y_train_1 = y_train[low_activity_idx]
            m_train_1 = m_train[low_activity_idx]
            # Pages with high activity
            x_train_2 = x_train[high_activity_idx]
            y_train_2 = y_train[high_activity_idx]
            m_train_2 = m_train[high_activity_idx]
            print('Pages with low activity:', x_train_1.shape)
            print('Pages with high activity:', x_train_2.shape)

        return x_train_1, y_train_1, m_train_1, x_train_2, y_train_2, m_train_2

