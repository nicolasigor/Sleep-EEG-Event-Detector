"""Class definition to manipulate data spindle EEG datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import numpy as np

from src.common import pkeys
from src.common import constants
from src.common import checks
from src.data import utils

KEY_EEG = 'signal'
KEY_N2_PAGES = 'n2_pages'
KEY_ALL_PAGES = 'all_pages'
KEY_MARKS = 'marks'
KEY_HYPNOGRAM = 'hypnogram'


class Dataset(object):
    """This is a base class for data micro-events datasets.
    It provides the option to load and create checkpoints of the processed
    data, and provides methods to query data from specific ids or entire
    subsets.
    You have to overwrite the method '_load_from_files'.
    """

    def __init__(
            self,
            dataset_dir,
            load_checkpoint,
            dataset_name,
            all_ids,
            event_name,
            hypnogram_sleep_labels,
            hypnogram_page_duration,
            n_experts=1,
            default_expert=None,
            default_page_subset=constants.N2_RECORD,
            params=None,
            verbose=True,
    ):
        """Constructor.

        Args:
            dataset_dir: (String) Path to the folder containing the dataset.
               This path can be absolute, or relative to the project root.
            load_checkpoint: (Boolean). Whether to load from a checkpoint or to
               load from scratch using the original files of the dataset.
            dataset_name: (String) Name of the dataset. This name will be used for
               checkpoints.
            all_ids: (list of int) List of available IDs.
            event_name: (str) Name of the event annotated in the dataset.
            hypnogram_sleep_labels: (list) IDs corresponding to sleep stages in the
                hypnogram to compute the global standard deviation.
            hypnogram_page_duration: (int) The duration in seconds of the pages in the hypnogram
                (could be different from the page duration for data segmentation in pkeys.PAGE_DURATION).
            n_experts: (Optional, Int, default to 1) Number of available event annotation sets.
            default_expert: (Optional, Int, default to None) The assumed expert whenever an specific expert
                (i.e., an specific set of annotations) is not specified when retrieving data.
                Useful for FeederDataset.
            default_page_subset: (Optional, str, default to 'n2') The assumed page subset (e.g., only N2)
                whenever a specific subset is not specified when retrieving data. Useful for FeederDataset.
            params: (Optional, dict, default to None) Parameters that overwrite those in pkeys.default_params.
                If None, all values are as in the default. Note that you can pass a dictionary
                containing only those parameter that you want to overwrite.
            verbose: (Optional, bool, default to True) Whether you want to print stuff.
        """
        # Save attributes
        if os.path.isabs(dataset_dir):
            self.dataset_dir = dataset_dir
        else:
            self.dataset_dir = os.path.abspath(
                os.path.join(utils.PATH_DATA, dataset_dir))
        # We verify that the directory exists
        checks.check_directory(self.dataset_dir)

        self.load_checkpoint = load_checkpoint
        self.dataset_name = dataset_name
        self.event_name = event_name
        self.hypnogram_sleep_labels = hypnogram_sleep_labels
        self.hypnogram_page_duration = hypnogram_page_duration
        self.n_experts = n_experts
        self.default_expert = 1 if (n_experts == 1) else default_expert
        self.default_page_subset = default_page_subset
        self.ckpt_dir = os.path.abspath(os.path.join(self.dataset_dir, '..', 'ckpt_%s' % self.dataset_name))
        self.all_ids = all_ids
        self.all_ids.sort()
        if verbose:
            print('Dataset %s with %d patients.' % (self.dataset_name, len(self.all_ids)))

        # events and data EEG related parameters
        self.params = pkeys.default_params.copy()
        if params is not None:
            self.params.update(params)  # Overwrite defaults

        # Sampling frequency [Hz] to be used (not the original)
        self.fs = self.params[pkeys.FS]
        # Time of window page [s]
        self.page_duration = self.params[pkeys.PAGE_DURATION]
        self.page_size = int(self.page_duration * self.fs)

        # Ckpt file associated with the sampling frequency
        self.ckpt_file = os.path.join(self.ckpt_dir, '%s_fs%d.pickle' % (self.dataset_name, self.fs))

        # Data loading
        self.data = self._load_data(verbose=verbose)
        self.global_std = 1.0

    def cv_split(self, n_folds, fold_id, seed=0, subject_ids=None):
        """k-fold CV splits, by-subject split.
        Inputs:
            n_folds: number of folds of CV
            fold_id: integer in [0, 1, ..., n_folds - 1] (which fold to retrieve)
            seed: random seed (determines the permutation of subjects to generate k-fold CV)
            subject_ids: optional list of subject to restrict the cv. By default uses all ids.
        Returns 3 lists of IDs: train_ids, val_ids, test_ids.
        """
        if fold_id >= n_folds:
            raise ValueError("fold id %s invalid for %d folds" % (fold_id, n_folds))
        # Retrieve data
        if subject_ids is None:
            subject_ids = np.asarray(self.all_ids.copy())

        n_test = int(np.ceil(len(subject_ids) / n_folds))
        attempts = 1
        while True:
            # Random permutation
            subject_ids_1 = np.random.RandomState(seed=seed).permutation(subject_ids)
            subject_ids_2 = np.random.RandomState(seed=seed + attempts).permutation(subject_ids)
            subject_ids_extended = np.concatenate([subject_ids_1, subject_ids_2])
            # Form folds
            test_folds = []
            for i in range(n_folds):
                start_loc = i * n_test
                end_loc = (i + 1) * n_test
                subset_of_subjects = subject_ids_extended[start_loc:end_loc]
                subset_of_subjects = np.unique(subset_of_subjects)
                test_folds.append(subset_of_subjects)
            # Test two breaking conditions to avoid problems when extending the list
            # Test length condition
            concat_folds = np.concatenate(test_folds)
            cond1 = (concat_folds.size == (n_test * n_folds))
            # val sets and test sets do not overlap
            val_folds = [test_folds[(loc + 1) % n_folds] for loc in range(n_folds)]
            concat_test_and_val = [np.concatenate([v, t]) for v, t in zip(val_folds, test_folds)]
            concat_test_and_val = [np.unique(a) for a in concat_test_and_val]
            concat_test_and_val = np.concatenate(concat_test_and_val)
            cond2 = (concat_test_and_val.size == (2 * n_test * n_folds))
            if cond1 and cond2:
                break
            else:
                attempts = attempts + 1000
        # Select split
        test_ids = test_folds[fold_id]
        val_ids = test_folds[(fold_id + 1) % n_folds]
        train_ids = [s for s in subject_ids if s not in np.concatenate([val_ids, test_ids])]
        # Sort
        train_ids = np.sort(train_ids).tolist()
        val_ids = np.sort(val_ids).tolist()
        test_ids = np.sort(test_ids).tolist()
        return train_ids, val_ids, test_ids

    def compute_global_std(self, subject_ids):
        # Memory-efficient method:
        # Var(x) = E(x ** 2) - (E(x) ** 2)
        total_samples = 0
        sum_x = 0.0
        sum_x2 = 0.0
        for subject_id in subject_ids:
            ind_dict = self.read_subject_data(subject_id)
            x = ind_dict[KEY_EEG]

            # Only sleep
            hypno = ind_dict[KEY_HYPNOGRAM]
            pages = np.concatenate([np.where(hypno == lbl)[0] for lbl in self.hypnogram_sleep_labels])
            hypnogram_page_size = int(np.round(self.hypnogram_page_duration * self.fs))
            x = utils.extract_pages(x, pages, hypnogram_page_size).flatten()

            outlier_thr = np.percentile(np.abs(x), 99)
            x = x[np.abs(x) <= outlier_thr]
            total_samples += x.shape[0]
            sum_x += np.sum(x)
            sum_x2 += np.sum(x ** 2)
        mean_squared_x = sum_x2 / total_samples
        mean_x = sum_x / total_samples
        global_variance = mean_squared_x - (mean_x ** 2)
        global_std = np.sqrt(global_variance)

        # Old method:
        # x_list = self.get_subset_signals(
        #     subject_id_list=subject_ids, normalize_clip=False)
        # tmp_list = []
        # for x in x_list:
        #     outlier_thr = np.percentile(np.abs(x), 99)
        #     tmp_signal = x[np.abs(x) <= outlier_thr]
        #     tmp_list.append(tmp_signal)
        # all_signals = np.concatenate(tmp_list)
        # global_std = all_signals.std()

        return global_std

    def read_subject_data(self, subject_id):
        return self.data[subject_id]

    def get_subject_signal(
            self,
            subject_id,
            normalize_clip=True,
            normalization_mode=None,
            which_expert=None,
            verbose=False
    ):
        if which_expert is None:
            which_expert = self.default_expert
        if normalization_mode is None:
            normalization_mode = self.default_page_subset

        checks.check_valid_value(subject_id, 'ID', self.all_ids)
        valid_experts = [(i + 1) for i in range(self.n_experts)]
        checks.check_valid_value(which_expert, 'which_expert', valid_experts)
        checks.check_valid_value(
            normalization_mode, 'normalization_mode',
            [constants.N2_RECORD, constants.WN_RECORD])

        ind_dict = self.read_subject_data(subject_id)

        # Unpack data
        signal = ind_dict[KEY_EEG]

        if normalize_clip:
            if normalization_mode == constants.WN_RECORD:
                if verbose:
                    print('Normalization with stats from '
                          'pages containing true events.')
                # Normalize using stats from pages with true events.
                marks = ind_dict['%s_%d' % (KEY_MARKS, which_expert)]
                # Transform stamps into sequence
                marks = utils.stamp2seq(marks, 0, signal.shape[0] - 1)
                tmp_pages = ind_dict[KEY_ALL_PAGES]
                activity = utils.extract_pages(marks, tmp_pages, self.page_size, border_size=0)
                activity = activity.sum(axis=1)
                activity = np.where(activity > 0)[0]
                tmp_pages = tmp_pages[activity]
                signal, _ = utils.norm_clip_signal(
                    signal, tmp_pages, self.page_size, clip_value=self.params[pkeys.CLIP_VALUE],
                    norm_computation=self.params[pkeys.NORM_COMPUTATION_MODE],
                    global_std=self.global_std)
            else:
                if verbose:
                    print('Normalization with stats from '
                          'N2 pages.')
                n2_pages = ind_dict[KEY_N2_PAGES]
                signal, _ = utils.norm_clip_signal(
                    signal, n2_pages, self.page_size, clip_value=self.params[pkeys.CLIP_VALUE],
                    norm_computation=self.params[pkeys.NORM_COMPUTATION_MODE],
                    global_std=self.global_std)
        return signal

    def get_subset_signals(
            self,
            subject_id_list,
            normalize_clip=True,
            normalization_mode=None,
            which_expert=None,
            verbose=False
    ):
        subset_signals = []
        for subject_id in subject_id_list:
            signal = self.get_subject_signal(
                subject_id,
                normalize_clip=normalize_clip,
                normalization_mode=normalization_mode,
                which_expert=which_expert,
                verbose=verbose)
            subset_signals.append(signal)
        return subset_signals

    def get_signals(
            self,
            normalize_clip=True,
            normalization_mode=None,
            which_expert=None,
            verbose=False
    ):
        subset_signals = self.get_subset_signals(
            self.all_ids,
            normalize_clip=normalize_clip,
            normalization_mode=normalization_mode,
            which_expert=which_expert,
            verbose=verbose)
        return subset_signals

    def get_ids(self):
        return self.all_ids

    def get_subject_pages(
            self,
            subject_id,
            pages_subset=None,
            verbose=False
    ):
        """Returns the indices of the pages of this subject."""
        if pages_subset is None:
            pages_subset = self.default_page_subset
        checks.check_valid_value(subject_id, 'ID', self.all_ids)
        checks.check_valid_value(
            pages_subset, 'pages_subset',
            [constants.N2_RECORD, constants.WN_RECORD])

        ind_dict = self.read_subject_data(subject_id)

        if pages_subset == constants.WN_RECORD:
            pages = ind_dict[KEY_ALL_PAGES]
        else:
            pages = ind_dict[KEY_N2_PAGES]

        if verbose:
            print('Getting ID %s, %d %s pages'
                  % (subject_id, pages.size, pages_subset))
        return pages

    def get_subset_pages(
            self,
            subject_id_list,
            pages_subset=None,
            verbose=False
    ):
        """Returns the list of pages from a list of subjects."""
        subset_pages = []
        for subject_id in subject_id_list:
            pages = self.get_subject_pages(
                subject_id,
                pages_subset=pages_subset,
                verbose=verbose)
            subset_pages.append(pages)
        return subset_pages

    def get_pages(
            self,
            pages_subset=None,
            verbose=False
    ):
        """Returns the list of pages from all subjects."""
        subset_pages = self.get_subset_pages(
            self.all_ids,
            pages_subset=pages_subset,
            verbose=verbose
        )
        return subset_pages

    def get_subject_stamps(
            self,
            subject_id,
            which_expert=None,
            pages_subset=None,
            verbose=False
    ):
        """Returns the sample-stamps of marks of this subject."""
        if which_expert is None:
            which_expert = self.default_expert
        if pages_subset is None:
            pages_subset = self.default_page_subset
        checks.check_valid_value(subject_id, 'ID', self.all_ids)
        valid_experts = [(i + 1) for i in range(self.n_experts)]
        checks.check_valid_value(which_expert, 'which_expert', valid_experts)
        checks.check_valid_value(
            pages_subset, 'pages_subset',
            [constants.N2_RECORD, constants.WN_RECORD])

        ind_dict = self.read_subject_data(subject_id)

        marks = ind_dict['%s_%d' % (KEY_MARKS, which_expert)]

        if pages_subset == constants.WN_RECORD:
            pages = ind_dict[KEY_ALL_PAGES]
        else:
            pages = ind_dict[KEY_N2_PAGES]

        # Get stamps that are inside selected pages
        marks = utils.extract_pages_for_stamps(marks, pages, self.page_size)

        if verbose:
            print('Getting ID %s, %s pages, %d stamps'
                  % (subject_id, pages_subset, marks.shape[0]))
        return marks

    def get_subset_stamps(
            self,
            subject_id_list,
            which_expert=None,
            pages_subset=None,
            verbose=False
    ):
        """Returns the list of stamps from a list of subjects."""
        subset_marks = []
        for subject_id in subject_id_list:
            marks = self.get_subject_stamps(
                subject_id,
                which_expert=which_expert,
                pages_subset=pages_subset,
                verbose=verbose)
            subset_marks.append(marks)
        return subset_marks

    def get_stamps(
            self,
            which_expert=None,
            pages_subset=None,
            verbose=False
    ):
        """Returns the list of stamps from all subjects."""
        subset_marks = self.get_subset_stamps(
            self.all_ids,
            which_expert=which_expert,
            pages_subset=pages_subset,
            verbose=verbose
        )
        return subset_marks

    def get_subject_hypnogram(
            self,
            subject_id,
            verbose=False
    ):
        """Returns the hypogram of this subject."""
        checks.check_valid_value(subject_id, 'ID', self.all_ids)

        ind_dict = self.read_subject_data(subject_id)

        hypno = ind_dict[KEY_HYPNOGRAM]

        if verbose:
            print('Getting Hypnogram of ID %s' % subject_id)
        return hypno

    def get_subset_hypnograms(
            self,
            subject_id_list,
            verbose=False
    ):
        """Returns the list of hypograms from a list of subjects."""
        subset_hypnos = []
        for subject_id in subject_id_list:
            hypno = self.get_subject_hypnogram(
                subject_id,
                verbose=verbose)
            subset_hypnos.append(hypno)
        return subset_hypnos

    def get_hypnograms(
            self,
            verbose=False
    ):
        """Returns the list of hypograms from all subjects."""
        subset_hypnos = self.get_subset_hypnograms(
            self.all_ids,
            verbose=verbose
        )
        return subset_hypnos

    def get_subject_data(
            self,
            subject_id,
            augmented_page=False,
            border_size=0,
            forced_mark_separation_size=0,
            which_expert=None,
            pages_subset=None,
            normalize_clip=True,
            normalization_mode=None,
            return_page_mask=False,
            verbose=False,
    ):
        """Returns segments of signal and marks from pages for the given id.

        Args:
            subject_id: (int) id of the subject of interest.
            augmented_page: (Optional, boolean, defaults to False) whether to
                augment the page with half page at each side.
            border_size: (Optional, int, defaults to 0) number of samples to be
                added at each border of the segments.
            forced_mark_separation_size: (Optional, int, defaults to 0) number
                of samples that are forced to exist between contiguous marks.
                If 0, no modification is performed.
            which_expert: (Optional, int, defaults to 1) Which expert
                annotations should be returned. It has to be consistent with
                the given n_experts, in a one-based counting.
            pages_subset: (Optional, string, [WN_RECORD, N2_RECORD]) If
                WN_RECORD (default), pages from the whole record. If N2_RECORD,
                only N2 pages are returned.
            normalize_clip: (Optional, boolean, defaults to True) If true,
                the signal is normalized and clipped from pages statistics.
            normalization_mode: (Optional, string, [WN_RECORD, N2_RECORD]) If
                WN_RECORD (default), statistics for normalization are
                computed from pages containing true events. If N2_RECORD,
                statistics are computed from N2 pages.
            return_page_mask: (Optional, boolean, defaults to False) If true,
                a binary mask will be returned indicating whether the samples are within
                the page subset.
            verbose: (Optional, boolean, defaults to False) Whether to print
                what is being read.

        Returns:
            signal: (2D array) each row is an (augmented) page of the signal
            marks: (2D array) each row is an (augmented) page of the marks
        Optional return:
            page_mask: (2D array) each row is an (augmented) page of a binary mask.
        """
        if which_expert is None:
            which_expert = self.default_expert
        if pages_subset is None:
            pages_subset = self.default_page_subset
        if normalization_mode is None:
            normalization_mode = self.default_page_subset

        checks.check_valid_value(subject_id, 'ID', self.all_ids)
        valid_experts = [(i+1) for i in range(self.n_experts)]
        checks.check_valid_value(which_expert, 'which_expert', valid_experts)
        checks.check_valid_value(
            pages_subset, 'pages_subset',
            [constants.N2_RECORD, constants.WN_RECORD])
        checks.check_valid_value(
            normalization_mode, 'normalization_mode',
            [constants.N2_RECORD, constants.WN_RECORD])

        ind_dict = self.read_subject_data(subject_id)

        # Unpack data
        signal = ind_dict[KEY_EEG]
        marks = ind_dict['%s_%d' % (KEY_MARKS, which_expert)]
        if pages_subset == constants.WN_RECORD:
            pages = ind_dict[KEY_ALL_PAGES]
        else:
            pages = ind_dict[KEY_N2_PAGES]

        # Transform stamps into sequence
        if forced_mark_separation_size > 0:
            print('Forcing separation of %d samples between marks' % forced_mark_separation_size)
            marks = utils.stamp2seq_with_separation(
                marks, 0, signal.shape[0] - 1, min_separation_samples=forced_mark_separation_size)
        else:
            marks = utils.stamp2seq(marks, 0, signal.shape[0] - 1)

        # Transform page subset into sequence
        pages = pages.astype(np.int32)
        pages_start = pages * self.page_size
        pages_end = (pages + 1) * self.page_size - 1
        pages_stamps = np.stack([pages_start, pages_end], axis=1).astype(np.int32)
        page_mask = utils.stamp2seq(pages_stamps, 0, signal.shape[0] - 1)

        # Compute border to be added
        if augmented_page:
            total_border = self.page_size // 2 + border_size
        else:
            total_border = border_size

        if normalize_clip:
            if normalization_mode == constants.WN_RECORD:
                if True:  # verbose:
                    print('Normalization with stats from pages containing true events.')
                # Normalize using stats from pages with true events.
                tmp_pages = ind_dict[KEY_ALL_PAGES]
                activity = utils.extract_pages(
                    marks, tmp_pages,
                    self.page_size, border_size=0)
                activity = activity.sum(axis=1)
                activity = np.where(activity > 0)[0]
                tmp_pages = tmp_pages[activity]
                signal, _ = utils.norm_clip_signal(
                    signal, tmp_pages, self.page_size,
                    norm_computation=self.params[pkeys.NORM_COMPUTATION_MODE],
                    global_std=self.global_std, clip_value=self.params[pkeys.CLIP_VALUE])
            else:
                if verbose:
                    print('Normalization with stats from N2 pages.')
                n2_pages = ind_dict[KEY_N2_PAGES]
                signal, _ = utils.norm_clip_signal(
                    signal, n2_pages, self.page_size,
                    norm_computation=self.params[pkeys.NORM_COMPUTATION_MODE],
                    global_std=self.global_std, clip_value=self.params[pkeys.CLIP_VALUE])

        # Extract segments
        signal = utils.extract_pages(
            signal, pages, self.page_size, border_size=total_border)
        marks = utils.extract_pages(
            marks, pages, self.page_size, border_size=total_border)
        page_mask = utils.extract_pages(
            page_mask, pages, self.page_size, border_size=total_border)

        # Set dtype
        signal = signal.astype(np.float32)
        marks = marks.astype(np.int8)
        page_mask = page_mask.astype(np.int8)

        if verbose:
            print('Getting ID %s, %d %s pages, Expert %d'
                  % (subject_id, pages.size, pages_subset, which_expert))
        if return_page_mask:
            return signal, marks, page_mask
        else:
            return signal, marks

    def get_subset_data(
            self,
            subject_id_list,
            augmented_page=False,
            border_size=0,
            forced_mark_separation_size=0,
            which_expert=None,
            pages_subset=None,
            normalize_clip=True,
            normalization_mode=None,
            return_page_mask=False,
            verbose=False,
    ):
        """Returns the list of signals and marks from a list of subjects.
        """
        subset_signals = []
        subset_marks = []
        subset_page_mask = []
        for subject_id in subject_id_list:
            signal, marks, page_mask = self.get_subject_data(
                subject_id,
                augmented_page=augmented_page,
                border_size=border_size,
                forced_mark_separation_size=forced_mark_separation_size,
                which_expert=which_expert,
                pages_subset=pages_subset,
                normalize_clip=normalize_clip,
                normalization_mode=normalization_mode,
                return_page_mask=True,
                verbose=verbose,
            )
            subset_signals.append(signal)
            subset_marks.append(marks)
            subset_page_mask.append(page_mask)
        if return_page_mask:
            return subset_signals, subset_marks, subset_page_mask
        else:
            return subset_signals, subset_marks

    def get_data(
            self,
            augmented_page=False,
            border_size=0,
            forced_mark_separation_size=0,
            which_expert=None,
            pages_subset=None,
            normalize_clip=True,
            normalization_mode=None,
            return_page_mask=False,
            verbose=False
    ):
        """Returns the list of signals and marks from all subjects.
        """
        subset_signals, subset_marks, subset_page_mask = self.get_subset_data(
            self.all_ids,
            augmented_page=augmented_page,
            border_size=border_size,
            forced_mark_separation_size=forced_mark_separation_size,
            which_expert=which_expert,
            pages_subset=pages_subset,
            normalize_clip=normalize_clip,
            normalization_mode=normalization_mode,
            return_page_mask=True,
            verbose=verbose
        )
        if return_page_mask:
            return subset_signals, subset_marks, subset_page_mask
        else:
            return subset_signals, subset_marks

    def get_sub_dataset(self, subject_id_list):
        """Data structure of a subset of subjects"""
        data_subset = {}
        for pat_id in subject_id_list:
            data_subset[pat_id] = self.data[pat_id].copy()
        return data_subset

    def save_checkpoint(self):
        """Saves a pickle file containing the loaded data."""
        os.makedirs(self.ckpt_dir, exist_ok=True)
        with open(self.ckpt_file, 'wb') as handle:
            pickle.dump(
                self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Checkpoint saved at %s' % self.ckpt_file)

    def _load_data(self, verbose):
        """Loads data either from a checkpoint or from scratch."""
        if self.load_checkpoint and self._exists_checkpoint():
            if verbose:
                print('Loading from checkpoint... ', flush=True, end='')
            data = self._load_from_checkpoint()
        else:
            if verbose:
                if self.load_checkpoint:
                    print("A checkpoint doesn't exist at %s."
                          " Loading from source instead." % self.ckpt_file)
                else:
                    print('Loading from source.')
            data = self._load_from_source()
        if verbose:
            print('Loaded')
        return data

    def _load_from_checkpoint(self):
        """Loads the pickle file containing the loaded data."""
        with open(self.ckpt_file, 'rb') as handle:
            data = pickle.load(handle)
        return data

    def _exists_checkpoint(self):
        """Checks whether the pickle file with the checkpoint exists."""
        return os.path.isfile(self.ckpt_file)

    def _load_from_source(self):
        """Loads and return the data from files and transforms it appropriately.
        This is just a template for the specific implementation of the dataset.
        the value of the key KEY_ID has to be an integer.

        Signal is an 1D array, pages are indices, marks are 2D sample-stamps.
        """
        # Data structure
        data = {}
        for pat_id in self.all_ids:
            pat_dict = {
                KEY_EEG: None,
                KEY_N2_PAGES: None,
                KEY_ALL_PAGES: None,
                KEY_HYPNOGRAM: None
            }
            for i in range(self.n_experts):
                pat_dict.update(
                    {'%s_%d' % (KEY_MARKS, i+1): None})
            data[pat_id] = pat_dict
        return data

