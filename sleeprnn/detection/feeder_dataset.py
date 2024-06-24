"""mass_ss.py: Defines the MASS class that manipulates the MASS database."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from sleeprnn.data.dataset import Dataset
from sleeprnn.data.dataset import KEY_EEG, KEY_MARKS, KEY_N2_PAGES, KEY_ALL_PAGES
from sleeprnn.data import utils
from sleeprnn.common import constants, checks


class FeederDataset(Dataset):

    def __init__(
        self,
        dataset: Dataset,
        sub_ids,
        task_mode=constants.N2_RECORD,
        which_expert=1,
        verbose=False,
        n2_subsampling_factor=1.0,
    ):
        """Constructor"""
        checks.check_valid_value(
            task_mode, "task_mode", [constants.WN_RECORD, constants.N2_RECORD]
        )

        self.parent_dataset = dataset
        self.parent_dataset_class = dataset.__class__
        self.task_mode = task_mode
        self.which_expert = which_expert
        self.n2_subsampling_factor = n2_subsampling_factor

        super(FeederDataset, self).__init__(
            dataset_dir=dataset.dataset_dir,
            load_checkpoint=False,
            dataset_name="%s_subset" % dataset.dataset_name,
            all_ids=sub_ids,
            event_name=dataset.event_name,
            hypnogram_sleep_labels=dataset.hypnogram_sleep_labels,
            hypnogram_page_duration=dataset.hypnogram_page_duration,
            n_experts=dataset.n_experts,
            default_expert=which_expert,
            default_page_subset=task_mode,
            params=dataset.params.copy(),
            verbose=verbose,
        )
        self.global_std = dataset.global_std

    def read_subject_data(self, subject_id):
        # Original data
        ind_dict = self.parent_dataset_class.read_subject_data(self, subject_id)
        # Return a subsample if required
        if self.n2_subsampling_factor < 1:
            n2_pages = ind_dict[KEY_N2_PAGES]

            # Random permutation with a fixed seed so that we can sample N2 pages from anywhere
            # print("Shuffling N2 pages before subsampling")
            n2_pages = np.random.RandomState(seed=0).permutation(n2_pages)

            n_pages_to_extract = int(
                np.ceil(self.n2_subsampling_factor * n2_pages.size)
            )
            ind_dict[KEY_N2_PAGES] = n2_pages[:n_pages_to_extract]  # contiguous segment
        return ind_dict

    def _load_from_source(self):
        """Loads the data from source."""
        data = self.parent_dataset.get_sub_dataset(self.all_ids)
        self.parent_dataset = None
        return data

    def get_data_for_training(
        self,
        border_size=0,
        forced_mark_separation_size=0,
        return_page_mask=False,
        verbose=False,
    ):
        output = super().get_data(
            augmented_page=True,
            border_size=border_size,
            forced_mark_separation_size=forced_mark_separation_size,
            which_expert=self.which_expert,
            pages_subset=self.task_mode,
            normalize_clip=True,
            normalization_mode=self.task_mode,
            return_page_mask=return_page_mask,
            verbose=verbose,
        )
        return output

    def get_data_for_prediction(
        self,
        border_size=0,
        predict_with_augmented_page=True,
        return_page_mask=False,
        verbose=False,
    ):
        output = super().get_data(
            augmented_page=predict_with_augmented_page,
            border_size=border_size,
            which_expert=self.which_expert,
            pages_subset=constants.WN_RECORD,
            normalize_clip=True,
            normalization_mode=self.task_mode,
            return_page_mask=return_page_mask,
            verbose=verbose,
        )
        return output

    def get_data_for_stats(self, border_size=0, verbose=False):
        subset_signals = []
        for subject_id in self.all_ids:
            signal = self.get_subject_data_for_stats(
                subject_id=subject_id, border_size=border_size, verbose=verbose
            )
            subset_signals.append(signal)
        return subset_signals

    def get_subject_data_for_stats(self, subject_id, border_size=0, verbose=False):
        checks.check_valid_value(subject_id, "ID", self.all_ids)
        ind_dict = self.read_subject_data(subject_id)

        # Unpack data
        signal = ind_dict[KEY_EEG]
        marks = ind_dict["%s_%d" % (KEY_MARKS, self.which_expert)]
        # Transform stamps into sequence
        marks = utils.stamp2seq(marks, 0, signal.shape[0] - 1)

        if self.task_mode == constants.WN_RECORD:
            if verbose:
                print("Stats from pages containing true events.")
            # Normalize using stats from pages with true events.
            stat_pages = ind_dict[KEY_ALL_PAGES]
            activity = utils.extract_pages(
                marks, stat_pages, self.page_size, border_size=0
            )
            activity = activity.sum(axis=1)
            activity = np.where(activity > 0)[0]
            stat_pages = stat_pages[activity]
        else:
            if verbose:
                print("Stats from N2 pages.")
            stat_pages = ind_dict[KEY_N2_PAGES]
        signal, _ = utils.norm_clip_signal(signal, stat_pages, self.page_size)
        # Extract segments
        signal = utils.extract_pages(
            signal, stat_pages, self.page_size, border_size=border_size
        )
        return signal
