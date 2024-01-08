"""mass_ss.py: Defines the MASS class that manipulates the MASS database."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from src.data.dataset import Dataset
from src.data.dataset import KEY_EEG, KEY_MARKS, KEY_N2_PAGES, KEY_ALL_PAGES
from src.helpers.reader import load_dataset
from src.common import constants, pkeys
from src.detection.feeder_dataset import FeederDataset
from src.detection.postprocessor import PostProcessor
from src.detection import postprocessing
from src.detection import det_utils


class PredictedDataset(Dataset):

    def __init__(
            self,
            dataset: FeederDataset,
            probabilities_dict,
            params=None,
            verbose=False,
            skip_setting_threshold=False
    ):
        # make the changes local
        params = {} if (params is None) else params.copy()

        self.parent_dataset = dataset
        self.task_mode = dataset.task_mode
        self.probabilities_dict = probabilities_dict
        self.postprocessor = PostProcessor(event_name=dataset.event_name, params=params)
        self.probability_threshold = None

        """Constructor"""
        super(PredictedDataset, self).__init__(
            dataset_dir=dataset.dataset_dir,
            load_checkpoint=False,
            dataset_name='%s_predicted' % dataset.dataset_name,
            all_ids=dataset.all_ids,
            event_name=dataset.event_name,
            hypnogram_sleep_labels=dataset.hypnogram_sleep_labels,
            hypnogram_page_duration=dataset.hypnogram_page_duration,
            default_expert=1,
            default_page_subset=dataset.task_mode,
            n_experts=1,
            params=dataset.params.copy(),
            verbose=verbose,
        )
        self.global_std = dataset.global_std
        # Check that subject ids in probabilities are the same as the ones
        # on the dataset
        ids_proba = list(self.probabilities_dict.keys())
        ids_data = list(dataset.all_ids)
        ids_proba.sort()
        ids_data.sort()
        if ids_data != ids_proba:
            raise ValueError(
                'IDs mismatch: IDs from predictions are %s '
                'but IDs from given dataset are %s' % (ids_proba, ids_data))
        if not skip_setting_threshold:
            self.set_probability_threshold(0.5)

    def _load_from_source(self):
        """Loads the data from source."""
        # Extract only necessary stuff
        data = {}
        for sub_id in self.all_ids:
            ind_dict = self.parent_dataset.read_subject_data(sub_id)
            pat_dict = {
                KEY_EEG: None,
                KEY_N2_PAGES: ind_dict[KEY_N2_PAGES],
                KEY_ALL_PAGES: ind_dict[KEY_ALL_PAGES],
                '%s_%d' % (KEY_MARKS, 1): None
            }
            data[sub_id] = pat_dict
        self.parent_dataset = None
        return data

    def set_probability_threshold(self, new_probability_threshold, adjusted_by_threshold=None, verbose=False):
        """Sets a new probability threshold and updates the stamps accordingly.

        If adjusted_by_threshold (float between 0 and 1) is set, then the given
        new probability threshold is treated as a threshold for probabilities ADJUSTED by the
        given value, i.e., probabilities that satisfy:

        adjusted_proba > 0.5 <=> predicted_proba > adjusted_by_threshold

        Therefore, the value of new_probability_threshold is first transformed to its equivalent
        in the predicted probabilities domain, and then it is applied to them.

        If adjusted_by_threshold is None, then the given new probability threshold is used directly
        on the predicted probabilities.
        """
        if adjusted_by_threshold is not None:
            new_probability_threshold = det_utils.transform_thr_for_adjusted_to_thr_for_predicted(
                new_probability_threshold, adjusted_by_threshold)
            print("New threshold: %1.8f" % new_probability_threshold) if verbose else None
        self.probability_threshold = new_probability_threshold
        self._update_stamps()

    def _update_stamps(self):
        probabilities_list = []
        for sub_id in self.all_ids:
            # print("debug: adjusting proba")
            sub_proba = self.get_subject_probabilities(sub_id, return_adjusted=True)
            probabilities_list.append(sub_proba)

        if self.task_mode == constants.N2_RECORD:
            # Keep only N2 stamps
            n2_pages_val = self.get_pages(
                pages_subset=constants.N2_RECORD)
        else:
            n2_pages_val = None

        stamps_list = self.postprocessor.proba2stamps_with_list(
            probabilities_list,
            pages_indices_subset_list=n2_pages_val,
            thr=0.5)  # thr is 0.5 because probas are adjusted

        # KC postprocessing
        if self.event_name == constants.KCOMPLEX:
            signals = self.get_signals_external()
            new_stamps_list = []
            for k, sub_id in enumerate(self.all_ids):
                # Load signal
                signal = signals[k]
                stamps = stamps_list[k]
                stamps = postprocessing.kcomplex_stamp_split(
                    signal, stamps, self.fs,
                    signal_is_filtered=True)
                new_stamps_list.append(stamps)
            stamps_list = new_stamps_list

        # NSRR Amplitude removal
        if 'nsrr' in self.parent_dataset.dataset_name:
            max_amplitude = 134.12087769782073  # uV, from MODA spindles
            new_stamps_list = []
            for k, sub_id in enumerate(self.all_ids):
                # Load signal
                sub_data = self.parent_dataset.read_subject_data(sub_id, exclusion_of_pages=False)
                signal = sub_data['signal']
                stamps = stamps_list[k]
                if stamps.size > 0:
                    stamps, no_peaks_found = postprocessing.spindle_amplitude_filtering(signal, stamps, self.fs, max_amplitude)
                    if no_peaks_found:
                        print("Found error 'no peaks found' in subject %s" % sub_id)
                new_stamps_list.append(stamps)
            stamps_list = new_stamps_list

        # Now save model stamps
        stamp_key = '%s_%d' % (KEY_MARKS, 1)
        for k, sub_id in enumerate(self.all_ids):
            self.data[sub_id][stamp_key] = stamps_list[k]

    def get_subject_stamps_probabilities(
            self,
            subject_id,
            pages_subset=None,
            return_adjusted=True,
            proba_prc=75,
    ):
        subject_stamps = self.get_subject_stamps(subject_id, pages_subset=pages_subset)
        subject_proba = self.get_subject_probabilities(subject_id, return_adjusted=return_adjusted)
        subject_stamp_proba = det_utils.get_event_probabilities(
            subject_stamps, subject_proba,
            downsampling_factor=self.params[pkeys.TOTAL_DOWNSAMPLING_FACTOR], proba_prc=proba_prc)
        return subject_stamp_proba

    def get_subset_stamps_probabilities(
            self,
            subject_ids,
            pages_subset=None,
            return_adjusted=True,
            proba_prc=75,
    ):
        stamp_proba_list = []
        for sub_id in subject_ids:
            stamp_proba_list.append(self.get_subject_stamps_probabilities(
                sub_id, pages_subset=pages_subset, return_adjusted=return_adjusted, proba_prc=proba_prc))
        return stamp_proba_list

    def get_stamps_probabilities(
            self,
            pages_subset=None,
            return_adjusted=True,
            proba_prc=75,
    ):
        stamp_proba_list = self.get_subset_stamps_probabilities(
            self.all_ids, pages_subset=pages_subset, return_adjusted=return_adjusted, proba_prc=proba_prc)
        return stamp_proba_list

    def get_subject_probabilities(self, subject_id, return_adjusted=False):
        """ Returns the subject's predicted probability vector.

        If return_adjusted is False (default), the predicted probabilities are returned.
        If return_adjusted is True, the predicted probabilities are first ADJUSTED by the
        set probability threshold, i.e., they are transformed to probabilities that satisfy:

        adjusted_proba > 0.5 <=> predicted_proba > probability_threshold

        after the adjustment, the adjusted probabilities are returned.
        """
        subject_probabilities = self.probabilities_dict[subject_id].copy()
        if return_adjusted:
            subject_probabilities = det_utils.transform_predicted_proba_to_adjusted_proba(
                subject_probabilities, self.probability_threshold)
        return subject_probabilities

    def get_subset_probabilities(self, subject_ids, return_adjusted=False):
        proba_list = []
        for sub_id in subject_ids:
            proba_list.append(self.get_subject_probabilities(sub_id, return_adjusted=return_adjusted))
        return proba_list

    def get_probabilities(self, return_adjusted=False):
        return self.get_subset_probabilities(self.all_ids, return_adjusted=return_adjusted)

    def set_parent_dataset(self, dataset):
        self.parent_dataset = dataset

    def delete_parent_dataset(self):
        self.parent_dataset = None

    def get_signals_external(self):
        if self.parent_dataset is None:
            tmp_name = self.dataset_name
            parent_dataset_name = "_".join(tmp_name.split("_")[:2])
            parent_dataset = load_dataset(
                parent_dataset_name, params=self.params,
                load_checkpoint=True, verbose=False)
        else:
            parent_dataset = self.parent_dataset
        if not parent_dataset.exists_filt_signal_cache():
            print('Creating cache that does not exist')
            parent_dataset.create_signal_cache()
        signals = parent_dataset.get_subset_filt_signals(self.all_ids)
        return signals
