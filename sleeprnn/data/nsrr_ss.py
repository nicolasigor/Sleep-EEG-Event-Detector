from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import pickle

import numpy as np
import pandas as pd

from sleeprnn.common import constants
from sleeprnn.data import utils
from sleeprnn.data import stamp_correction
from sleeprnn.data.dataset import Dataset
from sleeprnn.data.dataset import KEY_EEG, KEY_N2_PAGES, KEY_ALL_PAGES, KEY_MARKS, KEY_HYPNOGRAM

PATH_NSRR_RELATIVE = 'nsrr'
PATH_REC_AND_STATE = 'register_and_state'
SUBDATASETS = ['shhs1', 'mros1', 'chat1', 'sof', 'cfs', 'ccshs']

KEY_AGE = 'age'
KEY_SEX = 'sex'


class NsrrSS(Dataset):
    """This is a class to manipulate the NSRR data EEG dataset.
    """

    def __init__(self, params=None, load_checkpoint=False, verbose=True, **kwargs):
        """Constructor"""
        # NSRR parameters
        self.state_ids = np.array([
            'Wake|0',
            'Stage 1 sleep|1',
            'Stage 2 sleep|2',
            'Stage 3 sleep|3',
            'Stage 4 sleep|4',
            'REM sleep|5',
            'Movement|6',
            'Unscored|9'])
        self.unknown_id = 'Unscored|9'  # Character for unknown state in hypnogram
        self.n2_id = 'Stage 2 sleep|2'  # Character for N2 identification in hypnogram
        self.original_page_duration = 30  # Time of window page [s]

        all_ids = [1, 2, 3]  # Dummy, will be established after processing data
        all_ids.sort()

        hypnogram_sleep_labels = np.array([
            'Stage 1 sleep|1',
            'Stage 2 sleep|2',
            'Stage 3 sleep|3',
            'Stage 4 sleep|4',
            'REM sleep|5'])
        hypnogram_page_duration = 30

        super(NsrrSS, self).__init__(
            dataset_dir=PATH_NSRR_RELATIVE,
            load_checkpoint=load_checkpoint,
            dataset_name=constants.NSRR_SS_NAME,
            all_ids=all_ids,
            event_name=constants.SPINDLE,
            hypnogram_sleep_labels=hypnogram_sleep_labels,
            hypnogram_page_duration=hypnogram_page_duration,
            n_experts=1,  # Dummy
            params=params,
            verbose=verbose
        )
        self.global_std = None
        if verbose:
            print('Global STD:', self.global_std)
        if verbose:
            print('Dataset %s with %d patients.' % (self.dataset_name, len(self.all_ids)))

    def _load_from_source(self):
        """Loads the data from files and transforms it appropriately."""
        data_paths = self._get_file_paths()
        data = {}
        save_dir = os.path.join(self.dataset_dir, 'pretty_files')
        os.makedirs(save_dir, exist_ok=True)
        start = time.time()
        global_count = 0
        for i_dataset, subdataset in enumerate(data_paths.keys()):
            print("\nLoading subdataset %s" % subdataset)
            meta_df = pd.read_csv(data_paths[subdataset]['metadata'])
            meta_dict = meta_df.set_index('subject_id').to_dict(orient='index')

            subject_paths = data_paths[subdataset]['eeg_and_state']
            subject_ids = list(subject_paths.keys())
            for i_subject, subject_id in enumerate(subject_ids):
                print('\nLoading ID %s' % subject_id)

                subject_eeg_state_file = subject_paths[subject_id]
                subject_age = meta_dict[subject_id]['age']
                subject_sex = meta_dict[subject_id]['sex']

                signal, hypnogram_original = self._read_npz(subject_eeg_state_file)

                marks = np.zeros((0, 2), dtype=np.int32)  # Dummy

                n2_pages_original = np.where(hypnogram_original == self.n2_id)[0]

                total_pages = int(signal.size / self.page_size)
                all_pages = np.arange(1, total_pages - 1, dtype=np.int16)

                print('N2 pages: %d' % n2_pages_original.shape[0])
                print('Whole-night pages: %d' % all_pages.shape[0])
                print('Marks SS : %d' % marks.shape[0])
                print("Age:", subject_age, "Sex:", subject_sex)

                # Save data
                ind_dict = {
                    KEY_EEG: signal.astype(np.float32),
                    KEY_N2_PAGES: n2_pages_original.astype(np.int16),
                    KEY_ALL_PAGES: all_pages.astype(np.int16),
                    KEY_HYPNOGRAM: hypnogram_original,
                    '%s_1' % KEY_MARKS: marks.astype(np.int32),
                    KEY_AGE: subject_age,
                    KEY_SEX: subject_sex,
                }

                # Save data to disk and only save in object the path
                fname = os.path.join(save_dir, 'subject_%s.npz' % subject_id)
                np.savez(fname, **ind_dict)

                data[subject_id] = {'pretty_file_path': fname}

                print('Loaded ID %s (%02d/%02d ready). Time elapsed: %1.4f [s]' % (
                    subject_id, i_subject+1, len(subject_ids), time.time()-start))
                global_count += 1
        print('%d records have been read.' % global_count)
        return data

    def read_subject_data(self, subject_id, exclusion_of_pages=True):
        path_dict = self.data[subject_id]
        ind_dict = np.load(path_dict['pretty_file_path'])

        loaded_ind_dict = {}
        for key in ind_dict.files:
            loaded_ind_dict[key] = ind_dict[key]

        n2_pages = loaded_ind_dict[KEY_N2_PAGES]
        if n2_pages.size == 0 or (not exclusion_of_pages):
            return loaded_ind_dict

        # ################
        # Exclude some N2 pages
        mass_amplitude_criterion = 200  # uV
        moda_standard_deviation_min = 5.0895143
        moda_standard_deviation_max = 37.46395
        moda_power_law_scale_min = 1.2120386356729853
        moda_power_law_scale_max = 34.88810450777444
        moda_power_law_exponent_min = -1.8909820062793914
        moda_power_law_exponent_max = -0.6304845212942357
        moda_power_law_max_ratio_max = 8.041562207454062
        moda_power_law_r2_min = 0.6995619824694014

        # Page wise signals
        signal = loaded_ind_dict[KEY_EEG]
        sub_signal = signal.reshape(-1, self.fs * self.original_page_duration)[n2_pages]  # [n_pages, n_samples]

        # Amplitude criteria
        pages_amplitude = np.max(np.abs(sub_signal), axis=1)  # (n_pages,)
        valid_1 = (pages_amplitude <= mass_amplitude_criterion).astype(np.int32)  # (n_pages,)

        # Standard dev
        pages_std = sub_signal.std(axis=1)  # (n_pages,)
        valid_2 = (pages_std >= moda_standard_deviation_min) * (pages_std <= moda_standard_deviation_max)
        valid_2 = valid_2.astype(np.int32)  # (n_pages,)

        # Spectrum
        freq, pages_spectrum = utils.compute_pagewise_fft(sub_signal, self.fs, window_duration=2)
        pages_scales, pages_exponents = utils.compute_pagewise_powerlaw(
            freq, pages_spectrum, broad_band=(2, 30), sigma_band=(10, 17))  # (n_pages,)

        valid_3 = (pages_scales >= moda_power_law_scale_min) * (pages_scales <= moda_power_law_scale_max)
        valid_3 = valid_3.astype(np.int32)  # (n_pages,)

        valid_4 = (pages_exponents >= moda_power_law_exponent_min) * (pages_exponents <= moda_power_law_exponent_max)
        valid_4 = valid_4.astype(np.int32)  # (n_pages,)

        # Deviation from power law fit: Max ratio
        f_min = 2
        f_max = 30
        valid_locs = np.where((freq >= f_min) & (freq <= f_max))[0]
        dev_f = freq[valid_locs]
        dev_x = pages_spectrum[:, valid_locs]
        dev_x_law = [fit_s * (dev_f ** fit_e) for fit_s, fit_e in zip(pages_scales, pages_exponents)]
        dev_x_law = np.stack(dev_x_law, axis=0)
        ratio = dev_x / dev_x_law  # n_pages, n_freqs
        max_ratio = np.max(ratio, axis=1)  # to detect weird peaks, shape (n_pages,)

        valid_5 = (max_ratio <= moda_power_law_max_ratio_max).astype(np.int32)  # (n_pages,)

        # Deviation from power law fit: Coefficient R2
        # for r2, we remove sigma
        valid_locs = np.where((dev_f < 10) | (dev_f > 17))[0]
        log_dev_x = np.log(dev_x[:, valid_locs])
        log_dev_x_law = np.log(dev_x_law[:, valid_locs])
        squared_data = np.sum((log_dev_x - log_dev_x.mean(axis=1).reshape(-1, 1)) ** 2, axis=1)
        squared_residuals = np.sum((log_dev_x - log_dev_x_law) ** 2, axis=1)
        r2 = 1 - squared_residuals / squared_data  # (n_pages,)

        valid_6 = (r2 >= moda_power_law_r2_min).astype(np.int32)  # (n_pages,)

        # All validations must occur
        valid = valid_1 * valid_2 * valid_3 * valid_4 * valid_5 * valid_6
        weird_pages = np.array([
            n2_pages[i]
            for i in range(n2_pages.size)
            if valid[i] == 0
        ], dtype=np.int32)

        # Overwrite labels
        hypnogram = loaded_ind_dict[KEY_HYPNOGRAM]
        hypnogram[weird_pages] = self.unknown_id
        # first and last pages cannot be used otherwise context is lacking (for quality detection and filtering stuff)
        hypnogram[0] = self.unknown_id
        hypnogram[-1] = self.unknown_id

        n2_pages = np.where(hypnogram == self.n2_id)[0]
        # Save new data
        loaded_ind_dict[KEY_HYPNOGRAM] = hypnogram
        loaded_ind_dict[KEY_N2_PAGES] = n2_pages
        # ################

        return loaded_ind_dict

    def _get_file_paths(self):
        """Returns a list of dicts containing paths to load the database."""
        # Build list of paths
        data_paths = {}
        all_ids = []
        for subdataset in SUBDATASETS:
            data_dir = os.path.join(self.dataset_dir, subdataset)
            eeg_dir = os.path.join(data_dir, PATH_REC_AND_STATE)
            meta_file = [f for f in os.listdir(data_dir) if 'metadata.csv' in f][0]
            meta_file = os.path.join(data_dir, meta_file)
            subject_ids = np.array(
                [".".join(f.split(".")[:-1]) for f in os.listdir(eeg_dir) if 'npz' in f], dtype='<U40')
            # Only keep those subject ids that intersect with metafile subject ids
            subject_ids_meta = pd.read_csv(meta_file)['subject_id'].values
            subject_ids_common = list(set.intersection(set(subject_ids), set(subject_ids_meta)))
            subdataset_paths = {
                'metadata': meta_file,
                'eeg_and_state': {
                    s: os.path.join(eeg_dir, '%s.npz' % s)
                    for s in subject_ids_common}
            }
            data_paths[subdataset] = subdataset_paths
            # Collect IDs
            all_ids.append(np.array(subject_ids_common, dtype='<U40'))
        all_ids = np.sort(np.concatenate(all_ids))
        # Replace all_ids dummy
        self.all_ids = all_ids

        print('%d records in %s dataset.' % (len(self.all_ids), self.dataset_name))
        return data_paths

    def _load_from_checkpoint(self):
        """Loads the pickle file containing the loaded data."""
        with open(self.ckpt_file, 'rb') as handle:
            data = pickle.load(handle)
        all_ids = list(data.keys())
        all_ids = np.sort(np.array(all_ids, dtype='<U40'))
        # Replace all_ids dummy
        self.all_ids = all_ids
        return data

    def _read_npz(self, path_eeg_state_file):
        # Keys in npz:
        #     'dataset'
        #     'subject_id'
        #     'channel'
        #     'signal'
        #     'sampling_rate'
        #     'hypnogram'
        #     'epoch_duration'
        #     'bandpass_filter'
        #     'resampling_function'
        #     'original_sampling_rate'

        data = np.load(path_eeg_state_file)
        signal = data['signal']
        hypnogram = data['hypnogram']
        # Filter signal
        original_fs = data['sampling_rate']
        signal = utils.broad_filter_moda(signal, original_fs)
        # Now resample to the required frequency
        if self.fs != original_fs:
            print('Resampling from %d Hz to required %d Hz' % (original_fs, self.fs))
            signal = utils.resample_signal(signal, fs_old=original_fs, fs_new=self.fs)
        signal = signal.astype(np.float32)
        return signal, hypnogram
