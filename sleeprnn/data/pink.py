"""@Author: Nicolas I. Tapia-Rivas"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
from scipy.interpolate import interp1d

from sleeprnn.common import constants
from sleeprnn.data.dataset import Dataset
from sleeprnn.data.dataset import KEY_EEG, KEY_MARKS
from sleeprnn.data.dataset import KEY_N2_PAGES, KEY_ALL_PAGES, KEY_HYPNOGRAM
from sleeprnn.data import utils

PATH_PINK_RELATIVE = 'pink'


class Pink(Dataset):
    def __init__(self, params=None, load_checkpoint=False, verbose=True, **kwargs):
        self.channel = 'artificial'
        self.n_signals = 25
        self.n2_id = '2'
        self.unknown_id = '?'
        # Generation parameters
        self.signal_duration = 3600 + 2 * 20  # 1 hour of useful signal + 1 page at borders
        self.power_matching_highcut = 8  # [Hz]
        self.power_matching_target_value = 0.7286483227138594
        self.spectrum_profile_fn = self._get_profile_fn()

        all_ids = np.arange(1, self.n_signals + 1).tolist()
        super(Pink, self).__init__(
            dataset_dir=PATH_PINK_RELATIVE,
            load_checkpoint=load_checkpoint,
            dataset_name=constants.PINK_NAME,
            all_ids=all_ids,
            event_name='none',
            hypnogram_sleep_labels=['2'],
            hypnogram_page_duration=[20],
            params=params,
            verbose=verbose
        )
        self.global_std = None
        if verbose:
            print("Global STD", self.global_std)

        self.filt_signal_dict = {}
        self.exists_cache = False

    def _load_from_source(self):
        n_pages = self.signal_duration // self.page_duration
        data = {}
        start = time.time()
        for i, subject_id in enumerate(self.all_ids):
            print("\nGenerating pink noise ID %s" % subject_id)
            signal = self._generate_signal(subject_id)
            hypnogram = [self.unknown_id] + (n_pages - 2) * [self.n2_id] + [self.unknown_id]
            hypnogram = np.asarray(hypnogram)
            n2_pages = np.where(hypnogram == self.n2_id)[0].astype(np.int16)
            all_pages = np.arange(1, n_pages - 1, dtype=np.int16)
            marks = np.zeros(shape=(0, 2)).astype(np.int32)
            print('N2 pages: %d' % n2_pages.shape[0])
            print('Whole-night pages: %d' % all_pages.shape[0])
            print('Marks SS from E1: %d' % marks.shape[0])
            # Save data
            ind_dict = {
                KEY_EEG: signal,
                KEY_N2_PAGES: n2_pages,
                KEY_ALL_PAGES: all_pages,
                KEY_HYPNOGRAM: hypnogram,
                '%s_1' % KEY_MARKS: marks
            }
            data[subject_id] = ind_dict
            print('Loaded ID %d (%02d/%02d ready). Time elapsed: %1.4f [s]'
                  % (subject_id, i + 1, self.n_signals, time.time() - start))
        print('%d records have been read.' % len(data))
        return data

    def _get_profile_fn(self):
        pink_profile = np.load(os.path.join(
            utils.PATH_DATA, PATH_PINK_RELATIVE, "pink_profile.npy"
        ))
        profile_fn = interp1d(pink_profile[0], pink_profile[1])
        return profile_fn

    def _generate_signal(self, seed):
        # Base noise
        n_samples = int(self.signal_duration * self.fs)
        x = np.random.RandomState(seed=seed).normal(size=n_samples)
        # Scale the FFT spectrum
        y = np.fft.rfft(x)
        freq_gen = np.fft.rfftfreq(x.size, d=1. / self.fs)
        scaling = self.spectrum_profile_fn(freq_gen)
        y = y * scaling
        # Return to time domain and normalize
        x = np.fft.irfft(y)
        x = x - x.mean()
        x = x / x.std()
        # Filter to desired band
        x = utils.broad_filter(x, self.fs)
        # Scale to target amplitude
        f, p = utils.power_spectrum_by_sliding_window(x, self.fs)
        power_in_band = p[f <= self.power_matching_highcut].mean()
        correction_factor = self.power_matching_target_value / power_in_band
        x = x * correction_factor
        # Cast to desired type
        x = x.astype(np.float32)
        return x

    def create_signal_cache(self, highcut=4):
        signals = self.get_signals(normalize_clip=False)
        for k, sub_id in enumerate(self.all_ids):
            filt_signal = utils.filter_iir_lowpass(signals[k], self.fs, highcut=highcut)
            filt_signal = filt_signal.astype(np.float32)
            self.filt_signal_dict[sub_id] = filt_signal
        self.exists_cache = True

    def delete_signal_cache(self):
        self.filt_signal_dict = {}
        self.exists_cache = False

    def get_subject_filt_signal(self, subject_id):
        if self.exists_cache:
            signal = self.filt_signal_dict[subject_id]
        else:
            signal = None
        return signal

    def get_subset_filt_signals(self, subject_id_list):
        if self.exists_cache:
            subset_signals = [
                self.get_subject_filt_signal(sub_id)
                for sub_id in subject_id_list]
        else:
            subset_signals = None
        return subset_signals

    def get_filt_signals(self):
        if self.exists_cache:
            subset_signals = [
                self.get_subject_filt_signal(sub_id)
                for sub_id in self.all_ids]
        else:
            subset_signals = None
        return subset_signals

    def exists_filt_signal_cache(self):
        return self.exists_cache
