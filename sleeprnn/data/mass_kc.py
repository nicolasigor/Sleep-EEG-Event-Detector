"""mass_kc.py: Defines the MASS class that manipulates the MASS database."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import pyedflib

from sleeprnn.common import constants
from sleeprnn.data import utils
from sleeprnn.data import stamp_correction
from sleeprnn.data.dataset import Dataset
from sleeprnn.data.dataset import KEY_EEG, KEY_MARKS
from sleeprnn.data.dataset import KEY_N2_PAGES, KEY_ALL_PAGES, KEY_HYPNOGRAM

PATH_MASS_RELATIVE = "mass"
PATH_REC = "register"
PATH_MARKS = os.path.join("label", "kcomplex")
PATH_STATES = os.path.join("label", "state")

KEY_FILE_EEG = "file_eeg"
KEY_FILE_STATES = "file_states"
KEY_FILE_MARKS = "file_marks"

IDS_INVALID = [4, 8, 15, 16]
IDS_TEST = [2, 6, 12, 13]
# IDS_INVALID = []
# IDS_TEST = [2, 6, 12, 13, 4, 8, 15, 16]


class MassKC(Dataset):
    """This is a class to manipulate the MASS data EEG dataset.
    For K-complex events

    Expected directory tree inside DATA folder (see utils.py):

    PATH_MASS_RELATIVE
    |__ PATH_REC
        |__ 01-02-0001 PSG.edf
        |__ 01-02-0002 PSG.edf
        |__ ...
    |__ PATH_STATES
        |__ 01-02-0001 Base.edf
        |__ 01-02-0002 Base.edf
        |__ ...
    |__ PATH_MARKS
        |__ 01-02-0001 KComplexesE1.edf
        |__ 01-02-0002 KComplexesE1.edf
        |__ ...
    """

    def __init__(self, params=None, load_checkpoint=False, verbose=True, **kwargs):
        """Constructor"""
        # MASS parameters
        self.channel = "EEG C3-CLE"  # Channel for SS marks
        # In MASS, we need to index by name since not all the lists are
        # sorted equally

        # Hypnogram parameters
        self.state_ids = np.array(["1", "2", "3", "4", "R", "W", "?"])
        self.unknown_id = "?"  # Character for unknown state in hypnogram
        self.n2_id = "2"  # Character for N2 identification in hypnogram

        # KComplex characteristics
        self.min_kc_duration = 0.2  # Minimum duration of KC in seconds

        valid_ids = [i for i in range(1, 20) if i not in IDS_INVALID]
        self.test_ids = IDS_TEST
        self.train_ids = [i for i in valid_ids if i not in self.test_ids]

        if verbose:
            print(
                "Train size: %d. Test size: %d"
                % (len(self.train_ids), len(self.test_ids))
            )
            print("Train subjects: \n", self.train_ids)
            print("Test subjects: \n", self.test_ids)

        super(MassKC, self).__init__(
            dataset_dir=PATH_MASS_RELATIVE,
            load_checkpoint=load_checkpoint,
            dataset_name=constants.MASS_KC_NAME,
            all_ids=self.train_ids + self.test_ids,
            event_name=constants.KCOMPLEX,
            hypnogram_sleep_labels=["1", "2", "3", "4", "R"],
            hypnogram_page_duration=20,
            params=params,
            verbose=verbose,
        )

        self.global_std = self.compute_global_std(self.train_ids)
        if verbose:
            print("Global STD:", self.global_std)

        self.filt_signal_dict = {}
        self.exists_cache = False

    def _load_from_source(self):
        """Loads the data from files and transforms it appropriately."""
        data_paths = self._get_file_paths()
        data = {}
        n_data = len(data_paths)
        start = time.time()
        for i, subject_id in enumerate(data_paths.keys()):
            print("\nLoading ID %d" % subject_id)
            path_dict = data_paths[subject_id]

            # Read data
            signal = self._read_eeg(path_dict[KEY_FILE_EEG])
            hypnogram, start_sample = self._read_states_raw(path_dict[KEY_FILE_STATES])
            signal, hypnogram, end_sample = self._fix_signal_and_states(
                signal, hypnogram, start_sample
            )
            all_pages, n2_pages = self._hypnogram_selections(hypnogram)
            marks_1 = self._read_marks(path_dict["%s_1" % KEY_FILE_MARKS])
            marks_1 = self._fix_marks(marks_1, start_sample, end_sample)
            print("N2 pages: %d" % n2_pages.shape[0])
            print("Whole-night pages: %d" % all_pages.shape[0])
            print("Hypnogram pages: %d" % hypnogram.shape[0])
            print("Marks KC from E1: %d" % marks_1.shape[0])

            # Save data
            ind_dict = {
                KEY_EEG: signal,
                KEY_N2_PAGES: n2_pages,
                KEY_ALL_PAGES: all_pages,
                "%s_1" % KEY_MARKS: marks_1,
                KEY_HYPNOGRAM: hypnogram,
            }
            data[subject_id] = ind_dict
            print(
                "Loaded ID %d (%02d/%02d ready). Time elapsed: %1.4f [s]"
                % (subject_id, i + 1, n_data, time.time() - start)
            )
        print("%d records have been read." % len(data))
        return data

    def _get_file_paths(self):
        """Returns a list of dicts containing paths to load the database."""
        # Build list of paths
        data_paths = {}
        for subject_id in self.all_ids:
            path_eeg_file = os.path.join(
                self.dataset_dir, PATH_REC, "01-02-%04d PSG.edf" % subject_id
            )
            path_states_file = os.path.join(
                self.dataset_dir, PATH_STATES, "01-02-%04d Base.edf" % subject_id
            )
            path_marks_1_file = os.path.join(
                self.dataset_dir, PATH_MARKS, "01-02-%04d KComplexesE1.edf" % subject_id
            )
            # Save paths
            ind_dict = {
                KEY_FILE_EEG: path_eeg_file,
                KEY_FILE_STATES: path_states_file,
                "%s_1" % KEY_FILE_MARKS: path_marks_1_file,
            }
            # Check paths
            for key in ind_dict:
                if not os.path.isfile(ind_dict[key]):
                    print("File not found: %s" % ind_dict[key])
            data_paths[subject_id] = ind_dict
        print("%d records in %s dataset." % (len(data_paths), self.dataset_name))
        print("Subject IDs: %s" % self.all_ids)
        return data_paths

    def _read_eeg(self, path_eeg_file):
        """Loads signal from 'path_eeg_file', does filtering and resampling."""
        with pyedflib.EdfReader(path_eeg_file) as file:
            channel_names = file.getSignalLabels()
            channel_to_extract = channel_names.index(self.channel)
            signal = file.readSignal(channel_to_extract)
            fs_old = file.samplefrequency(channel_to_extract)
            # Check
            print("Channel extracted: %s" % file.getLabel(channel_to_extract))

        # Particular fix for mass dataset:
        fs_old_round = int(np.round(fs_old))
        # Transform the original fs frequency with decimals to rounded version
        signal = utils.resample_signal_linear(
            signal, fs_old=fs_old, fs_new=fs_old_round
        )

        # Broand bandpass filter to signal
        signal = utils.broad_filter(signal, fs_old_round)

        # Now resample to the required frequency
        if self.fs != fs_old_round:
            print("Resampling from %d Hz to required %d Hz" % (fs_old_round, self.fs))
            signal = utils.resample_signal(signal, fs_old=fs_old_round, fs_new=self.fs)
        else:
            print("Signal already at required %d Hz" % self.fs)

        signal = signal.astype(np.float32)
        return signal

    def _read_marks(self, path_marks_file):
        """Loads data spindle annotations from 'path_marks_file'.
        Marks with a duration outside feasible boundaries are removed.
        Returns the sample-stamps of each mark."""
        with pyedflib.EdfReader(path_marks_file) as file:
            annotations = file.readAnnotations()
        onsets = np.array(annotations[0])
        durations = np.array(annotations[1])
        offsets = onsets + durations
        marks_time = np.stack((onsets, offsets), axis=1)  # time-stamps
        # Transforms to sample-stamps
        marks = np.round(marks_time * self.fs).astype(np.int32)
        # Fix durations that are outside standards
        marks = stamp_correction.filter_duration_stamps(
            marks, self.fs, self.min_kc_duration, None
        )
        return marks

    def _read_states_raw(self, path_states_file):
        """Loads hypnogram from 'path_states_file'."""
        with pyedflib.EdfReader(path_states_file) as file:
            annotations = file.readAnnotations()
        onsets = np.array(annotations[0])  # In seconds
        durations = np.round(np.array(annotations[1]))  # In seconds
        stages_str = annotations[2]
        # keep only 20s durations
        valid_idx = durations == self.page_duration
        onsets = onsets[valid_idx]
        stages_str = stages_str[valid_idx]
        stages_char = np.asarray([single_annot[-1] for single_annot in stages_str])
        # Sort by onset
        sorted_locs = np.argsort(onsets)
        onsets = onsets[sorted_locs]
        stages_char = stages_char[sorted_locs]
        # The hypnogram could start at a sample different from 0
        start_time = onsets[0]
        onsets_relative = onsets - start_time
        onsets_pages = np.round(onsets_relative / self.page_duration).astype(np.int32)
        n_scored_pages = (
            1 + onsets_pages[-1]
        )  # might be greater than onsets_pages.size if some labels are missing
        start_sample = int(start_time * self.fs)
        hypnogram = (n_scored_pages + 1) * [
            self.unknown_id
        ]  # if missing, it will be "?", we add one final '?'
        for scored_pos, scored_label in zip(onsets_pages, stages_char):
            hypnogram[scored_pos] = scored_label
        hypnogram = np.asarray(hypnogram)
        return hypnogram, start_sample

    def _fix_signal_and_states(self, signal, hypnogram, start_sample):
        # Crop start of signal
        signal = signal[start_sample:]
        # Find the largest valid sample, common in both signal and hypnogram, with an integer number of pages
        n_samples_from_signal = int(self.page_size * (signal.size // self.page_size))
        n_samples_from_hypnogram = int(hypnogram.size * self.page_size)
        n_samples_valid = min(n_samples_from_signal, n_samples_from_hypnogram)
        n_pages_valid = int(n_samples_valid / self.page_size)
        # Fix signal and hypnogram according to this maximum sample
        signal = signal[:n_samples_valid]
        hypnogram = hypnogram[:n_pages_valid]
        end_sample = (
            start_sample + n_samples_valid
        )  # wrt original beginning of recording, useful for marks
        return signal, hypnogram, end_sample

    def _hypnogram_selections(self, hypnogram):
        total_pages = hypnogram.size
        n2_pages = np.where(hypnogram == self.n2_id)[0].astype(np.int16)
        # Drop first and last page of the whole registers if they where selected.
        last_page = total_pages - 1
        n2_pages = n2_pages[(n2_pages != 0) & (n2_pages != last_page)]
        all_pages = np.arange(1, total_pages - 1, dtype=np.int16)
        return all_pages, n2_pages

    def _fix_marks(self, marks, start_sample, end_sample):
        marks = marks - start_sample  # reference to new start
        end_sample = end_sample - start_sample
        marks = utils.filter_stamps(marks, 0, end_sample - 1)  # avoid runaway
        return marks

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
                self.get_subject_filt_signal(sub_id) for sub_id in subject_id_list
            ]
        else:
            subset_signals = None
        return subset_signals

    def get_filt_signals(self):
        if self.exists_cache:
            subset_signals = [
                self.get_subject_filt_signal(sub_id) for sub_id in self.all_ids
            ]
        else:
            subset_signals = None
        return subset_signals

    def exists_filt_signal_cache(self):
        return self.exists_cache
