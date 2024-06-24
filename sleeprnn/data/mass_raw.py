import os
import pyedflib
import numpy as np

from sleeprnn.data import utils

PATH_REC = "register"
PATH_MARKS = os.path.join("label", "spindle")
PATH_STATES = os.path.join("label", "state")
KEY_FILE_EEG = "file_eeg"
KEY_FILE_STATES = "file_states"
KEY_FILE_MARKS = "file_marks"
IDS_INVALID = [4, 8, 15, 16]
IDS_TEST = [2, 6, 12, 13]


class MassRaw(object):
    def __init__(self):
        self.fs = 256
        self.page_duration = 20
        self.page_size = int(self.page_duration * self.fs)
        self.channel = "EEG C3-CLE"
        self.state_ids = np.array(["1", "2", "3", "4", "R", "W", "?"])
        self.unknown_id = "?"  # Character for unknown state in hypnogram
        self.n2_id = "2"  # Character for N2 identification in hypnogram
        valid_ids = [i for i in range(1, 20) if i not in IDS_INVALID]
        self.test_ids = IDS_TEST
        self.train_ids = [i for i in valid_ids if i not in self.test_ids]
        self.dataset_dir = os.path.abspath(os.path.join(utils.PATH_DATA, "mass"))
        self.all_ids = self.train_ids + self.test_ids
        self.dataset_name = "mass_raw"

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
                self.dataset_dir, PATH_MARKS, "01-02-%04d SpindleE1.edf" % subject_id
            )
            path_marks_2_file = os.path.join(
                self.dataset_dir, PATH_MARKS, "01-02-%04d SpindleE2.edf" % subject_id
            )
            # Save paths
            ind_dict = {
                KEY_FILE_EEG: path_eeg_file,
                KEY_FILE_STATES: path_states_file,
                "%s_1" % KEY_FILE_MARKS: path_marks_1_file,
                "%s_2" % KEY_FILE_MARKS: path_marks_2_file,
            }
            # Check paths
            for key in ind_dict:
                if not os.path.isfile(ind_dict[key]):
                    print("File not found: %s" % ind_dict[key])
            data_paths[subject_id] = ind_dict
        return data_paths

    def get_subject_data(self, subject_id):
        data_paths = self._get_file_paths()
        path_dict = data_paths[subject_id]
        signal = self._read_eeg(path_dict[KEY_FILE_EEG])
        hypnogram, start_sample = self._read_states_raw(path_dict[KEY_FILE_STATES])
        signal, hypnogram, end_sample = self._fix_signal_and_states(
            signal, hypnogram, start_sample
        )
        return signal, hypnogram

    def _read_eeg(self, path_eeg_file):
        """Loads signal from 'path_eeg_file', does filtering and resampling."""
        with pyedflib.EdfReader(path_eeg_file) as file:
            channel_names = file.getSignalLabels()
            channel_to_extract = channel_names.index(self.channel)
            signal = file.readSignal(channel_to_extract)
            fs_old = file.samplefrequency(channel_to_extract)
        # Particular fix for mass dataset:
        fs_old_round = int(np.round(fs_old))
        # Transform the original fs frequency with decimals to rounded version
        signal = utils.resample_signal_linear(
            signal, fs_old=fs_old, fs_new=fs_old_round
        )
        signal = signal.astype(np.float32)
        return signal

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
