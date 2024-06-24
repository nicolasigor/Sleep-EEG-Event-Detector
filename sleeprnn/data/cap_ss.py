from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import pandas as pd

from sleeprnn.common import constants
from sleeprnn.data import utils
from sleeprnn.data import stamp_correction
from sleeprnn.data.dataset import Dataset
from sleeprnn.data.dataset import (
    KEY_EEG,
    KEY_N2_PAGES,
    KEY_ALL_PAGES,
    KEY_MARKS,
    KEY_HYPNOGRAM,
)

PATH_CAP_RELATIVE = "unlabeled_cap_npz"
PATH_REC_AND_STATE = "register_and_state"
PATH_MARKS = "spindle"

KEY_FILE_EEG_STATE = "file_eeg_state"
KEY_FILE_MARKS = "file_marks"

ALL_IDS = [
    "1-001",
    "1-002",
    "1-003",
    "1-004",
    "1-005",
    "1-006",
    "1-007",
    "1-008",
    "1-009",
    "1-010",
    "1-011",
    "1-012",
    "1-013",
    "1-014",
    "1-015",
    "1-016",
    "2-001",
    "2-002",
    "3-001",
    "3-002",
    "3-003",
    "3-004",
    "3-005",
    "3-006",
    "3-007",
    "3-008",
    "3-009",
    "4-001",
    "4-002",
    "4-003",
    "4-004",
    "4-005",
    "5-001",
    "5-002",
    "5-003",
    "5-004",
    "5-005",
    "5-007",
    "5-008",
    "5-009",
    "5-010",
    "5-011",
    "5-012",
    "5-013",
    "5-014",
    "5-015",
    "5-016",
    "5-017",
    "5-018",
    "5-019",
    "5-020",
    "5-021",
    "5-022",
    "5-023",
    "5-024",
    "5-025",
    "5-026",
    "5-028",
    "5-029",
    "5-030",
    "5-032",
    "5-034",
    "5-035",
    "5-036",
    "5-037",
    "5-038",
    "5-039",
    "5-040",
    "6-001",
    "6-002",
    "6-003",
    "6-004",
    "6-005",
    "6-006",
    "6-007",
    "6-008",
    "6-009",
    "6-010",
    "7-001",
    "7-002",
    "7-003",
    "7-004",
    "7-005",
    "7-006",
    "7-007",
    "7-008",
    "7-009",
    "7-010",
    "7-011",
    "7-012",
    "7-013",
    "7-014",
    "7-015",
    "7-016",
    "7-017",
    "7-018",
    "7-019",
    "7-020",
    "7-021",
    "7-022",
    "8-001",
    "8-002",
    "8-003",
    "8-004",
]
IDS_INVALID = [
    "3-002",
    "3-003",
    "4-002",
    "5-001",
    "5-006",
    "5-027",
    "5-031",
    "5-033",
    "6-004",
    "6-006",
    "6-007",
    "7-001",
    "7-002",
    "7-003",
    "7-004",
    "7-005",
    "7-007",
    "7-008",
    "7-009",
    "7-010",
    "7-011",
    "7-013",
    "7-014",
    "7-015",
    "7-017",
    "7-020",
    "8-003",
    "8-004",
]


class CapSS(Dataset):
    """This is a class to manipulate the CAP data EEG dataset.

    Expert 1:
    The sleep spindle marks are detections made by the A7 algorithm:
    Lacourse, K., Delfrate, J., Beaudry, J., Peppard, P., & Warby, S. C. (2019).
    A sleep spindle detection algorithm that emulates human expert spindle scoring.
    Journal of neuroscience methods, 316, 3-11.
    The four parameters were fitted on MASS-SS2.

    Expert 2: S1 (abs)
    Expert 3: S2 (rel)

    Expected directory tree inside DATA folder (see utils.py):

    PATH_CAP_RELATIVE
    |__ PATH_REC_AND_STATE
        |__ cap_1-001.npz
        |__ cap_1-002.npz
    |__ PATH_MARKS
        |__ EventDetection_s1-001_*.txt
        |__ ...
    """

    def __init__(self, params=None, load_checkpoint=False, verbose=True, **kwargs):
        """Constructor"""
        # CAP parameters
        self.state_ids = np.array(["S1", "S2", "S3", "S4", "R", "W", "?"])
        self.unknown_id = "?"  # Character for unknown state in hypnogram
        self.n2_id = "S2"  # Character for N2 identification in hypnogram
        self.original_page_duration = 30  # Time of window page [s]

        # Sleep spindles characteristics
        self.min_ss_duration = 0.5  # Minimum duration of SS in seconds
        self.max_ss_duration = 3  # Maximum duration of SS in seconds

        all_ids = [s for s in ALL_IDS if s not in IDS_INVALID]
        all_ids.sort()

        super(CapSS, self).__init__(
            dataset_dir=PATH_CAP_RELATIVE,
            load_checkpoint=load_checkpoint,
            dataset_name=constants.CAP_SS_NAME,
            all_ids=all_ids,
            event_name=constants.SPINDLE,
            hypnogram_sleep_labels=["S1", "S2", "S3", "S4", "R"],
            hypnogram_page_duration=20,
            n_experts=3,
            params=params,
            verbose=verbose,
        )
        self.global_std = None
        if verbose:
            print("Global STD:", self.global_std)

    def _load_from_source(self):
        """Loads the data from files and transforms it appropriately."""
        data_paths = self._get_file_paths()
        data = {}
        save_dir = os.path.join(self.dataset_dir, "pretty_files")
        os.makedirs(save_dir, exist_ok=True)
        n_data = len(data_paths)
        start = time.time()
        for i, subject_id in enumerate(data_paths.keys()):
            print("\nLoading ID %s" % subject_id)
            path_dict = data_paths[subject_id]
            # Read data
            signal, hypnogram_original = self._read_npz(path_dict[KEY_FILE_EEG_STATE])
            n2_pages = self._get_n2_pages(hypnogram_original)
            marks_1 = self._read_marks(path_dict["%s_1" % KEY_FILE_MARKS])
            signal, n2_pages, marks_1, hypnogram_20s = self._short_signals(
                signal, n2_pages, marks_1
            )

            # ################
            # Remove weird pages from N2 labels
            weird_locs = np.where(np.abs(signal) > 300)[0]
            weird_pages = np.unique(np.floor(weird_locs / self.page_size)).astype(
                np.int32
            )
            if subject_id == "2-001":
                weird_pages = np.concatenate(
                    [weird_pages, np.arange(0, 60 + 0.001), np.arange(120, 130 + 0.001)]
                )
                weird_pages = np.unique(weird_pages).astype(np.int32)
            hypnogram_20s[weird_pages] = self.unknown_id
            n2_pages = np.where(hypnogram_20s == self.n2_id)[0]
            # ################

            total_pages = int(signal.size / self.page_size)
            all_pages = np.arange(1, total_pages - 1, dtype=np.int16)

            # Marks from simple detectors S1 (abs) and S2 (rel), respectively
            marks_2 = self._read_marks_simple(path_dict["%s_2" % KEY_FILE_MARKS])
            marks_3 = self._read_marks_simple(path_dict["%s_3" % KEY_FILE_MARKS])

            print("N2 pages: %d" % n2_pages.shape[0])
            print("Whole-night pages: %d" % all_pages.shape[0])
            print(
                "Marks SS from A7 with original paper params         : %d"
                % marks_1.shape[0]
            )
            print(
                "Marks SS from S1-abs with thr 10 uV and thr low 0.86: %d"
                % marks_2.shape[0]
            )
            print(
                "Marks SS from S2-rel with thr 2.9 and thr low 0.80  : %d"
                % marks_3.shape[0]
            )

            # Save data
            ind_dict = {
                KEY_EEG: signal.astype(np.float32),
                KEY_N2_PAGES: n2_pages.astype(np.int16),
                KEY_ALL_PAGES: all_pages.astype(np.int16),
                KEY_HYPNOGRAM: hypnogram_20s,
                "%s_1" % KEY_MARKS: marks_1.astype(np.int32),
                "%s_2" % KEY_MARKS: marks_2.astype(np.int32),
                "%s_3" % KEY_MARKS: marks_3.astype(np.int32),
            }
            fname = os.path.join(save_dir, "subject_%s.npz" % subject_id)
            data[subject_id] = {"pretty_file_path": fname}
            np.savez(fname, **ind_dict)
            print(
                "Loaded ID %s (%02d/%02d ready). Time elapsed: %1.4f [s]"
                % (subject_id, i + 1, n_data, time.time() - start)
            )
        print("%d records have been read." % n_data)
        return data

    def read_subject_data(self, subject_id):
        path_dict = self.data[subject_id]
        ind_dict = np.load(path_dict["pretty_file_path"])

        loaded_ind_dict = {}
        for key in ind_dict.files:
            loaded_ind_dict[key] = ind_dict[key]

        return loaded_ind_dict

    def _get_file_paths(self):
        """Returns a list of dicts containing paths to load the database."""
        # Build list of paths
        data_paths = {}
        for subject_id in self.all_ids:
            path_eeg_state_file = os.path.join(
                self.dataset_dir, PATH_REC_AND_STATE, "cap_%s.npz" % subject_id
            )
            path_marks_1_file = os.path.join(
                self.dataset_dir,
                PATH_MARKS,
                "EventDetection_s%s_absSigPow(1.25)_relSigPow(1.6)_sigCov(1.3)_sigCorr(0.69).txt"
                % subject_id,
            )
            path_marks_2_file = os.path.join(
                self.dataset_dir,
                "%s_s1_abs" % PATH_MARKS,
                "SimpleDetectionAbsolute_s%s_thr10-0.86_fs200.txt" % subject_id,
            )
            path_marks_3_file = os.path.join(
                self.dataset_dir,
                "%s_s2_rel" % PATH_MARKS,
                "SimpleDetectionRelative_s%s_thr2.9-0.80_fs200.txt" % subject_id,
            )
            # Save paths
            ind_dict = {
                KEY_FILE_EEG_STATE: path_eeg_state_file,
                "%s_1" % KEY_FILE_MARKS: path_marks_1_file,
                "%s_2" % KEY_FILE_MARKS: path_marks_2_file,
                "%s_3" % KEY_FILE_MARKS: path_marks_3_file,
            }
            # Check paths
            for key in ind_dict:
                if not os.path.isfile(ind_dict[key]):
                    print("File not found: %s" % ind_dict[key])
            data_paths[subject_id] = ind_dict
        print("%d records in %s dataset." % (len(data_paths), self.dataset_name))
        print("Subject IDs: %s" % self.all_ids)
        return data_paths

    def _read_npz(self, path_eeg_state_file):
        data = np.load(path_eeg_state_file)
        hypnogram = data["hypnogram"]
        hypnogram = self._fix_hypnogram(hypnogram)
        signal = data["signal"]
        # Signal is already filtered to 0.1-35 and sampled to 200Hz
        original_fs = data["sampling_rate"]
        # Now resample to the required frequency
        if self.fs != original_fs:
            print("Resampling from %d Hz to required %d Hz" % (original_fs, self.fs))
            signal = utils.resample_signal(signal, fs_old=original_fs, fs_new=self.fs)
        signal = signal.astype(np.float32)
        return signal, hypnogram

    def _fix_hypnogram(self, hypno):
        hypno[hypno == "1S3"] = "?"
        hypno[hypno == "MT"] = "?"
        hypno[hypno == "REM"] = "R"
        return hypno

    def _get_n2_pages(self, hypnogram_original):
        signal_total_duration = len(hypnogram_original) * self.original_page_duration
        # Extract N2 pages
        n2_pages_original = np.where(hypnogram_original == self.n2_id)[0]
        onsets_original = n2_pages_original * self.original_page_duration
        offsets_original = (n2_pages_original + 1) * self.original_page_duration
        total_pages = int(np.ceil(signal_total_duration / self.page_duration))
        n2_pages_onehot = np.zeros(total_pages, dtype=np.int16)
        for i in range(total_pages):
            onset_new_page = i * self.page_duration
            offset_new_page = (i + 1) * self.page_duration
            for j in range(n2_pages_original.size):
                intersection = (onset_new_page < offsets_original[j]) and (
                    onsets_original[j] < offset_new_page
                )
                if intersection:
                    n2_pages_onehot[i] = 1
                    break
        n2_pages = np.where(n2_pages_onehot == 1)[0]
        # Drop first, last and second to last page of the whole registers
        # if they where selected.
        last_page = total_pages - 1
        n2_pages = n2_pages[
            (n2_pages != 0) & (n2_pages != last_page) & (n2_pages != last_page - 1)
        ]
        n2_pages = n2_pages.astype(np.int16)
        return n2_pages

    def _read_marks(self, path_marks_file):
        """Loads data spindle annotations from 'path_marks_file'.
        Marks with a duration outside feasible boundaries are removed.
        Returns the sample-stamps of each mark."""
        pred_data = pd.read_csv(path_marks_file, sep="\t")
        # We substract 1 to translate from matlab to numpy indexing system
        start_samples = pred_data.start_sample.values - 1
        end_samples = pred_data.end_sample.values - 1
        marks = np.stack([start_samples, end_samples], axis=1).astype(np.int32)

        # Sample-stamps assume 200Hz sampling rate
        if self.fs != 200:
            print("Correcting marks from 200 Hz to %d Hz" % self.fs)
            # We need to transform the marks to the new sampling rate
            marks_time = marks.astype(np.float32) / 200.0
            # Transform to sample-stamps
            marks = np.round(marks_time * self.fs).astype(np.int32)

        # Combine marks that are too close according to standards
        marks = stamp_correction.combine_close_stamps(
            marks, self.fs, self.min_ss_duration
        )
        # Fix durations that are outside standards
        marks = stamp_correction.filter_duration_stamps(
            marks, self.fs, self.min_ss_duration, self.max_ss_duration
        )
        return marks

    def _read_marks_simple(self, path_marks_file):
        # files assume fs = 200Hz and shortened signals
        marks = np.loadtxt(path_marks_file, delimiter=",")
        marks = marks.astype(np.int32)
        # Sample-stamps assume 200Hz sampling rate
        if self.fs != 200:
            print("Correcting marks from 200 Hz to %d Hz" % self.fs)
            # We need to transform the marks to the new sampling rate
            marks_time = marks.astype(np.float32) / 200.0
            # Transform to sample-stamps
            marks = np.round(marks_time * self.fs).astype(np.int32)
        return marks

    def _short_signals(self, signal, n2_pages, marks):
        valid_pages = np.concatenate([n2_pages - 1, n2_pages, n2_pages + 1])
        valid_pages = np.clip(valid_pages, a_min=0, a_max=None)
        valid_pages = np.unique(
            valid_pages
        )  # it is ensured to have context at each side of n2 pages

        # Hypnogram of 20s pages with only N2 labels (everything else is invalid stage)
        n_pages_original = int(signal.size / self.page_size)
        hypnogram = np.array([self.unknown_id] * n_pages_original, dtype="<U2")
        hypnogram[n2_pages] = self.n2_id

        # Now simplify
        hypnogram = hypnogram[valid_pages]
        n2_pages = np.where(hypnogram == self.n2_id)[0]

        marks = utils.stamp2seq(marks, 0, signal.size - 1)
        marks = utils.extract_pages(marks, valid_pages, self.page_size)
        marks = marks.flatten()
        marks = utils.seq2stamp(marks)

        signal = utils.extract_pages(signal, valid_pages, self.page_size)
        signal = signal.flatten()
        return signal, n2_pages, marks, hypnogram
