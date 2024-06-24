"""inta_ss.py: Defines the INTA class that manipulates the INTA database."""

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
from sleeprnn.data.dataset import KEY_EEG, KEY_N2_PAGES, KEY_ALL_PAGES, KEY_MARKS, KEY_HYPNOGRAM

PATH_INTA_RELATIVE = 'inta'
PATH_REC = 'register'
PATH_MARKS = os.path.join('label', 'spindle')
PATH_STATES = os.path.join('label', 'state')

KEY_FILE_EEG = 'file_eeg'
KEY_FILE_STATES = 'file_states'
KEY_FILE_MARKS = 'file_marks'


class IntaSS(Dataset):
    """This is a class to manipulate the INTA data EEG dataset.

    Expected directory tree inside DATA folder (see utils.py):

    PATH_INTA_RELATIVE
    |__ PATH_REC
        |__ ADGU101504.rec
        |__ ALUR012904.rec
        |__ ...
    |__ PATH_STATES
        |__ StagesOnly_ADGU101504.txt
        |__ ...
    |__ PATH_MARKS
        |__ NewerWinsFix_SS_ADGU101504.txt
        |__ ...

    If '...Fix...' marks files do not exist, then you should
    set the 'repair_stamps' flag to True. In that case, it is expected:

    |__ PATH_MARKS
        |__ SS_ADGU101504.txt
        |__ ...
    """

    def __init__(self, params=None, load_checkpoint=False, repair_stamps=False, verbose=True, **kwargs):
        """Constructor"""
        # INTA parameters
        self.original_page_duration = 30  # Time of window page [s]
        self.channel = 0  # Channel for SS, first is F4-C4, third is F3-C3
        self.state_ids = np.array([1, 2, 3, 4, 5, 6])
        self.n2_id = 3  # Character for N2 identification in hypnogram
        # Sleep states dictionary for INTA:
        # 1:SQ4   2:SQ3   3:SQ2   4:SQ1   5:REM   6:WA
        self.names = np.array([
            'ADGU',
            'ALUR',
            'BECA',
            'BRCA',
            'BRLO',
            'BTOL',
            'CAPO',
            'CRCA',
            'ESCI',
            'TAGO'
        ])

        # Sleep spindles characteristics
        self.min_ss_duration = 0.5  # Minimum duration of SS in seconds (set to 0.5 according to INTA)
        self.max_ss_duration = 5  # Maximum duration of SS in seconds

        if repair_stamps:
            self._repair_stamps()

        all_ids = np.arange(1, self.names.size + 1).tolist()

        super(IntaSS, self).__init__(
            dataset_dir=PATH_INTA_RELATIVE,
            load_checkpoint=load_checkpoint,
            dataset_name=constants.INTA_SS_NAME,
            all_ids=all_ids,
            event_name=constants.SPINDLE,
            hypnogram_sleep_labels=[1, 2, 3, 4, 5],
            hypnogram_page_duration=self.original_page_duration,
            params=params,
            verbose=verbose
        )
        self.global_std = None
        if verbose:
            print('Global STD:', self.global_std)

    def get_name(self, subject_id):
        return self.names[subject_id-1]

    def _load_from_source(self):
        """Loads the data from files and transforms it appropriately."""
        data_paths = self._get_file_paths()
        data = {}
        n_data = len(data_paths)
        block_duration = np.lcm(self.page_duration, self.original_page_duration)
        start = time.time()
        for i, subject_id in enumerate(data_paths.keys()):
            print('\nLoading ID %d (%s)' % (subject_id, self.get_name(subject_id)))
            path_dict = data_paths[subject_id]
            if len(path_dict[KEY_FILE_EEG]) == 2:
                signal_1 = self._read_eeg(path_dict[KEY_FILE_EEG][0])
                signal_2 = self._read_eeg(path_dict[KEY_FILE_EEG][1])
                signal_1, hypnogram_original_1 = self._read_states(
                    path_dict[KEY_FILE_STATES][0], signal_1, block_duration)
                signal_2, hypnogram_original_2 = self._read_states(
                    path_dict[KEY_FILE_STATES][1], signal_2, block_duration)
                signal_between = np.zeros(int(self.fs * block_duration))  # 3 20s, 2 30s
                states_between = [6, 6]  # in original 30s pages
                signal = np.concatenate([signal_1, signal_between, signal_2])
                hypnogram_original = np.concatenate([hypnogram_original_1, states_between, hypnogram_original_2])
                # Now marks
                marks_1 = self._read_marks(path_dict['%s_1' % KEY_FILE_MARKS][0])
                marks_2 = self._read_marks(path_dict['%s_1' % KEY_FILE_MARKS][1])
                marks_1 = utils.filter_stamps(marks_1, 0, signal_1.size - 1)  # avoid runaway
                marks_2 = utils.filter_stamps(marks_2, 0, signal_2.size - 1)
                offset_for_marks_2 = signal_1.size + signal_between.size
                marks_2_shifted = marks_2 + offset_for_marks_2
                marks = np.concatenate([marks_1, marks_2_shifted], axis=0)
            else:
                signal = self._read_eeg(path_dict[KEY_FILE_EEG][0])
                signal, hypnogram_original = self._read_states(
                    path_dict[KEY_FILE_STATES][0], signal, block_duration)
                marks = self._read_marks(path_dict['%s_1' % KEY_FILE_MARKS][0])
            n2_pages = self._get_n2_pages(hypnogram_original)
            total_pages = int(np.ceil(signal.size / self.page_size))
            all_pages = np.arange(1, total_pages - 1, dtype=np.int16)
            print('N2 pages: %d' % n2_pages.shape[0])
            print('Whole-night pages: %d' % all_pages.shape[0])
            print('Marks SS from E1: %d' % marks.shape[0])

            # Save data
            ind_dict = {
                KEY_EEG: signal,
                KEY_N2_PAGES: n2_pages,
                KEY_ALL_PAGES: all_pages,
                KEY_HYPNOGRAM: hypnogram_original,
                '%s_1' % KEY_MARKS: marks
            }
            data[subject_id] = ind_dict
            print('Loaded ID %d (%s) (%02d/%02d ready). Time elapsed: %1.4f [s]'
                  % (subject_id, self.get_name(subject_id), i+1, n_data, time.time()-start))
        print('%d records have been read.' % len(data))
        return data

    def _get_file_paths(self):
        """Returns a list of dicts containing paths to load the database."""
        # Build list of paths
        data_paths = {}
        register_files = os.listdir(os.path.join(self.dataset_dir, PATH_REC))
        state_files = os.listdir(os.path.join(self.dataset_dir, PATH_STATES))
        spindle_files = os.listdir(os.path.join(self.dataset_dir, PATH_MARKS))
        register_files = [f for f in register_files if '.rec' in f]
        state_files = [f for f in state_files if 'StagesOnly' in f]
        spindle_files_manual_fix = [f for f in spindle_files if 'Revision_SS' in f]
        spindle_files_auto_fix = [f for f in spindle_files if 'NewerWinsFix_v2_SS' in f]
        for subject_id in self.all_ids:
            subject_name = self.names[subject_id - 1]
            # Find rec
            subject_recs = [f for f in register_files if subject_name in f]
            subject_recs.sort()
            path_eeg_file = tuple([
                os.path.join(self.dataset_dir, PATH_REC, f)
                for f in subject_recs
            ])
            # Find states
            subject_states = [f for f in state_files if subject_name in f]
            subject_states.sort()
            path_states_file = tuple([
                os.path.join(self.dataset_dir, PATH_STATES, f)
                for f in subject_states
            ])
            # Find spindles
            subject_spindles = [f for f in spindle_files_manual_fix if subject_name in f]
            if len(subject_spindles) == 0:
                print("Subject %02d (%s): Spindle manual session not found, using automatic NewerWins fix instead." % (
                    subject_id, subject_name))
                subject_spindles = [f for f in spindle_files_auto_fix if subject_name in f]
            else:
                print("Subject %02d (%s): Using spindle manual session." % (subject_id, subject_name))
            subject_spindles.sort()
            path_marks_1_file = tuple([
                os.path.join(self.dataset_dir, PATH_MARKS, f)
                for f in subject_spindles
            ])
            # Save paths
            ind_dict = {
                KEY_FILE_EEG: path_eeg_file,
                KEY_FILE_STATES: path_states_file,
                '%s_1' % KEY_FILE_MARKS: path_marks_1_file
            }
            data_paths[subject_id] = ind_dict
        print('%d records in %s dataset.' % (len(data_paths), self.dataset_name))
        print('Subject IDs: %s' % self.all_ids)
        return data_paths

    def _read_eeg(self, path_eeg_file):
        """Loads signal from 'path_eeg_file', does filtering."""
        with pyedflib.EdfReader(path_eeg_file) as file:
            signal = file.readSignal(self.channel)
            fs_old = file.samplefrequency(self.channel)
            # Check
            print('Channel extracted: %s' % file.getLabel(self.channel))
        # Particular for INTA dataset
        fs_old = int(np.round(fs_old))
        # Broand bandpass filter to signal
        signal = utils.broad_filter(signal, fs_old)
        # Now resample to the required frequency
        if self.fs != fs_old:
            print('Resampling from %d Hz to required %d Hz' % (fs_old, self.fs))
            signal = utils.resample_signal(signal, fs_old=fs_old, fs_new=self.fs)
        else:
            print('Signal already at required %d Hz' % self.fs)
        signal = signal.astype(np.float32)
        return signal

    def _read_marks(self, path_marks_file):
        """Loads data spindle annotations from 'path_marks_file'.
        Marks with a duration outside feasible boundaries are removed.
        Returns the sample-stamps of each mark."""
        # Recovery sample-stamps
        marks_file = np.loadtxt(path_marks_file, dtype='i', delimiter=' ')
        marks = marks_file[marks_file[:, 5] == self.channel + 1][:, [0, 1]]
        marks = np.round(marks).astype(np.int32)
        # Sample-stamps assume 200Hz sampling rate
        marks_fs = 200
        if self.fs != marks_fs:
            print('Correcting marks from 200 Hz to %d Hz' % self.fs)
            # We need to transform the marks to the new sampling rate
            marks_time = marks.astype(np.float32) / marks_fs
            # Transform to sample-stamps
            marks = np.round(marks_time * self.fs).astype(np.int32)
        # Combine marks that are too close according to standards
        marks = stamp_correction.combine_close_stamps(marks, self.fs, self.min_ss_duration)
        # Fix durations that are outside standards
        marks = stamp_correction.filter_duration_stamps(marks, self.fs, self.min_ss_duration, self.max_ss_duration)
        return marks

    def _read_states(self, path_states_file, signal, block_duration):
        states = np.loadtxt(path_states_file, dtype='i', delimiter=' ')
        # Crop signal and states to a valid length
        block_size = block_duration * self.fs
        n_blocks = np.floor(signal.size / block_size)
        max_sample = int(n_blocks * block_size)
        signal = signal[:max_sample]
        max_page = int(signal.size / (self.original_page_duration * self.fs))
        states = states[:max_page]
        return signal, states

    def _get_n2_pages(self, hypnogram_original):
        signal_total_duration = len(hypnogram_original) * self.original_page_duration
        # Extract N2 pages
        n2_pages_original = np.where(hypnogram_original == self.n2_id)[0]
        print("Original N2 pages: %d" % len(n2_pages_original))
        onsets_original = n2_pages_original * self.original_page_duration
        offsets_original = (n2_pages_original + 1) * self.original_page_duration
        total_pages = int(np.ceil(signal_total_duration / self.page_duration))
        n2_pages_onehot = np.zeros(total_pages, dtype=np.int16)
        for i in range(total_pages):
            onset_new_page = i * self.page_duration
            offset_new_page = (i + 1) * self.page_duration
            for j in range(n2_pages_original.size):
                intersection = (onset_new_page < offsets_original[j]) and (onsets_original[j] < offset_new_page)
                if intersection:
                    n2_pages_onehot[i] = 1
                    break
        n2_pages = np.where(n2_pages_onehot == 1)[0]
        # Drop first and last page of the whole registers if they where selected.
        last_page = total_pages - 1
        n2_pages = n2_pages[(n2_pages != 0) & (n2_pages != last_page)]
        n2_pages = n2_pages.astype(np.int16)
        return n2_pages

    def _repair_stamps(self):
        print('Repairing INTA stamps (Newer Wins Strategy + 0.5s criterion)')
        filename_format = 'NewerWinsFix_v2_SS_%s.txt'
        inta_folder = os.path.join(utils.PATH_DATA, PATH_INTA_RELATIVE)
        channel_for_txt = self.channel + 1
        names_in_files = [
            'ADGU101504',
            'ALUR012904',
            'BECA011405',
            'BRCA062405',
            'BRLO041102',
            'BTOL083105',
            'BTOL090105',
            'CAPO092605',
            'CRCA020205',
            'ESCI031905',
            'TAGO061203']
        for name in names_in_files:
            print('Fixing %s' % name)
            path_marks_file = os.path.abspath(os.path.join(inta_folder, PATH_MARKS, 'original', 'SS_%s.txt' % name))
            path_eeg_file = os.path.abspath(os.path.join(inta_folder, PATH_REC, '%s.rec' % name))

            # Read marks
            print('Loading %s' % path_marks_file)
            data = np.loadtxt(path_marks_file)
            for_this_channel = data[:, -1] == channel_for_txt
            data = data[for_this_channel]
            data = np.round(data).astype(np.int32)

            # Remove zero duration marks, and ensure that start time < end time
            new_data = []
            for i in range(data.shape[0]):
                if data[i, 0] > data[i, 1]:
                    print('Start > End time found and fixed.')
                    aux = data[i, 0]
                    data[i, 0] = data[i, 1]
                    data[i, 1] = aux
                    new_data.append(data[i, :])
                elif data[i, 0] < data[i, 1]:
                    new_data.append(data[i, :])
                else:  # Zero duration (equality)
                    print('Zero duration mark found and removed')
            data = np.stack(new_data, axis=0)

            # Remove stamps outside signal boundaries
            print('Loading %s' % path_eeg_file)
            with pyedflib.EdfReader(path_eeg_file) as file:
                signal = file.readSignal(0)
                signal_len = signal.shape[0]
            new_data = []
            for i in range(data.shape[0]):
                if data[i, 1] < signal_len:
                    new_data.append(data[i, :])
                else:
                    print('Stamp outside boundaries found and removed')
            data = np.stack(new_data, axis=0)

            raw_marks = data[:, [0, 1]]
            valid = data[:, 4]

            print('Starting correction... ', flush=True)
            # Separate according to valid value. Valid = 0 is ignored.

            raw_marks_1 = raw_marks[valid == 1]
            raw_marks_2 = raw_marks[valid == 2]

            print('Originally: %d marks with valid=1, %d marks with valid=2'
                  % (len(raw_marks_1), len(raw_marks_2)))

            # Remove marks with duration less than 0.5s
            size_thr = int(0.5 * 200)
            print('Removing events with less than %d samples.' % size_thr)
            durations_1 = raw_marks_1[:, 1] - raw_marks_1[:, 0]
            raw_marks_1 = raw_marks_1[durations_1 >= size_thr]
            durations_2 = raw_marks_2[:, 1] - raw_marks_2[:, 0]
            raw_marks_2 = raw_marks_2[durations_2 >= size_thr]

            print('After duration criterion: %d marks with valid=1, %d marks with valid=2'
                  % (len(raw_marks_1), len(raw_marks_2)))

            # Now we add sequentially from the end (from newer marks), and we
            # only add marks if they don't intersect with the current set.
            # In this way, we effectively choose newer stamps over old ones
            # We start with valid=2, and then we continue with valid=1, to
            # follow the correction rule:
            # Keep valid=2 always
            # Keep valid=1 only if there is no intersection with valid=2

            n_v1 = 0
            n_v2 = 0

            if len(raw_marks_2) > 0:
                final_marks = [raw_marks_2[-1, :]]
                final_valid = [2]
                n_v2 += 1
                for i in range(raw_marks_2.shape[0] - 2, -1, -1):
                    candidate_mark = raw_marks_2[i, :]
                    current_set = np.stack(final_marks, axis=0)
                    if not utils.stamp_intersects_set(candidate_mark, current_set):
                        # There is no intersection
                        final_marks.append(candidate_mark)
                        final_valid.append(2)
                        n_v2 += 1
                for i in range(raw_marks_1.shape[0] - 1, -1, -1):
                    candidate_mark = raw_marks_1[i, :]
                    current_set = np.stack(final_marks, axis=0)
                    if not utils.stamp_intersects_set(candidate_mark,
                                                      current_set):
                        # There is no intersection
                        final_marks.append(candidate_mark)
                        final_valid.append(1)
                        n_v1 += 1
            else:
                print('There is no valid=2 marks.')
                final_marks = [raw_marks_1[-1, :]]
                final_valid = [1]
                n_v1 += 1
                for i in range(raw_marks_1.shape[0] - 2, -1, -1):
                    candidate_mark = raw_marks_1[i, :]
                    current_set = np.stack(final_marks, axis=0)
                    if not utils.stamp_intersects_set(candidate_mark,
                                                      current_set):
                        # There is no intersection
                        final_marks.append(candidate_mark)
                        final_valid.append(1)
                        n_v1 += 1

            print('Finally: %d with valid=1, %d with valid=2' % (n_v1, n_v2))

            # Now concatenate everything
            final_marks = np.stack(final_marks, axis=0)
            final_valid = np.stack(final_valid, axis=0)

            # And sort according to time
            idx_sorted = np.argsort(final_marks[:, 0])
            final_marks = final_marks[idx_sorted]
            final_valid = final_valid[idx_sorted]

            # Now create array in right format
            # [start end -50 -50 valid channel]

            number_for_txt = -50
            n_marks = final_marks.shape[0]
            channel_column = channel_for_txt * np.ones(n_marks).reshape(
                [n_marks, 1])
            number_column = number_for_txt * np.ones(n_marks).reshape(
                [n_marks, 1])
            valid_column = final_valid.reshape([n_marks, 1])
            table = np.concatenate(
                [final_marks,
                 number_column, number_column,
                 valid_column, channel_column],
                axis=1
            )
            table = table.astype(np.int32)

            # Now sort according to start time
            table = table[table[:, 0].argsort()]
            print('Done. %d marks for channel %d' % (n_marks, channel_for_txt))

            # Now save into a file
            path_new_marks_file = os.path.abspath(os.path.join(
                inta_folder, PATH_MARKS, filename_format % name))
            np.savetxt(path_new_marks_file, table, fmt='%d', delimiter=' ')
            print('Fixed marks saved at %s\n' % path_new_marks_file)
