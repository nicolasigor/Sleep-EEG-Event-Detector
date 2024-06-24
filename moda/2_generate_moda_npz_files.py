"""Extracts only the relevant data from the full MODA database to save storage.

It saves the extracted signal for each subject in independent npz files under
the resources/datasets/moda/signals_npz/ directory.
These files are the input for the generate_moda_segments.py script.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from pprint import pprint

import numpy as np
import pandas as pd
import pyedflib

project_root = os.path.abspath('..')
sys.path.append(project_root)

# Change this to the path where the MASS dataset is stored:
PATH_MODA_RAW = '/home/ntapia/Projects/Sleep_Databases/MASS_Database_2020_Full/C1'
# Change this to the path where the 8_MODA_primChan_180sjt.txt file is stored:
PATH_SUBJECT_CHANNEL_INFO = '../resources/datasets/moda/8_MODA_primChan_180sjt.txt'


def get_signal(file, chn_name):
    channel_names = file.getSignalLabels()
    channel_loc = channel_names.index(chn_name)
    check = file.getLabel(channel_loc)
    assert check == chn_name
    fs = file.samplefrequency(channel_loc)
    signal = file.readSignal(channel_loc)
    return signal, fs


if __name__ == "__main__":
    save_dir = "../resources/datasets/moda/signals_npz"
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    print("Files will be saved at %s" % save_dir)

    info = pd.read_csv(PATH_SUBJECT_CHANNEL_INFO, delimiter='\t')
    subject_ids = info.subject.values
    subject_ids = [s.split(".")[0] for s in subject_ids]
    channels_for_moda = info.channel.values

    n_subjects = len(subject_ids)
    print("%d subjects" % n_subjects)
    for i in range(n_subjects):
        subject_id = subject_ids[i]
        channel_for_moda = channels_for_moda[i]
        signal_f = os.path.join(PATH_MODA_RAW, '%s PSG.edf' % subject_id)
        print("Loading %s from %s" % (channel_for_moda, signal_f))
        with pyedflib.EdfReader(signal_f) as file:
            channel_names = file.getSignalLabels()
            if channel_for_moda == 'C3-A2':
                # re-reference
                required_channel = [chn for chn in channel_names if 'C3' in chn][0]
                reference_channel = [chn for chn in channel_names if 'A2' in chn][0]
                required_signal, required_fs = get_signal(file, required_channel)
                reference_signal, reference_fs = get_signal(file, reference_channel)
                assert required_fs == reference_fs
                signal = required_signal - reference_signal
                channel_extracted = '(%s)-(%s)' % (required_channel, reference_channel)
            else:
                # use channel as-is
                required_channel = [chn for chn in channel_names if 'C3' in chn][0]
                required_signal, required_fs = get_signal(file, required_channel)
                signal = required_signal
                channel_extracted = required_channel
            signal = signal.astype(np.float32)
            fs = int(required_fs)
        print(
            '(%03d/%03d) Subject %s, sampling %s Hz, channel %s' % (i + 1, n_subjects, subject_id, fs, channel_extracted),
            flush=True)
        data_dict = {
            'dataset_id': 'MASS-C1',
            'subject_id': subject_id,
            'sampling_rate': fs,
            'channel': channel_extracted,
            'signal': signal
        }
        fname = os.path.join(save_dir, "moda_%s.npz" % subject_id)
        np.savez(fname, **data_dict)
