"""Some EDA."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from pprint import pprint


import numpy as np
import pyedflib

project_root = os.path.abspath('..')
sys.path.append(project_root)

PATH_MODA_RAW = '/home/ntapia/Projects/Sleep_Databases/MASS_Database_2020_Full/C1'


def get_filepaths(main_path):
    files = os.listdir(main_path)
    files = [f for f in files if '.edf' in f]
    signal_files = [f for f in files if 'PSG' in f]
    states_files = [f for f in files if 'Base' in f]
    signal_files = [os.path.join(main_path, f) for f in signal_files]
    states_files = [os.path.join(main_path, f) for f in states_files]
    signal_files.sort()
    states_files.sort()
    return signal_files, states_files


if __name__ == "__main__":
    # States are not necessary for the MODA dataset

    required_channel = 'C3'

    signal_files, states_files = get_filepaths(PATH_MODA_RAW)
    assert len(signal_files) == len(states_files)
    print("%d subjects" % len(signal_files))

    all_channel_names = []
    all_fs = []

    n_max = 10
    for signal_f, states_f in zip(signal_files[:n_max], states_files[:n_max]):
        subject_id_1 = signal_f.split("/")[-1].split(" ")[0]
        subject_id_2 = states_f.split("/")[-1].split(" ")[0]
        assert subject_id_1 == subject_id_2
        subject_id = subject_id_1
        print("ID %s" % subject_id)
        # Read signal
        with pyedflib.EdfReader(signal_f) as file:
            channel_names = file.getSignalLabels()
            channel_names_valid = [chn for chn in channel_names if required_channel in chn]
            fs_valid = []
            for chn in channel_names_valid:
                extraction_loc = channel_names.index(chn)
                fs_original = file.samplefrequency(extraction_loc)
                chn_check = file.getLabel(extraction_loc)
                assert chn_check == chn
                fs_valid.append(fs_original)
        print("    Channels", channel_names_valid)
        print("    Freqs   ", fs_valid)
        all_channel_names.append(channel_names_valid)
        all_fs.append(fs_valid)

    for l in [all_channel_names, all_fs]:
        l = np.concatenate(l).flatten()
        values, counts = np.unique(l, return_counts=True)
        for v, c in zip(values, counts):
            print("Value %s with count %s" % (v, c))
