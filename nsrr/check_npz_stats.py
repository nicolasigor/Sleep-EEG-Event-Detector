import os
import sys

import numpy as np

project_root = ".."
sys.path.append(project_root)

DATASETS_PATH = os.path.join(project_root, 'resources', 'datasets', 'nsrr')


if __name__ == "__main__":

    dataset_name_list = [
        'shhs1',
        'mros1',
        'chat1',
        'sof',
        'cfs',
        'ccshs'
    ]

    # Keys in dataset:
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

    n2_id = 'Stage 2 sleep|2'

    for dataset_name in dataset_name_list:
        print("Check %s" % dataset_name)

        npz_dir = os.path.abspath(os.path.join(DATASETS_PATH, dataset_name, 'register_and_state'))
        all_files = os.listdir(npz_dir)
        all_files.sort()
        all_files = [f for f in all_files if '.npz' in f]
        all_fname = np.array(all_files)
        all_files = [os.path.join(npz_dir, f) for f in all_files]
        all_files = np.array(all_files)

        all_original_fs = []
        all_channel = []
        duration_in_seconds = 0
        stage_labels = []
        for f in all_files:
            data_dict = np.load(f)

            # check original fs
            original_fs = data_dict['original_sampling_rate']
            all_original_fs.append(original_fs)

            # check channel extracted
            channel = data_dict['channel']
            all_channel.append(channel)

            # check N2 duration
            hypnogram = data_dict['hypnogram']
            stage_labels.append(np.unique(hypnogram))
            epoch_duration = data_dict['epoch_duration']
            n2_pages = (hypnogram == n2_id).sum()
            n2_duration = epoch_duration * n2_pages
            duration_in_seconds += n2_duration
        stage_labels = np.unique(np.concatenate(stage_labels))

        print("\nReport:")
        print("Subjects %d" % len(all_files))
        print("Duration: %1.4f s, %1.4f h" % (
            duration_in_seconds,
            duration_in_seconds / 3600
        ))

        print("\nOriginal fs found:")
        values, counts = np.unique(all_original_fs, return_counts=True)
        for v, c in zip(values, counts):
            print("%s: %d" % (v, c))

        print("\nChannels found:")
        values, counts = np.unique(all_channel, return_counts=True)
        for v, c in zip(values, counts):
            print("%s: %d" % (v, c))

        print('\nSleep stage labels found:', stage_labels)
