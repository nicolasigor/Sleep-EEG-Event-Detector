import os
import sys
from pprint import pprint

import numpy as np
import pandas as pd

project_root = ".."
sys.path.append(project_root)

DATASETS_PATH = os.path.join(project_root, "resources", "datasets", "nsrr")


if __name__ == "__main__":

    dataset_name_list = [
        "chat1",
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

    for dataset_name in dataset_name_list:
        print("Check %s" % dataset_name)

        metadata_dir = os.path.abspath(
            os.path.join(DATASETS_PATH, dataset_name, "datasets")
        )
        all_files = os.listdir(metadata_dir)
        all_files.sort()

        # variables
        var_file = [
            f
            for f in all_files
            if ("variables" in f) and ("dictionary" in f) and (".csv" in f)
        ]
        pprint(var_file)

        var_file = os.path.join(metadata_dir, var_file[0])

        var_df = pd.read_csv(var_file)
        var_df = var_df[[("Demographics" in s) for s in var_df["folder"]]]
        print(var_df[["folder", "id", "display_name", "type"]])

        # Dataset file
        dataset_file = [f for f in all_files if ("dataset" in f) and (".csv" in f)]
        print("Datasets:")
        pprint(dataset_file)

        # id_col = var_df['id']
        # useful_id = [s for s in id_col if ('age' in s) or ('sex' in s) or ('gender' in s)]
        #
        # print(useful_id)
        #
        # # load dataset
        # metadata_path = os.path.join(metadata_dir, 'shhs1-dataset-0.14.0.csv')
        # meta_df = pd.read_csv(metadata_path)
        # names = meta_df.columns
        # useful_names = [n for n in names if 'age' in n]
        # print(useful_names)
