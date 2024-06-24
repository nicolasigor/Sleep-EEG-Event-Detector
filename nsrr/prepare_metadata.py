"""Generates preprocessed metadata (age and sex) from NSRR datasets.

It creates a csv file with columns subject_id, age and sex, one row per subject,
one file per dataset. This file allows analyzing the distribution of per-subject
sleep spindle features by sex and age.
"""

import os
import sys
from pprint import pprint

import numpy as np
import pandas as pd

project_root = ".."
sys.path.append(project_root)

DATASETS_PATH = os.path.join(project_root, 'resources', 'datasets', 'nsrr')


def prepare_suffix(subject_id, dataset_name):
    if dataset_name == 'mros1':
        return subject_id.lower()
    elif dataset_name == 'sof':
        return '%05d' % subject_id
    else:
        return str(subject_id)


if __name__ == "__main__":

    configs = [
        dict(
            dataset_name='shhs1',
            filename='shhs1-dataset-0.14.0.csv',
            subject_id='nsrrid',
            age='age_s1',
            sex='gender', sex_map={1: 'm', 2: 'f'},
            prefix='shhs1-'
        ),
        dict(
            dataset_name='mros1',
            filename='mros-visit1-dataset-0.5.0.csv',
            subject_id='nsrrid',
            age='vsage1',
            sex='gender', sex_map={2: 'm'},
            prefix='mros-visit1-'
        ),
        dict(
            dataset_name='chat1',
            filename='chat-baseline-dataset-0.11.0.csv',
            subject_id='nsrrid',
            age='ageyear_at_meas',
            sex='chi2', sex_map={1: 'm', 2: 'f'},
            prefix='chat-baseline-'
        ),
        dict(
            dataset_name='chat1',
            filename='chat-nonrandomized-dataset-0.11.0.csv',
            subject_id='nsrrid',
            age='age_nr',
            sex='ref9', sex_map={1: 'm', 2: 'f'},
            prefix='chat-baseline-nonrandomized-'
        ),
        dict(
            dataset_name='sof',
            filename='sof-visit-8-dataset-0.6.0.csv',
            subject_id='sofid',
            age='V8AGE',
            sex='gender', sex_map={1: 'f'},
            prefix='sof-visit-8-'
        ),
        dict(
            dataset_name='cfs',
            filename='cfs-visit5-dataset-0.5.0.csv',
            subject_id='nsrrid',
            age='age',
            sex='SEX', sex_map={0: 'f', 1: 'm'},
            prefix='cfs-visit5-'
        ),
        dict(
            dataset_name='ccshs',
            filename='ccshs-trec-dataset-0.6.0.csv',
            subject_id='nsrrid',
            age='age',
            sex='male', sex_map={0: 'f', 1: 'm'},
            prefix='ccshs-trec-'
        ),
    ]

    for config in configs:
        print("\nProcessing %s" % config['prefix'])
        metadata_file = os.path.join(DATASETS_PATH, config['dataset_name'], 'datasets', config['filename'])
        meta_df = pd.read_csv(metadata_file, low_memory=False)
        # subject id
        subject_ids = meta_df[config['subject_id']].values
        subject_ids = [prepare_suffix(sub_id, config['dataset_name']) for sub_id in subject_ids]
        subject_ids = np.array(['%s%s' % (config['prefix'], sub_id) for sub_id in subject_ids], dtype='<U40')

        # age
        ages = meta_df[config['age']].values.astype(np.float32)
        print(ages.dtype)
        print("Age NaNs: %d out of %d" % (np.isnan(ages).sum(), ages.size))
        print(ages[:5])

        # sex
        sexes = meta_df[config['sex']].values.astype(np.float32)
        print(sexes.dtype)
        print("Sex NaNs: %d out of %d" % (np.isnan(sexes).sum(), sexes.size))
        print(sexes[:5])

        # find nans and remove
        invalid_locs = np.where(np.isnan(ages) | np.isnan(sexes))[0]
        print("Subjects with at least one NaN:", invalid_locs.size)

        valid_locs = np.where((~np.isnan(ages)) & (~np.isnan(sexes)))[0]
        print("Subjects without NaN:", valid_locs.size)

        subject_ids = subject_ids[valid_locs]
        ages = ages[valid_locs]
        sexes = sexes[valid_locs]

        # Now change sex format
        sexes_str = [config['sex_map'][sex_value] for sex_value in sexes.astype(np.int32)]
        sexes_str = np.array(sexes_str)
        print(sexes_str[:10])

        # Check proportions
        female_fraction = np.mean(sexes_str == 'f')
        print("Females percentage %1.2f" % (female_fraction * 100))
        print("Mean age %1.2f" % np.mean(ages))

        # Create new table
        table = {
            'subject_id': subject_ids,
            'age': ages,
            'sex': sexes_str
        }
        table = pd.DataFrame.from_dict(table)

        # save
        save_fname = '%smetadata.csv' % config['prefix']
        save_dir = os.path.join(DATASETS_PATH, config['dataset_name'], save_fname)
        table.to_csv(save_dir, index=False)


    # Special fix for chat
    fname1 = os.path.join(DATASETS_PATH, 'chat1', 'datasets', 'chat-baseline-metadata.csv')
    fname2 = os.path.join(DATASETS_PATH, 'chat1', 'datasets', 'chat-baseline-nonrandomized-metadata.csv')
    table1 = pd.read_csv(fname1)
    table2 = pd.read_csv(fname2)
    table = pd.concat([table1, table2], ignore_index=True)
    # save
    save_fname = '%smetadata.csv' % 'chat-combined-'
    save_dir = os.path.join(DATASETS_PATH, 'chat1', save_fname)
    table.to_csv(save_dir, index=False)
