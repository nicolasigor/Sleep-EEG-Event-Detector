from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import datetime
import itertools
import json
import os
import pickle
from pprint import pprint
import sys

# TF logging control
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

project_root = os.path.abspath('..')
sys.path.append(project_root)

from src.data import utils
from src.detection.feeder_dataset import FeederDataset
from src.nn.models import WaveletBLSTM
from src.helpers.reader import load_dataset
from src.common import constants
from src.common import checks
from src.common import pkeys

RESULTS_PATH = os.path.join(project_root, 'results')


if __name__ == '__main__':
    # ----- Experiment settings
    this_date = datetime.datetime.now().strftime("%Y%m%d")
    task_mode = constants.N2_RECORD
    description_str = 'experiments'
    experiment_name_base = '%s_standard_train' % this_date

    # Datasets
    dataset_configs = [
        # Internal, independent test
        {'name': constants.MASS_SS_NAME, 'expert': 1, 'strategy': 'fixed', 'n_seeds': 11},
        {'name': constants.MASS_SS_NAME, 'expert': 2, 'strategy': 'fixed', 'n_seeds': 11},
        {'name': constants.MASS_KC_NAME, 'expert': 1, 'strategy': 'fixed', 'n_seeds': 11},
        # External test
        {'name': constants.MODA_SS_NAME, 'expert': 1, 'strategy': '5cv', 'n_seeds': 3},
        # Internal, not independent test
        {'name': constants.MASS_SS_NAME, 'expert': 1, 'strategy': '5cv', 'n_seeds': 3},
        {'name': constants.MASS_SS_NAME, 'expert': 2, 'strategy': '5cv', 'n_seeds': 3},
        {'name': constants.MASS_KC_NAME, 'expert': 1, 'strategy': '5cv', 'n_seeds': 3},
    ]

    # Models
    model_configs = [
        {pkeys.MODEL_VERSION: constants.V2_TIME, pkeys.BORDER_DURATION: pkeys.DEFAULT_BORDER_DURATION_V2_TIME},
    ]

    # Default parameters with magnitudes in microvolts (uv)
    da_unif_noise_intens_uv = pkeys.DEFAULT_AUG_INDEP_UNIFORM_NOISE_INTENSITY_MICROVOLTS
    da_random_waves_map = {
        constants.SPINDLE: pkeys.DEFAULT_AUG_RANDOM_WAVES_PARAMS_SPINDLE,
        constants.KCOMPLEX: pkeys.DEFAULT_AUG_RANDOM_WAVES_PARAMS_KCOMPLEX}
    da_random_antiwaves_map = {
        constants.SPINDLE: pkeys.DEFAULT_AUG_RANDOM_ANTI_WAVES_PARAMS_SPINDLE,
        constants.KCOMPLEX: pkeys.DEFAULT_AUG_RANDOM_ANTI_WAVES_PARAMS_KCOMPLEX}

    for dataset_config in dataset_configs:
        dataset_name = dataset_config['name']
        which_expert = dataset_config['expert']
        strategy = dataset_config['strategy']
        experiment_name = '%s_%s_e%d' % (experiment_name_base, strategy, which_expert)
        print('\nModel training on %s_%s (marks %d)' % (dataset_name, task_mode, which_expert))
        dataset = load_dataset(dataset_name)
        print("Evaluation strategy:", strategy)
        train_ids_list = []
        val_ids_list = []
        test_ids_list = []
        if strategy == 'fixed':
            for fold_id in range(dataset_config['n_seeds']):
                train_ids, val_ids = utils.split_ids_list_v2(dataset.train_ids, split_id=fold_id)
                train_ids_list.append(train_ids)
                val_ids_list.append(val_ids)
                test_ids_list.append(dataset.test_ids)
        elif strategy == '5cv':
            for cv_seed in range(dataset_config['n_seeds']):
                for fold_id in range(5):
                    train_ids, val_ids, test_ids = dataset.cv_split(5, fold_id, cv_seed)
                    train_ids_list.append(train_ids)
                    val_ids_list.append(val_ids)
                    test_ids_list.append(test_ids)
        else:
            raise ValueError
        print("Partitions to evaluate")
        for i in range(len(train_ids_list)):
            print("Train %d, Val %d, Test %d" % (
                len(train_ids_list[i]),
                len(val_ids_list[i]),
                len(test_ids_list[i])
            ))
        print("Checking validation set")
        values, counts = np.unique(np.concatenate(val_ids_list), return_counts=True)
        counts_unique = np.unique(counts)
        print("Unique subjects %d, unique counts %s" % (values.size, counts_unique))
        print("Checking testing set")
        values, counts = np.unique(np.concatenate(test_ids_list), return_counts=True)
        counts_unique = np.unique(counts)
        print("Unique subjects %d, unique counts %s" % (values.size, counts_unique))

        for fold_id in range(len(train_ids_list)):

            print("\nStarting evaluation of partition %d (%d/%d)" % (fold_id, fold_id+1, len(train_ids_list)))
            train_ids = train_ids_list[fold_id]
            val_ids = val_ids_list[fold_id]
            test_ids = test_ids_list[fold_id]
            # Compute global std
            fold_global_std = dataset.compute_global_std(np.concatenate([train_ids, val_ids]))
            dataset.global_std = fold_global_std
            print("Global STD set to %s" % fold_global_std)
            # Create data feeders
            data_train = FeederDataset(dataset, train_ids, task_mode, which_expert=which_expert)
            data_val = FeederDataset(dataset, val_ids, task_mode, which_expert=which_expert)
            data_test = FeederDataset(dataset, test_ids, task_mode, which_expert=which_expert)
            # Create base parameters for this partition
            base_params = copy.deepcopy(pkeys.default_params)
            base_params[pkeys.AUG_INDEP_UNIFORM_NOISE_INTENSITY] = da_unif_noise_intens_uv / dataset.global_std
            da_random_waves = copy.deepcopy(da_random_waves_map[dataset.event_name])
            da_random_antiwaves = copy.deepcopy(da_random_antiwaves_map[dataset.event_name])
            for da_id in range(len(da_random_waves)):
                da_random_waves[da_id]['max_amplitude'] = da_random_waves[da_id]['max_amplitude_microvolts'] / dataset.global_std
                da_random_waves[da_id].pop('max_amplitude_microvolts')
            base_params[pkeys.AUG_RANDOM_WAVES_PARAMS] = da_random_waves
            base_params[pkeys.AUG_RANDOM_ANTI_WAVES_PARAMS] = da_random_antiwaves

            # Run models
            for model_config in model_configs:
                params = copy.deepcopy(base_params)
                params.update(model_config)
                folder_name = '%s' % model_config[pkeys.MODEL_VERSION]
                base_dir = os.path.join(
                    '%s_%s_train_%s' % (experiment_name, task_mode, dataset_name), folder_name, 'fold%d' % fold_id)
                # Path to save results of run
                logdir = os.path.join(RESULTS_PATH, base_dir)
                print('This run directory: %s' % logdir)

                # Create and train model
                model = WaveletBLSTM(params=params, logdir=logdir)
                model.fit(data_train, data_val, verbose=True)
                # --------------  Predict
                # Save path for predictions
                save_dir = os.path.abspath(os.path.join(
                    RESULTS_PATH, 'predictions_%s' % dataset_name, base_dir))
                checks.ensure_directory(save_dir)
                feeders_dict = {
                    constants.TRAIN_SUBSET: data_train,
                    constants.VAL_SUBSET: data_val,
                    constants.TEST_SUBSET: data_test
                }
                for set_name in feeders_dict.keys():
                    print('Predicting %s' % set_name, flush=True)
                    data_inference = feeders_dict[set_name]
                    prediction = model.predict_dataset(data_inference, verbose=True)
                    filename = os.path.join(
                        save_dir,
                        'prediction_%s_%s.pkl' % (task_mode, set_name))
                    with open(filename, 'wb') as handle:
                        pickle.dump(prediction, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print('Predictions saved at %s' % save_dir)
                print('')
