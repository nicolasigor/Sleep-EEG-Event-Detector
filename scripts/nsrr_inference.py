"""Infers on the NSRR dataset using an ensemble of trained models"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import datetime
import json
import os
import pickle
from pprint import pprint
import sys
import time

# TF logging control
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np

project_root = os.path.abspath("..")
sys.path.append(project_root)

from sleeprnn.data import utils
from sleeprnn.detection import det_utils
from sleeprnn.nn.models import WaveletBLSTM
from sleeprnn.helpers.reader import load_dataset
from sleeprnn.common import constants
from sleeprnn.common import checks
from sleeprnn.common import pkeys
from sleeprnn.common.optimal_thresholds import OPTIMAL_THR_FOR_CKPT_DICT

RESULTS_PATH = os.path.join(project_root, "results")


def get_partitions(dataset, strategy, n_seeds):
    train_ids_list = []
    val_ids_list = []
    test_ids_list = []
    if strategy == "fixed":
        for fold_id in range(n_seeds):
            train_ids, val_ids = utils.split_ids_list_v2(
                dataset.train_ids, split_id=fold_id
            )
            train_ids_list.append(train_ids)
            val_ids_list.append(val_ids)
            test_ids_list.append(dataset.test_ids)
    elif strategy == "5cv":
        for cv_seed in range(n_seeds):
            for fold_id in range(5):
                train_ids, val_ids, test_ids = dataset.cv_split(5, fold_id, cv_seed)
                train_ids_list.append(train_ids)
                val_ids_list.append(val_ids)
                test_ids_list.append(test_ids)
    else:
        raise ValueError
    partitions = {"train": train_ids_list, "val": val_ids_list, "test": test_ids_list}
    return partitions


def get_standard_deviations(dataset, partitions):
    standard_deviations = {}
    n_folds = len(partitions["train"])
    for fold_id in range(n_folds):
        train_ids = partitions["train"][fold_id]
        val_ids = partitions["val"][fold_id]
        fold_global_std = dataset.compute_global_std(
            np.concatenate([train_ids, val_ids])
        )
        standard_deviations[fold_id] = fold_global_std
    return standard_deviations


def get_hash_name(config):
    return "%s_e%d" % (config["dataset_name"], config["which_expert"])


def get_parameters(train_path):
    fname = os.path.join(train_path, "params.json")
    with open(fname, "r") as handle:
        loaded_params = json.load(handle)
    params = copy.deepcopy(pkeys.default_params)
    params.update(loaded_params)
    return params


def get_model(train_path):
    params = get_parameters(train_path)
    # For postprocessing, we will allow more flexibility due to the demographics of NSRR
    # In particular, we will not restrict at this point the maximum duration, only the min duration and separation
    # Aftwards, it is easy to use a larger min duration or min separation, or to impose a max duration, in analysis
    params[pkeys.SS_MIN_SEPARATION] = 0.3
    params[pkeys.SS_MIN_DURATION] = 0.3
    params[pkeys.SS_MAX_DURATION] = None
    weight_ckpt_path = os.path.join(train_path, "model", "ckpt")
    print("Restoring weights from %s" % weight_ckpt_path)

    loaded_model = WaveletBLSTM(
        params=params, logdir=os.path.join(RESULTS_PATH, "tmp0")
    )
    loaded_model.load_checkpoint(weight_ckpt_path)

    return loaded_model


def get_configs_std(configs):
    configs_std = {}
    for config in configs:
        dataset = load_dataset(config["dataset_name"], verbose=False)
        partitions = get_partitions(dataset, config["strategy"], config["n_seeds"])
        stds = get_standard_deviations(dataset, partitions)
        configs_std[get_hash_name(config)] = stds
    return configs_std


def get_opt_thr_str(optimal_thr_list, ckpt_folder, grid_folder):
    seeds_best_thr_string = ", ".join(["%1.2f" % thr for thr in optimal_thr_list])
    str_to_register = "    os.path.join('%s', '%s'): [%s]," % (
        ckpt_folder,
        grid_folder,
        seeds_best_thr_string,
    )
    return str_to_register


if __name__ == "__main__":
    task_mode = constants.N2_RECORD
    this_date = "20210716"

    # Because the NSRR is big, here you can set the number of splits and the split id
    # to consider in this run. This is useful to run the script in parallel.
    # If you want to run the script in a single run, set n_splits = 1 and split_id = 0
    n_splits = 4
    split_id = 0

    # Prediction using the first 5 checkpoints of v2-time trained on MODA
    # Predictions are adjusted before ensembling, so ensemble always has opt thr 0.5

    source_configs = [
        dict(
            dataset_name=constants.MODA_SS_NAME,
            which_expert=1,
            strategy="5cv",
            n_seeds=1,
            ckpt_folder="20210529_thesis_indata_5cv_e1_n2_train_moda_ss",
        ),
    ]

    nsrr = load_dataset(constants.NSRR_SS_NAME, verbose=False)

    # ---- Folds of small samples
    subject_ids = nsrr.all_ids
    n_subjects = len(subject_ids)
    fold_size = 100
    n_folds_nsrr = int(np.ceil(n_subjects / fold_size))
    # Subjects are shuffled so that we can incrementally check the analysis by adding folds
    subject_ids = np.random.RandomState(seed=0).permutation(subject_ids)
    nsrr_folds = {}
    for nsrr_fold_id in range(n_folds_nsrr):
        start_subject = nsrr_fold_id * fold_size
        end_subject = start_subject + fold_size
        fold_subjects = subject_ids[start_subject:end_subject]
        nsrr_folds[nsrr_fold_id] = fold_subjects
    # ----

    print("Inference by folds:")
    nsrr_fold_ids = np.sort(list(nsrr_folds.keys()))
    for fold_id in nsrr_fold_ids:
        print("Fold %d: %d subjects" % (fold_id, len(nsrr_folds[fold_id])))
    # Check uniqueness
    all_subjects_check = np.concatenate(
        [nsrr_folds[fold_id] for fold_id in nsrr_fold_ids]
    )
    print("Check: Unique subjects: %d" % np.unique(all_subjects_check).size)

    # Only pick a subsample of folds
    nsrr_fold_ids = nsrr_fold_ids.reshape(n_splits, -1)[split_id]
    print("Inference on the following folds:", nsrr_fold_ids)

    configs_std = get_configs_std(source_configs)
    # example: fold_std = configs_std[hash_name][fold_id]
    for source_config in source_configs:
        print(
            "\nNSRR predicted using model trained on dataset %s-e%d"
            % (source_config["dataset_name"], source_config["which_expert"])
        )
        source_dataset = load_dataset(source_config["dataset_name"], verbose=False)
        partitions = get_partitions(
            source_dataset, source_config["strategy"], source_config["n_seeds"]
        )
        n_folds = len(partitions["test"])
        # example: ids = partitions[set_name][fold_id]

        experiment_name = "%s_from_%s_ensemble_to" % (
            this_date,
            source_config["ckpt_folder"],
        )
        experiment_name_full = "%s_e%d_%s_train_%s" % (
            experiment_name,
            1,
            task_mode,
            nsrr.dataset_name,
        )

        grid_folder_list = os.listdir(
            os.path.join(RESULTS_PATH, source_config["ckpt_folder"])
        )
        grid_folder = [g for g in grid_folder_list if (constants.V2_TIME in g)][
            0
        ]  # Only REDv2-Time
        print("\nProcessing grid folder %s" % grid_folder)
        grid_folder_complete = os.path.join(source_config["ckpt_folder"], grid_folder)
        optimal_thr_list = OPTIMAL_THR_FOR_CKPT_DICT[grid_folder_complete]

        # Now loop through NSRR folds
        for nsrr_fold_id in nsrr_fold_ids:
            nsrr_subjects = nsrr_folds[nsrr_fold_id]
            print(
                "\nProcessing NSRR fold %d with %d subjects"
                % (nsrr_fold_id, len(nsrr_subjects))
            )

            # Save path for predictions
            base_dir = os.path.join(
                experiment_name_full, grid_folder, "fold%d" % nsrr_fold_id
            )
            save_dir = os.path.abspath(
                os.path.join(
                    RESULTS_PATH, "predictions_%s" % nsrr.dataset_name, base_dir
                )
            )

            checks.ensure_directory(save_dir)

            # Inside this fold, loop through checkpoints to ensemble
            start_time = time.time()
            probabilities_by_subject = {s: [] for s in nsrr_subjects}
            for source_fold_id in range(n_folds):
                print("Processing model from source fold %d. " % source_fold_id)
                fold_opt_thr = optimal_thr_list[source_fold_id]
                # Set appropriate global STD
                nsrr.global_std = configs_std[get_hash_name(source_config)][
                    source_fold_id
                ]
                # Retrieve appropriate model
                source_path = os.path.join(
                    RESULTS_PATH, grid_folder_complete, "fold%d" % source_fold_id
                )
                model = get_model(source_path)
                page_size = model.get_page_size()
                border_size = model.get_border_size()
                total_border = page_size // 2 + border_size
                # --- Predict
                print("Starting prediction on %d subjects" % len(nsrr_subjects))
                for subject_id in nsrr_subjects:
                    x = nsrr.get_subject_signal(
                        subject_id, normalize_clip=True, verbose=False
                    )
                    y = model.predict_proba_from_vector(x, with_augmented_page=True)
                    y_adjusted = det_utils.transform_predicted_proba_to_adjusted_proba(
                        y, fold_opt_thr
                    )
                    probabilities_by_subject[subject_id].append(y_adjusted)
                print("E.T. %1.4f [s]" % (time.time() - start_time))
            # Ensemble
            ensemble_by_subject = {}
            ens_sizes = []
            for subject_id in nsrr_subjects:
                subject_probas = probabilities_by_subject[subject_id]
                ens_sizes.append(len(subject_probas))
                subject_average = (
                    np.stack(subject_probas, axis=0)
                    .astype(np.float32)
                    .mean(axis=0)
                    .astype(np.float16)
                )
                ensemble_by_subject[subject_id] = subject_average
            ens_sizes = np.unique(ens_sizes)
            print("Check: Size of ensembles:", ens_sizes)
            # Save ensembles
            filename = os.path.join(
                save_dir, "prediction_%s_%s.pkl" % (task_mode, constants.TEST_SUBSET)
            )
            with open(filename, "wb") as handle:
                pickle.dump(
                    ensemble_by_subject, handle, protocol=pickle.HIGHEST_PROTOCOL
                )

            print("Ensembled predictions saved at %s" % save_dir)
            print("")
