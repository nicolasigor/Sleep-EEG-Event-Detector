import numpy as np

from src.common import constants
from src.data import utils


def custom_linspace(start_value, end_value, step_value):
    n_points = int(np.round((end_value - start_value) / step_value))
    array = start_value + np.arange(n_points + 1) * step_value
    return array


def closest_index(single_value, array):
    return np.argmin((single_value - array) ** 2)


def get_splits_dict(dataset, seed_id_list, use_test_set=True, train_fraction=0.75):
    ids_dict = {}
    for k in seed_id_list:
        train_ids, val_ids = utils.split_ids_list_v2(
            dataset.train_ids, split_id=k, train_fraction=train_fraction)
        ids_dict[k] = {
            constants.TRAIN_SUBSET: train_ids,
            constants.VAL_SUBSET: val_ids}
        if use_test_set:
            ids_dict[k][constants.TEST_SUBSET] = dataset.test_ids
    return ids_dict
