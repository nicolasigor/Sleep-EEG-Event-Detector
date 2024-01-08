import numpy as np
from src.data import utils
from src.detection.predicted_dataset import PredictedDataset


def generate_ensemble_from_probabilities(dict_of_proba, reference_feeder_dataset, skip_setting_threshold=False):
    """
    dict_of_proba = {
        subject_id_1: list of probabilities to ensemble,
        subject_id_2: list of probabilities to ensemble,
        etc
    }
    """
    subject_ids = reference_feeder_dataset.get_ids()
    avg_dict = {}
    for subject_id in subject_ids:
        probabilities = np.stack(dict_of_proba[subject_id], axis=0).astype(np.float32).mean(axis=0).astype(np.float16)
        avg_dict[subject_id] = probabilities
    ensemble_prediction = PredictedDataset(
        dataset=reference_feeder_dataset,
        probabilities_dict=avg_dict,
        params=reference_feeder_dataset.params.copy(),
        skip_setting_threshold=skip_setting_threshold)
    return ensemble_prediction


def generate_ensemble_from_stamps(
        dict_of_stamps, reference_feeder_dataset, downsampling_factor=8, skip_setting_threshold=False):
    """
    dict_of_stamps = {
        subject_id_1: list of stamps to ensemble,
        subject_id_2: list of stamps to ensemble,
        etc
    }
    """
    subject_ids = reference_feeder_dataset.get_ids()
    dict_of_proba = {}
    for subject_id in subject_ids:
        stamps_list = dict_of_stamps[subject_id]
        subject_max_sample = np.max([
            (1 if single_stamp.size == 0 else single_stamp.max())
            for single_stamp in stamps_list])
        subject_max_sample = downsampling_factor * ((subject_max_sample // downsampling_factor) + 10)
        probabilities = [
            utils.stamp2seq(single_stamp, 0, subject_max_sample - 1).reshape(-1, downsampling_factor).mean(axis=1)
            for single_stamp in stamps_list]
        dict_of_proba[subject_id] = probabilities
    ensemble_prediction = generate_ensemble_from_probabilities(
        dict_of_proba, reference_feeder_dataset, skip_setting_threshold=skip_setting_threshold)
    return ensemble_prediction


def generate_ensemble_from_predicted_datasets(
        predicted_dataset_list,
        reference_feeder_dataset,
        use_probabilities=False,
        skip_setting_threshold=False
):
    subject_ids = reference_feeder_dataset.get_ids()
    dict_of_data = {}
    for subject_id in subject_ids:
        if use_probabilities:
            data_list = [
                pred.get_subject_probabilities(subject_id, return_adjusted=True)
                for pred in predicted_dataset_list]
        else:
            data_list = [
                pred.get_subject_stamps(subject_id)
                for pred in predicted_dataset_list]
        dict_of_data[subject_id] = data_list
    if use_probabilities:
        ensemble_prediction = generate_ensemble_from_probabilities(
            dict_of_data, reference_feeder_dataset, skip_setting_threshold=skip_setting_threshold)
    else:
        ensemble_prediction = generate_ensemble_from_stamps(
            dict_of_data, reference_feeder_dataset, skip_setting_threshold=skip_setting_threshold)
    return ensemble_prediction
