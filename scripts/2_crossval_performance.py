"""Computes crossval performance while also fits detection threshold of model.

Always run this script after training a model, otherwise you will not have the
best performing output threshold for the model.

In the "Experiment settings" block of this script, set the location of the training run.

In the "Data settings" block, you should match what was used in the training run.
The "n2_subsampling" flag should be set to True if the model was trained with
N2 subsampling (only relevant for fine-tuning experiments).

In the "Evaluation settings", you can control how to compute the metrics.
Most likely, you want to evaluate on the test set, fit on the combined train+val set
(it's more stable, the fit refers to the fitting of the output threshold).
Also, most of the time the "average mode" should be set to "macro average", which
means averaging first within each subject. "micro average" combines all subjects
before reducing, so it's useful when per-subject data is too small. For the datasets
used in the paper, we only used "micro average" for the MODA dataset.

After the script runs, it will print both the performance and the best threshold
per fold and grid setting. For ease of use, copy the best threshold list
to the sleeprnn.common.optimal_thresholds.OPTIMAL_THR_FOR_CKPT_DICT dictionary,
so that it can be easily imported in other scripts.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from pprint import pprint

import numpy as np

project_root = os.path.abspath("..")
sys.path.append(project_root)

from sleeprnn.helpers import reader
from sleeprnn.data import utils
from sleeprnn.detection.feeder_dataset import FeederDataset
from sleeprnn.detection import metrics
from sleeprnn.detection.threshold_optimization import fit_threshold
from sleeprnn.common import constants, pkeys

RESULTS_PATH = os.path.join(project_root, "results")


def get_subsampling_factor(
    grid_folder, subsampling_str_prefix="signalsize", subsampling_str_is_percentage=True
):
    grid_folder = grid_folder.split("_")
    data = ""
    for s in grid_folder:
        if subsampling_str_prefix in s:
            data = s
    data = float(data.split(subsampling_str_prefix)[-1])
    if subsampling_str_is_percentage:
        data = data / 100
    return data


if __name__ == "__main__":
    # Experiment setting
    # Indicate the folder inside /results/ to be used.
    # Note that here you only need to provide the 'prefix', since we build the checkpoint folder name as:
    # ckpt_folder = '%s_%s_train_%s' % (ckpt_folder_prefix, task_mode, dataset_name)
    ckpt_folder_prefix = "20210529_thesis_indata_5cv_e1"
    # You may specify certain runs within that ckpt_folder in grid_folder_list.
    # If None then all runs are used
    grid_folder_list = ["v2_time"]

    # Data settings
    dataset_name = constants.MASS_SS_NAME
    which_expert = 1
    task_mode = constants.N2_RECORD
    dataset_params = {pkeys.FS: 200}
    load_dataset_from_ckpt = True
    n2_subsampling = False

    # Evaluation settings
    evaluation_sets = [constants.TEST_SUBSET]
    fitting_sets = [constants.TRAIN_SUBSET, constants.VAL_SUBSET]
    evaluation_average_mode = constants.MACRO_AVERAGE
    fitting_average_mode = constants.MACRO_AVERAGE
    threshold_space = {"min": 0.2, "max": 0.8, "step": 0.02}
    iou_threshold_report = 0.2

    # -----------------------------------------------------------
    # -----------------------------------------------------------

    print(
        "Cross-validation performance evaluation (fitting %s, evaluation %s)"
        % (fitting_average_mode, evaluation_average_mode)
    )
    print("Fitting in %s and evaluating in %s" % (fitting_sets, evaluation_sets))
    print("Predictions loaded from PredictedDataset objects in pickle files")
    print(
        "Threshold space: %s:%s:%s"
        % (threshold_space["min"], threshold_space["step"], threshold_space["max"])
    )
    # Load data
    dataset = reader.load_dataset(
        dataset_name,
        params=dataset_params,
        load_checkpoint=load_dataset_from_ckpt,
        verbose=False,
    )
    print("Loaded: dataset %s at %s Hz" % (dataset.dataset_name, dataset.fs))
    # Load paths
    ckpt_folder = "%s_%s_train_%s" % (ckpt_folder_prefix, task_mode, dataset_name)
    if grid_folder_list is None:
        experiment_path = os.path.join(
            RESULTS_PATH, "predictions_%s" % dataset_name, ckpt_folder
        )
        grid_folder_list = os.listdir(experiment_path)
        grid_folder_list = [
            f
            for f in grid_folder_list
            if os.path.isdir(
                os.path.join(
                    RESULTS_PATH, "predictions_%s" % dataset_name, ckpt_folder, f
                )
            )
        ]
        grid_folder_list.sort()
    print("Grid settings to be evaluated from %s:" % ckpt_folder)
    pprint(grid_folder_list)
    if n2_subsampling:
        print("N2 subsampling enabled")
        subsampling_factor_list = [get_subsampling_factor(n) for n in grid_folder_list]
        for g, s in zip(grid_folder_list, subsampling_factor_list):
            print("Grid %s with subsampling %s" % (g, s))
    else:
        subsampling_factor_list = [1.0] * len(grid_folder_list)
    # Load predictions
    predictions_dict = {}
    for grid_folder in grid_folder_list:
        grid_folder_complete = os.path.join(ckpt_folder, grid_folder)
        predictions_dict[grid_folder] = reader.read_predictions_crossval(
            grid_folder_complete, dataset, task_mode
        )
    print("Predictions loaded.\n")
    # Useful lists
    fold_ids = list(predictions_dict[grid_folder_list[0]].keys())
    fold_ids.sort()
    fold_ids_imputed = reader.generate_imputed_ids(fold_ids)

    # Search optimum threshold
    fitted_thresholds = {}
    for i_grid, grid_folder in enumerate(grid_folder_list):
        print("Evaluating grid: %s" % grid_folder, flush=True)
        fitted_thresholds[grid_folder] = {}
        for k in fold_ids:
            print("    Fold %d" % k)
            fold_predictions = predictions_dict[grid_folder][k]
            ids_dict = {
                set_name: fold_predictions[set_name].get_ids()
                for set_name in fitting_sets
            }
            feeder_dataset_list = [
                FeederDataset(
                    dataset,
                    ids_dict[set_name],
                    task_mode,
                    which_expert=which_expert,
                    n2_subsampling_factor=subsampling_factor_list[i_grid],
                )
                for set_name in fitting_sets
            ]
            predicted_dataset_list = [
                predictions_dict[grid_folder][k][set_name] for set_name in fitting_sets
            ]
            fitted_thresholds[grid_folder][k] = fit_threshold(
                feeder_dataset_list,
                predicted_dataset_list,
                threshold_space,
                fitting_average_mode,
            )

    # Report results with metrics
    # In test sets, predictions are always complete (full N2 pages)
    print(
        "\nVal AF1 report (iou >= %1.2f, from %s-%s to %s-%s) for %s"
        % (
            iou_threshold_report,
            fitting_sets,
            fitting_average_mode,
            evaluation_sets,
            evaluation_average_mode,
            ckpt_folder,
        )
    )
    average_metric_fn_dict = {
        constants.MACRO_AVERAGE: metrics.average_metric_macro_average,
        constants.MICRO_AVERAGE: metrics.average_metric_micro_average,
    }
    metric_vs_iou_fn_dict = {
        constants.MACRO_AVERAGE: metrics.metric_vs_iou_macro_average,
        constants.MICRO_AVERAGE: metrics.metric_vs_iou_micro_average,
    }
    metric_to_sort_list = []
    str_to_show_list = []
    str_to_register_list = []
    for grid_folder in grid_folder_list:
        outputs = {
            "af1_half": [],
            "af1_best": [],
            "f1_best": [],
            "prec_best": [],
            "rec_best": [],
        }
        for k in fold_ids:
            # Retrieve relevant data
            fold_predictions = predictions_dict[grid_folder][k]
            ids_dict = {
                set_name: fold_predictions[set_name].get_ids()
                for set_name in evaluation_sets
            }
            events_list = []
            for set_name in evaluation_sets:
                feed_d = FeederDataset(
                    dataset, ids_dict[set_name], task_mode, which_expert=which_expert
                )
                events_list = events_list + feed_d.get_stamps()
            # Half threshold
            detections_list = []
            for set_name in evaluation_sets:
                pred_d = fold_predictions[set_name]
                pred_d.set_probability_threshold(0.5)
                detections_list = detections_list + pred_d.get_stamps()
            af1_half = average_metric_fn_dict[evaluation_average_mode](
                events_list, detections_list
            )
            outputs["af1_half"].append(af1_half)
            # Optimal threshold
            best_thr = fitted_thresholds[grid_folder][k]
            detections_list = []
            for set_name in evaluation_sets:
                pred_d = fold_predictions[set_name]
                pred_d.set_probability_threshold(best_thr)
                detections_list = detections_list + pred_d.get_stamps()
            iou_matching_list, _ = metrics.matching_with_list(
                events_list, detections_list
            )
            af1_best = average_metric_fn_dict[evaluation_average_mode](
                events_list, detections_list, iou_matching_list=iou_matching_list
            )
            outputs["af1_best"].append(af1_best)
            f1_best = metric_vs_iou_fn_dict[evaluation_average_mode](
                events_list,
                detections_list,
                [iou_threshold_report],
                iou_matching_list=iou_matching_list,
            )
            outputs["f1_best"].append(f1_best)
            # Precision and recall
            if evaluation_average_mode == constants.MICRO_AVERAGE:
                precision_best = metrics.metric_vs_iou_micro_average(
                    events_list,
                    detections_list,
                    [iou_threshold_report],
                    metric_name=constants.PRECISION,
                    iou_matching_list=iou_matching_list,
                )
                recall_best = metrics.metric_vs_iou_micro_average(
                    events_list,
                    detections_list,
                    [iou_threshold_report],
                    metric_name=constants.RECALL,
                    iou_matching_list=iou_matching_list,
                )
            elif evaluation_average_mode == constants.MACRO_AVERAGE:
                # Here I will report by-subject std to track inter-subject dispersion
                precision_best = metrics.metric_vs_iou_macro_average(
                    events_list,
                    detections_list,
                    [iou_threshold_report],
                    metric_name=constants.PRECISION,
                    iou_matching_list=iou_matching_list,
                    collapse_values=False,
                )
                recall_best = metrics.metric_vs_iou_macro_average(
                    events_list,
                    detections_list,
                    [iou_threshold_report],
                    metric_name=constants.RECALL,
                    iou_matching_list=iou_matching_list,
                    collapse_values=False,
                )
            else:
                raise ValueError("Average mode '%s' invalid" % evaluation_average_mode)
            outputs["prec_best"].append(precision_best)
            outputs["rec_best"].append(recall_best)
        report_dict = {}
        for key in outputs:
            report_dict[key] = {
                "mean": 100 * np.mean(outputs[key]),
                "std": 100 * np.std(outputs[key]),
            }
        seeds_best_thr_string = ", ".join(
            [
                "None" if k is None else "%1.2f" % fitted_thresholds[grid_folder][k]
                for k in fold_ids_imputed
            ]
        )
        str_to_show = (
            "AF1 %1.2f/%1.2f [0.5] "
            "%1.2f/%1.2f [%s], "
            "F %1.2f/%1.2f, "
            "P %1.1f/%s, "
            "R %1.1f/%s for %s"
            % (
                report_dict["af1_half"]["mean"],
                report_dict["af1_half"]["std"],
                report_dict["af1_best"]["mean"],
                report_dict["af1_best"]["std"],
                seeds_best_thr_string,
                report_dict["f1_best"]["mean"],
                report_dict["f1_best"]["std"],
                report_dict["prec_best"]["mean"],
                ("%1.1f" % report_dict["prec_best"]["std"]).rjust(4),
                report_dict["rec_best"]["mean"],
                ("%1.1f" % report_dict["rec_best"]["std"]).rjust(4),
                grid_folder,
            )
        )
        str_to_register = "    os.path.join('%s', '%s'): [%s]," % (
            ckpt_folder,
            grid_folder,
            seeds_best_thr_string,
        )
        metric_to_sort_list.append(report_dict["af1_best"]["mean"])
        str_to_show_list.append(str_to_show)
        str_to_register_list.append(str_to_register)

    # Sort by descending order
    idx_sorted = np.argsort(-np.asarray(metric_to_sort_list))
    str_to_show_list = [str_to_show_list[i] for i in idx_sorted]
    str_to_register_list = [str_to_register_list[i] for i in idx_sorted]
    for str_to_show in str_to_show_list:
        print(str_to_show)
    print("")
    for str_to_show in str_to_register_list:
        print(str_to_show)
