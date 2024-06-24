import numpy as np

from sleeprnn.common import constants
from sleeprnn.detection import metrics
from sleeprnn.detection.feeder_dataset import FeederDataset
from sleeprnn.helpers import misc


def performance_vs_iou_with_seeds(
    dataset,
    predictions_dict,
    optimal_thr_list,
    iou_curve_axis,
    iou_hist_bins,
    task_mode,
    which_expert,
    set_name=constants.TEST_SUBSET,
):
    # Seeds
    seed_id_list = list(predictions_dict.keys())
    seed_id_list.sort()
    ids_dict = misc.get_splits_dict(dataset, seed_id_list)
    # Performance
    tmp_f1_vs_iou = []
    tmp_recall_vs_iou = []
    tmp_precision_vs_iou = []
    tmp_mean_af1 = []
    tmp_mean_iou = []
    tmp_iqr_low_iou = []
    tmp_iqr_high_iou = []
    tmp_iou_hist = []
    for k in seed_id_list:
        # Expert
        subset_data = FeederDataset(
            dataset, ids_dict[k][set_name], task_mode, which_expert=which_expert
        )
        events = subset_data.get_stamps()
        # Model
        prediction_data = predictions_dict[k][set_name]
        prediction_data.set_probability_threshold(optimal_thr_list[k])
        detections = prediction_data.get_stamps()
        # Measure stuff
        results = performance_vs_iou(events, detections, iou_curve_axis, iou_hist_bins)
        tmp_f1_vs_iou.append(results[constants.F1_VS_IOU])
        tmp_recall_vs_iou.append(results[constants.RECALL_VS_IOU])
        tmp_precision_vs_iou.append(results[constants.PRECISION_VS_IOU])
        tmp_mean_af1.append(results[constants.MEAN_AF1])
        tmp_mean_iou.append(results[constants.MEAN_IOU])
        tmp_iqr_low_iou.append(results[constants.IQR_LOW_IOU])
        tmp_iqr_high_iou.append(results[constants.IQR_HIGH_IOU])
        tmp_iou_hist.append(results[constants.IOU_HIST_VALUES])
    tmp_f1_vs_iou = np.stack(tmp_f1_vs_iou, axis=0)
    tmp_recall_vs_iou = np.stack(tmp_recall_vs_iou, axis=0)
    tmp_precision_vs_iou = np.stack(tmp_precision_vs_iou, axis=0)
    tmp_mean_af1 = np.stack(tmp_mean_af1, axis=0)
    tmp_mean_iou = np.stack(tmp_mean_iou, axis=0)
    tmp_iqr_low_iou = np.stack(tmp_iqr_low_iou, axis=0)
    tmp_iqr_high_iou = np.stack(tmp_iqr_high_iou, axis=0)
    tmp_iou_hist = np.stack(tmp_iou_hist, axis=0)
    model_data_dict = {
        constants.F1_VS_IOU: tmp_f1_vs_iou,
        constants.RECALL_VS_IOU: tmp_recall_vs_iou,
        constants.PRECISION_VS_IOU: tmp_precision_vs_iou,
        constants.IOU_HIST_BINS: iou_hist_bins,
        constants.IOU_CURVE_AXIS: iou_curve_axis,
        constants.IOU_HIST_VALUES: tmp_iou_hist,
        constants.MEAN_IOU: tmp_mean_iou,
        constants.MEAN_AF1: tmp_mean_af1,
        constants.IQR_LOW_IOU: tmp_iqr_low_iou,
        constants.IQR_HIGH_IOU: tmp_iqr_high_iou,
    }
    return model_data_dict


def performance_vs_iou(events, detections, iou_curve_axis, iou_hist_bins):
    # Matching
    iou_matchings, idx_matchings = metrics.matching_with_list(events, detections)
    # Measure stuff
    seed_f1_vs_iou = metrics.metric_vs_iou_with_list(
        events,
        detections,
        iou_curve_axis,
        iou_matching_list=iou_matchings,
        metric_name=constants.F1_SCORE,
    )
    seed_recall_vs_iou = metrics.metric_vs_iou_with_list(
        events,
        detections,
        iou_curve_axis,
        iou_matching_list=iou_matchings,
        metric_name=constants.RECALL,
    )
    seed_precision_vs_iou = metrics.metric_vs_iou_with_list(
        events,
        detections,
        iou_curve_axis,
        iou_matching_list=iou_matchings,
        metric_name=constants.PRECISION,
    )

    seed_mean_af1 = metrics.average_metric_with_list(
        events, detections, iou_matching_list=iou_matchings
    )
    seed_mean_iou = []
    seed_iou_hist = []
    seed_iqr_low_iou = []
    seed_iqr_high_iou = []
    for i in range(len(events)):
        iou_nonzero = iou_matchings[i][idx_matchings[i] > -1]
        iou_mean = np.mean(iou_nonzero)
        iou_low_iqr = np.percentile(iou_nonzero, 25)
        iou_high_iqr = np.percentile(iou_nonzero, 75)
        iou_hist, _ = np.histogram(iou_nonzero, bins=iou_hist_bins, density=True)
        seed_mean_iou.append(iou_mean)
        seed_iqr_low_iou.append(iou_low_iqr)
        seed_iqr_high_iou.append(iou_high_iqr)
        seed_iou_hist.append(iou_hist)
    seed_mean_iou = np.stack(seed_mean_iou, axis=0).mean(axis=0)
    seed_iqr_low_iou = np.stack(seed_iqr_low_iou, axis=0).mean(axis=0)
    seed_iqr_high_iou = np.stack(seed_iqr_high_iou, axis=0).mean(axis=0)
    seed_iou_hist = np.stack(seed_iou_hist, axis=0).mean(axis=0)
    results = {
        constants.F1_VS_IOU: seed_f1_vs_iou,
        constants.RECALL_VS_IOU: seed_recall_vs_iou,
        constants.PRECISION_VS_IOU: seed_precision_vs_iou,
        constants.IOU_HIST_BINS: iou_hist_bins,
        constants.IOU_CURVE_AXIS: iou_curve_axis,
        constants.IOU_HIST_VALUES: seed_iou_hist,
        constants.MEAN_IOU: seed_mean_iou,
        constants.MEAN_AF1: seed_mean_af1,
        constants.IQR_LOW_IOU: seed_iqr_low_iou,
        constants.IQR_HIGH_IOU: seed_iqr_high_iou,
    }
    return results


def duration_scatter_with_seeds(
    dataset,
    predictions_dict,
    optimal_thr_list,
    task_mode,
    which_expert,
    set_name=constants.TEST_SUBSET,
):
    # Seeds
    seed_id_list = list(predictions_dict.keys())
    seed_id_list.sort()
    ids_dict = misc.get_splits_dict(dataset, seed_id_list)
    # Performance
    results = {}
    for k in seed_id_list:
        # Expert
        subset_data = FeederDataset(
            dataset, ids_dict[k][set_name], task_mode, which_expert=which_expert
        )
        events = subset_data.get_stamps()
        # Model
        prediction_data = predictions_dict[k][set_name]
        prediction_data.set_probability_threshold(optimal_thr_list[k])
        detections = prediction_data.get_stamps()
        # Matching
        iou_matchings, idx_matchings = metrics.matching_with_list(events, detections)
        seed_matched_real_idx = []
        seed_matched_det_idx = []
        seed_matched_real_dur = []
        seed_matched_det_dur = []
        for i in range(len(events)):
            idx_matching = idx_matchings[i]
            matched_real_idx = np.where(idx_matching > -1)[0]
            matched_det_idx = idx_matching[idx_matching > -1]
            matched_real_event = events[i][matched_real_idx]
            matched_det_event = detections[i][matched_det_idx]
            matched_real_dur = matched_real_event[:, 1] - matched_real_event[:, 0]
            matched_det_dur = matched_det_event[:, 1] - matched_det_event[:, 0]
            seed_matched_real_idx.append(matched_real_idx)
            seed_matched_det_idx.append(matched_det_idx)
            seed_matched_real_dur.append(matched_real_dur)
            seed_matched_det_dur.append(matched_det_dur)
        results[k] = {
            "expert_idx": seed_matched_real_idx,
            "detection_idx": seed_matched_det_idx,
            "expert_duration": seed_matched_real_dur,
            "detection_duration": seed_matched_det_dur,
        }
    return results


def precision_recall_curve_with_seeds(
    dataset,
    predictions_dict,
    pr_curve_thr,
    iou_thr,
    task_mode,
    which_expert,
    set_name=constants.TEST_SUBSET,
):
    # Seeds
    seed_id_list = list(predictions_dict.keys())
    seed_id_list.sort()
    ids_dict = misc.get_splits_dict(dataset, seed_id_list)
    # Performance
    pr_curve = {}
    for k in seed_id_list:
        print("Processing seed %d" % k, flush=True)
        # Columns are [x: recall, y: precision]
        pr_curve[k] = {
            constants.RECALL: np.zeros(len(pr_curve_thr)),
            constants.PRECISION: np.zeros(len(pr_curve_thr)),
        }
        # Expert
        subset_data = FeederDataset(
            dataset, ids_dict[k][set_name], task_mode, which_expert=which_expert
        )
        events = subset_data.get_stamps()
        # Model
        prediction_data = predictions_dict[k][set_name]
        for i, thr in enumerate(pr_curve_thr):
            prediction_data.set_probability_threshold(thr)
            detections = prediction_data.get_stamps()
            # Measure stuff
            this_stats = [
                metrics.by_event_confusion(this_y, this_y_pred, iou_thr=iou_thr)
                for (this_y, this_y_pred) in zip(events, detections)
            ]
            this_recall = np.mean([m[constants.RECALL] for m in this_stats])
            this_precision = np.mean([m[constants.PRECISION] for m in this_stats])
            pr_curve[k][constants.RECALL][i] = this_recall
            pr_curve[k][constants.PRECISION][i] = this_precision
    return pr_curve
