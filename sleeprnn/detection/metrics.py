"""metrics.py: Module for general evaluation metrics operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from sleeprnn.common import constants
from sleeprnn.data.utils import stamp2seq


def by_sample_confusion(events, detections, input_is_binary=False):
    """Returns a dictionary with by-sample metrics.
    If input_is_binary is true, the inputs are assumed to be binary sequences.
    If False, is assumed to be sample-stamps
    in ascending order.
    """
    # We need binary sequences here, so let's transform if that's not the case
    if not input_is_binary:
        last_sample = max(events[-1, 1], detections[-1, 1])
        events = stamp2seq(events, 0, last_sample)
        detections = stamp2seq(detections, 0, last_sample)
    tp = np.sum((events == 1) & (detections == 1))
    fp = np.sum((events == 0) & (detections == 1))
    fn = np.sum((events == 1) & (detections == 0))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    by_sample_metrics = {
        constants.TP: tp,
        constants.FP: fp,
        constants.FN: fn,
        constants.PRECISION: precision,
        constants.RECALL: recall,
        constants.F1_SCORE: f1_score
    }
    return by_sample_metrics


def by_sample_iou(events, detections, input_is_binary=False):
    """Returns the IoU considering the entire eeg as a single segmentation
    problem.
    If input_is_binary is true, the inputs are assumed to be binary sequences.
    If False, is assumed to be sample-stamps
    in ascending order.
    """
    # We need binary sequences here, so let's transform if that's not the case
    if not input_is_binary:
        last_sample = max(events[-1, 1], detections[-1, 1])
        events = stamp2seq(events, 0, last_sample)
        detections = stamp2seq(detections, 0, last_sample)
    intersection = np.sum((events == 1) & (detections == 1))
    sum_areas = np.sum(events) + np.sum(detections)
    union = sum_areas - intersection
    iou = intersection / union
    return iou


def by_event_confusion(events, detections, iou_thr=0.3, iou_matching=None):
    """Returns a dictionary with by-events metrics.
    events and detections are assumed to be sample-stamps, and to be in
    ascending order.
    iou_matching can be provided if it is already computed. If this is the case,
    events and detections are ignored.
    """
    if iou_matching is None:
        iou_matching, _ = matching(events, detections)
    n_detections = detections.shape[0]
    n_events = events.shape[0]
    mean_all_iou = np.mean(iou_matching)
    # First, remove the zero iou_array entries
    iou_matching = iou_matching[iou_matching > 0]
    if iou_matching.size > 0:
        mean_nonzero_iou = np.mean(iou_matching)
        # Now, give credit only for iou >= iou_thr
        tp = np.sum((iou_matching >= iou_thr).astype(int))
        fp = n_detections - tp
        fn = n_events - tp
        precision = tp / n_detections
        recall = tp / n_events

        # f1-score is 2 * precision * recall / (precision + recall),
        # but considering the case tp=0, a more stable formula is:
        f1_score = 2 * tp / (n_detections + n_events)

    else:
        # There are no positive iou -> no true positives
        mean_nonzero_iou = 0.0
        tp = 0
        fp = n_detections
        fn = n_events
        precision = 0
        recall = 0
        f1_score = 0

    by_event_metrics = {
        constants.TP: tp,
        constants.FP: fp,
        constants.FN: fn,
        constants.PRECISION: precision,
        constants.RECALL: recall,
        constants.F1_SCORE: f1_score,
        constants.MEAN_ALL_IOU: mean_all_iou,
        constants.MEAN_NONZERO_IOU: mean_nonzero_iou
    }
    return by_event_metrics


def matching(events, detections):
    """Returns the IoU associated with each event. Events that has no detections
    have IoU zero. events and detections are assumed to be sample-stamps, and to
    be in ascending order."""

    # Matrix of overlap, rows are events, columns are detections
    n_det = detections.shape[0]
    n_gs = events.shape[0]

    if n_det == 0:
        # There are no detections
        iou_matching = np.zeros(n_gs)
        idx_matching = -1 * np.ones(n_gs, dtype=np.int32)
        return iou_matching, idx_matching

    overlaps = np.zeros((n_gs, n_det))
    for i in range(n_gs):
        candidates = np.where(
            (detections[:, 0] <= events[i, 1])
            & (detections[:, 1] >= events[i, 0]))[0]
        for j in candidates:
            intersection = min(
                events[i, 1], detections[j, 1]
            ) - max(
                events[i, 0], detections[j, 0]
            ) + 1
            if intersection > 0:
                union = max(
                    events[i, 1], detections[j, 1]
                ) - min(
                    events[i, 0], detections[j, 0]
                ) + 1
                overlaps[i, j] = intersection / union
    # Greedy matching
    iou_matching = []  # Array for IoU for every true event (gs)
    idx_matching = []  # Array for the index associated with the true event.
    # If no detection is found, this value is -1
    for i in range(n_gs):
        if np.sum(overlaps[i, :]) > 0:
            # Find max overlap
            max_j = np.argmax(overlaps[i, :])
            iou_matching.append(overlaps[i, max_j])
            idx_matching.append(max_j)
            # Remove this detection for further search
            overlaps[:, max_j] = 0
        else:
            iou_matching.append(0)
            idx_matching.append(-1)
    iou_matching = np.array(iou_matching)
    idx_matching = np.array(idx_matching)
    return iou_matching, idx_matching


def matching_with_list(events_list, detections_list):
    iou_matching_list = []
    idx_matching_list = []
    for events, detections in zip(events_list, detections_list):
        iou_matching, idx_matching = matching(events, detections)
        iou_matching_list.append(iou_matching)
        idx_matching_list.append(idx_matching)
    return iou_matching_list, idx_matching_list


def confusion_vs_iou(
        events,
        detections,
        iou_thr_list,
        iou_matching=None
):
    if iou_matching is None:
        iou_matching, _ = matching(events, detections)
    n_events = events.shape[0]
    n_detections = detections.shape[0]
    # First, remove the zero iou_array entries
    iou_matching = iou_matching[iou_matching > 0]
    if iou_matching.size > 0:
        tp_vs_iou = [np.sum(iou_matching >= iou_thr) for iou_thr in iou_thr_list]
        fp_vs_iou = [(n_detections - tp) for tp in tp_vs_iou]
        fn_vs_iou = [(n_events - tp) for tp in tp_vs_iou]
    else:
        # There are no positive iou -> no true positives
        n_thrs = len(iou_thr_list)
        tp_vs_iou = n_thrs * [0]
        fp_vs_iou = n_thrs * [n_detections]
        fn_vs_iou = n_thrs * [n_events]
    metrics = {
        'tp_vs_iou': tp_vs_iou,
        'fp_vs_iou': fp_vs_iou,
        'fn_vs_iou': fn_vs_iou,
    }
    return metrics


def confusion_vs_iou_with_list(
        events_list,
        detections_list,
        iou_thr_list,
        iou_matching_list=None
):
    if iou_matching_list is None:
        iou_matching_list = [None] * len(events_list)
    outputs_list = [
        confusion_vs_iou(events, detections, iou_thr_list, iou_matching)
        for (events, detections, iou_matching)
        in zip(events_list, detections_list, iou_matching_list)
    ]
    metrics = {}
    for key in outputs_list[0].keys():
        metrics[key] = np.stack([o[key] for o in outputs_list], axis=0)
    return metrics


def metric_vs_iou_micro_average(
        events_list,
        detections_list,
        iou_thr_list,
        metric_name=constants.F1_SCORE,
        iou_matching_list=None
):
    """Aggregates TP, FP and FN from all subjects and then computes a single metric."""
    metrics = confusion_vs_iou_with_list(events_list, detections_list, iou_thr_list, iou_matching_list)
    tp_vs_iou = metrics['tp_vs_iou'].sum(axis=0)
    edge_case = np.array([1.0] * len(tp_vs_iou))
    n_events = np.sum([e.shape[0] for e in events_list])
    n_detections = np.sum([d.shape[0] for d in detections_list])
    recall_vs_iou = edge_case if n_events == 0 else (tp_vs_iou / n_events)
    precision_vs_iou = edge_case if n_detections == 0 else (tp_vs_iou / n_detections)
    f1_score_vs_iou = edge_case if (n_events + n_detections) == 0 else (2 * tp_vs_iou / (n_events + n_detections))
    if metric_name == constants.RECALL:
        chosen_vs_iou = recall_vs_iou
    elif metric_name == constants.PRECISION:
        chosen_vs_iou = precision_vs_iou
    elif metric_name == constants.F1_SCORE:
        chosen_vs_iou = f1_score_vs_iou
    else:
        raise ValueError("Metric name '%s' unsupported" % metric_name)
    return chosen_vs_iou


def metric_vs_iou_macro_average(
        events_list,
        detections_list,
        iou_thr_list,
        metric_name=constants.F1_SCORE,
        iou_matching_list=None,
        collapse_values=True
):
    """ Computes metric independently for each subject and then combines them."""
    if iou_matching_list is None:
        iou_matching_list = [None] * len(events_list)
    chosen_vs_iou_list = [
        metric_vs_iou_micro_average(
            [events],
            [detections],
            iou_thr_list,
            metric_name,
            [iou_matching])
        for (events, detections, iou_matching)
        in zip(events_list, detections_list, iou_matching_list)
    ]
    chosen_vs_iou_list = np.stack(chosen_vs_iou_list, axis=0)
    if collapse_values:
        return chosen_vs_iou_list.mean(axis=0)
    else:
        return chosen_vs_iou_list


def average_metric_micro_average(
        events_list,
        detections_list,
        metric_name=constants.F1_SCORE,
        iou_matching_list=None
):
    # Go through several IoU values
    first_iou = 0
    last_iou = 1
    res_iou = 0.01
    n_points = int(np.round((last_iou - first_iou) / res_iou))
    full_iou_list = np.arange(n_points + 1) * res_iou + first_iou
    chosen_vs_iou = metric_vs_iou_micro_average(
        events_list, detections_list, full_iou_list, metric_name, iou_matching_list)
    # To compute the area under the curve, we'll use trapezoidal approximation
    # So we need to divide by 2 the extremes
    chosen_vs_iou[0] = chosen_vs_iou[0] / 2
    chosen_vs_iou[-1] = chosen_vs_iou[-1] / 2
    # And now we compute the AUC
    avg_metric = np.sum(chosen_vs_iou * res_iou)
    return avg_metric


def average_metric_macro_average(
        events_list,
        detections_list,
        metric_name=constants.F1_SCORE,
        iou_matching_list=None,
        collapse_values=True
):
    # Go through several IoU values
    first_iou = 0
    last_iou = 1
    res_iou = 0.01
    n_points = int(np.round((last_iou - first_iou) / res_iou))
    full_iou_list = np.arange(n_points + 1) * res_iou + first_iou
    chosen_vs_iou = metric_vs_iou_macro_average(
        events_list, detections_list, full_iou_list, metric_name, iou_matching_list,
        collapse_values=collapse_values)
    # To compute the area under the curve, we'll use trapezoidal approximation
    # So we need to divide by 2 the extremes
    chosen_vs_iou[..., 0] = chosen_vs_iou[..., 0] / 2
    chosen_vs_iou[..., -1] = chosen_vs_iou[..., -1] / 2
    # And now we compute the AUC
    avg_metric = np.sum(chosen_vs_iou * res_iou, axis=-1)
    return avg_metric


def metric_vs_iou(
        events,
        detections,
        iou_thr_list,
        metric_name=constants.F1_SCORE,
        iou_matching=None,
        verbose=False
):
    metric_list = []
    if verbose:
        print('Matching events... ', end='', flush=True)
    if iou_matching is None:
        iou_matching, _ = matching(events, detections)
    if verbose:
        print('Done', flush=True)
    for iou_thr in iou_thr_list:
        if verbose:
            print(
                'Processing IoU threshold %1.4f... ' % iou_thr,
                end='', flush=True)
        this_stat = by_event_confusion(
            events, detections,
            iou_thr=iou_thr, iou_matching=iou_matching)
        metric = this_stat[metric_name]
        metric_list.append(metric)
        if verbose:
            print('%s obtained: %1.4f' % (metric_name, metric), flush=True)
    if verbose:
        print('Done')
    metric_list = np.array(metric_list)
    return metric_list


def metric_vs_iou_with_list(
        events_list,
        detections_list,
        iou_thr_list,
        metric_name=constants.F1_SCORE,
        iou_matching_list=None,
        verbose=False
):
    if iou_matching_list is None:
        iou_matching_list = [None] * len(events_list)

    all_metric_list = [
        metric_vs_iou(
            events, detections, iou_thr_list,
            metric_name=metric_name,
            verbose=verbose,
            iou_matching=iou_matching
        )
        for (events, detections, iou_matching)
        in zip(events_list, detections_list, iou_matching_list)
    ]
    all_metric_curve = np.stack(all_metric_list, axis=1).mean(axis=1)
    return all_metric_curve


def average_metric(
        events,
        detections,
        metric_name=constants.F1_SCORE,
        iou_matching=None,
        verbose=False
):
    """Average F1 over several IoU values.

    The average F1 performance is
    computed as the area under the F1 vs IoU curve.
    """
    # Go through several IoU values
    first_iou = 0
    last_iou = 1
    res_iou = 0.01
    n_points = int(np.round((last_iou - first_iou) / res_iou))
    full_iou_list = np.arange(n_points + 1) * res_iou + first_iou
    if verbose:
        print('Using %d IoU thresholds from %1.1f to %1.1f'
              % (n_points + 1, first_iou, last_iou))
        print('Computing %s values' % metric_name, flush=True)

    metric_list = metric_vs_iou(
        events, detections, full_iou_list,
        metric_name=metric_name, iou_matching=iou_matching, verbose=verbose)

    # To compute the area under the curve, we'll use trapezoidal aproximation
    # So we need to divide by two the extremes
    metric_list[0] = metric_list[0] / 2
    metric_list[-1] = metric_list[-1] / 2
    # And now we compute the AUC
    avg_metric = np.sum(metric_list * res_iou)
    if verbose:
        print('Done')
    return avg_metric


def average_metric_with_list(
        events_list,
        detections_list,
        metric_name=constants.F1_SCORE,
        iou_matching_list=None,
        verbose=False
):
    if iou_matching_list is None:
        iou_matching_list = [None] * len(events_list)
    all_avg_list = [
        average_metric(
            events, detections,
            metric_name=metric_name,
            verbose=verbose,
            iou_matching=iou_matching
        )
        for (events, detections, iou_matching)
        in zip(events_list, detections_list, iou_matching_list)
    ]
    all_avg = np.mean(all_avg_list)
    return all_avg


def get_iou(single_event_1, single_event_2):
    intersection = min(
        single_event_1[1], single_event_2[1]
    ) - max(
        single_event_1[0], single_event_2[0]
    ) + 1
    union = max(
        single_event_1[1], single_event_2[1]
    ) - min(
        single_event_1[0], single_event_2[0]
    ) + 1
    if union > 0:
        this_iou = intersection / union
    else:
        this_iou = 0
    return this_iou
