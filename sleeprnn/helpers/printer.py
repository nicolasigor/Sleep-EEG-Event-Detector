import numpy as np

from sleeprnn.helpers import misc
from sleeprnn.common import constants


def print_available_ckpt(optimal_thr_for_ckpt_dict, filter_dates, file=None):
    if filter_dates[0] is None:
        filter_dates[0] = -1
    if filter_dates[1] is None:
        filter_dates[1] = 1e12
    print('Available ckpt:')
    for key in optimal_thr_for_ckpt_dict.keys():
        key_date = int(key.split("_")[0])
        if filter_dates[0] <= key_date <= filter_dates[1]:
            if file is not None:
                print('    %s' % key, file=file)
            print('    %s' % key)


def print_performance_at_iou(
        performance_data_dict, iou_thr, label, file=None,
        decimal_precision=1, show_iqr_iou=True
):
    str_to_print = '%%%d.%df \u00B1 %%%d.%df' % (
        1, decimal_precision, 1, decimal_precision)

    str_to_print_iqr = '%%%d.%df \u00B1 %%%d.%df [%%%d.%df - %%%d.%df]' % (
        1, decimal_precision, 1, decimal_precision,
        1, decimal_precision, 1, decimal_precision)

    iou_curve_axis = performance_data_dict[constants.IOU_CURVE_AXIS]
    idx_to_show = misc.closest_index(iou_thr, iou_curve_axis)

    mean_f1_vs_iou = performance_data_dict[constants.F1_VS_IOU].mean(axis=0)
    std_f1_vs_iou = performance_data_dict[constants.F1_VS_IOU].std(axis=0)
    msg = 'F1 ' + str_to_print % (
        100*mean_f1_vs_iou[idx_to_show], 100*std_f1_vs_iou[idx_to_show])

    mean_rec_vs_iou = performance_data_dict[constants.RECALL_VS_IOU].mean(axis=0)
    std_rec_vs_iou = performance_data_dict[constants.RECALL_VS_IOU].std(axis=0)
    msg = msg + ', Recall ' + str_to_print % (
        100*mean_rec_vs_iou[idx_to_show], 100*std_rec_vs_iou[idx_to_show])

    mean_pre_vs_iou = performance_data_dict[constants.PRECISION_VS_IOU].mean(
        axis=0)
    std_pre_vs_iou = performance_data_dict[constants.PRECISION_VS_IOU].std(axis=0)
    msg = msg + ', Precision ' + str_to_print % (
        100*mean_pre_vs_iou[idx_to_show], 100*std_pre_vs_iou[idx_to_show])

    msg = msg + ', AF1 ' + str_to_print % (
        100*performance_data_dict[constants.MEAN_AF1].mean(),
        100*performance_data_dict[constants.MEAN_AF1].std())

    if show_iqr_iou:
        msg = msg + ', IoU ' + str_to_print_iqr % (
            100*performance_data_dict[constants.MEAN_IOU].mean(),
            100*performance_data_dict[constants.MEAN_IOU].std(),
            100*performance_data_dict[constants.IQR_LOW_IOU].mean(),
            100*performance_data_dict[constants.IQR_HIGH_IOU].mean()
        )
    else:
        msg = msg + ', IoU ' + str_to_print % (
            100 * performance_data_dict[constants.MEAN_IOU].mean(),
            100 * performance_data_dict[constants.MEAN_IOU].std()
        )

    msg = msg + ' for %s' % label
    if file is not None:
        print(msg, file=file)
    print(msg)


def print_formatted_performance_at_iou(
        performance_data_dict, iou_thr, label, print_header=True,
        decimal_precision=1, file=None, show_iqr_iou=True
):
    iou_curve_axis = performance_data_dict[constants.IOU_CURVE_AXIS]
    idx_to_show = misc.closest_index(iou_thr, iou_curve_axis)

    str_to_print = '%%%d.%df \u00B1 %%%d.%df' % (
        1, decimal_precision, 1, decimal_precision)

    str_to_print_iqr = '%%%d.%df \u00B1 %%%d.%df [%%%d.%df - %%%d.%df]' % (
        1, decimal_precision, 1, decimal_precision,
        1, decimal_precision, 1, decimal_precision)

    if print_header:
        if file is not None:
            print('Model; F1; Recall; Precision; AF1; IoU', file=file)
        print('Model; F1; Recall; Precision; AF1; IoU')
    
    msg = label
    
    mean_f1_vs_iou = performance_data_dict[constants.F1_VS_IOU].mean(axis=0)
    std_f1_vs_iou = performance_data_dict[constants.F1_VS_IOU].std(axis=0)
    msg = msg + '; ' + str_to_print % (
        100*mean_f1_vs_iou[idx_to_show], 100*std_f1_vs_iou[idx_to_show])

    mean_rec_vs_iou = performance_data_dict[constants.RECALL_VS_IOU].mean(axis=0)
    std_rec_vs_iou = performance_data_dict[constants.RECALL_VS_IOU].std(axis=0)
    msg = msg + '; ' + str_to_print % (
        100*mean_rec_vs_iou[idx_to_show], 100*std_rec_vs_iou[idx_to_show])

    mean_pre_vs_iou = performance_data_dict[constants.PRECISION_VS_IOU].mean(
        axis=0)
    std_pre_vs_iou = performance_data_dict[constants.PRECISION_VS_IOU].std(axis=0)
    msg = msg + '; ' + str_to_print % (
        100*mean_pre_vs_iou[idx_to_show], 100*std_pre_vs_iou[idx_to_show])

    msg = msg + '; ' + str_to_print % (
        100*performance_data_dict[constants.MEAN_AF1].mean(),
        100*performance_data_dict[constants.MEAN_AF1].std())

    if show_iqr_iou:
        msg = msg + '; ' + str_to_print_iqr % (
            100 * performance_data_dict[constants.MEAN_IOU].mean(),
            100 * performance_data_dict[constants.MEAN_IOU].std(),
            100 * performance_data_dict[constants.IQR_LOW_IOU].mean(),
            100 * performance_data_dict[constants.IQR_HIGH_IOU].mean()
        )
    else:
        msg = msg + '; ' + str_to_print % (
            100 * performance_data_dict[constants.MEAN_IOU].mean(),
            100 * performance_data_dict[constants.MEAN_IOU].std()
        )
    if file is not None:
        print(msg, file=file)
    print(msg)

