import os
import pickle

import numpy as np
import pyedflib

from src.common import checks, constants
from src.data.mass_kc import MassKC
from src.data.mass_ss import MassSS
from src.data.moda_ss import ModaSS
from src.data.cap_ss import CapSS
from src.data.pink import Pink
from src.data.nsrr_ss import NsrrSS

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '../..')
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results')
BASELINES_PATH = os.path.join(PROJECT_ROOT, 'resources', 'comparison_data', 'baselines')
EXPERT_PATH = os.path.join(PROJECT_ROOT, 'resources', 'comparison_data', 'expert')


def replace_submodule_in_module(module, old_submodule, new_submodule):
    module_splitted = module.split(".")
    if old_submodule in module_splitted:
        idx_name = module_splitted.index(old_submodule)
        module_splitted[idx_name] = new_submodule
    new_module = ".".join(module_splitted)
    return new_module


class RefactorUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        module = replace_submodule_in_module(module, 'sleep', 'sleeprnn')
        module = replace_submodule_in_module(module, 'sleeprnn', 'src')
        module = replace_submodule_in_module(module, 'neuralnet', 'nn')
        return super().find_class(module, name)


def read_predictions_crossval(
        ckpt_folder,
        parent_dataset,
        task_mode,
        verbose=False
):
    print("Loading predictions") if verbose else None
    predictions_path = os.path.abspath(os.path.join(
        RESULTS_PATH,
        'predictions_%s' % parent_dataset.dataset_name,
        ckpt_folder))
    if verbose:
        print('Prediction path:', predictions_path)
    fold_ids, fold_prefix = parse_folds(predictions_path, verbose=verbose)
    predictions_dict = {}
    for k in fold_ids:
        ckpt_path = os.path.abspath(os.path.join(
            RESULTS_PATH,
            'predictions_%s' % parent_dataset.dataset_name,
            ckpt_folder,
            '%s%d' % (fold_prefix, k)))
        pred_path_dict = parse_prediction_files(ckpt_path)
        this_dict = {}
        for set_name in pred_path_dict.keys():
            with open(pred_path_dict[set_name], 'rb') as handle:
                this_pred = RefactorUnpickler(handle).load()
            this_pred.set_parent_dataset(parent_dataset)
            this_dict[set_name] = this_pred
        predictions_dict[k] = this_dict
    return predictions_dict


def parse_prediction_files(ckpt_path):
    available_preds = os.listdir(ckpt_path)
    available_preds.sort()
    set_names = [pred_fname.split(".")[0].split("_")[-1] for pred_fname in available_preds]
    path_dict = {
        set_name: os.path.join(ckpt_path, pred_fname)
        for (set_name, pred_fname) in zip(set_names, available_preds)
    }
    return path_dict


def parse_folds(predictions_path, verbose=False):
    folds = os.listdir(predictions_path)
    folds = [s for s in folds if '.' not in s]
    if verbose:
        print('Folds found:', folds)
    if not folds[0][-1].isdigit():
        raise ValueError("Inside %s there are no numbered folders" % predictions_path)
    # Get available fold ids
    fold_ids = [get_fold_id(fold) for fold in folds]
    fold_ids.sort()
    fold_prefix = get_fold_prefix(folds[0])
    return fold_ids, fold_prefix


def generate_imputed_ids(fold_ids):
    max_id = np.max(fold_ids)
    n_possible_ids = max_id + 1
    fold_ids_imputed = n_possible_ids * [None]
    for fold_id in fold_ids:
        fold_ids_imputed[fold_id] = fold_id
    return fold_ids_imputed


def get_fold_id(fold_name):
    x = [s for s in fold_name if s.isdigit()]
    x = "".join(x)
    x = int(x)
    return x


def get_fold_prefix(fold_name):
    x = [s for s in fold_name if s.isalpha()]
    x = "".join(x)
    return x


def read_prediction_with_seeds(
        ckpt_folder,
        dataset_name,
        task_mode,
        seed_id_list,
        set_list=None,
        verbose=True,
        parent_dataset=None,
):
    if verbose:
        print('Loading predictions')
    if set_list is None:
        set_list = [
            constants.TRAIN_SUBSET,
            constants.VAL_SUBSET,
            constants.TEST_SUBSET]
    predictions_dict = {}
    for k in seed_id_list:
        # Restore predictions
        ckpt_path = os.path.abspath(os.path.join(
            RESULTS_PATH,
            'predictions_%s' % dataset_name,
            ckpt_folder,
            'seed%d' % k
        ))
        this_dict = {}
        for set_name in set_list:
            filename = os.path.join(
                ckpt_path,
                'prediction_%s_%s.pkl' % (task_mode, set_name))
            with open(filename, 'rb') as handle:
                this_pred = RefactorUnpickler(handle).load()
            this_pred.set_parent_dataset(parent_dataset)
            this_dict[set_name] = this_pred
        predictions_dict[k] = this_dict
        if verbose:
            print('Loaded %s' % ckpt_path)
    return predictions_dict


def read_signals_from_edf(filepath):
    signal_dict = {}
    with pyedflib.EdfReader(filepath) as file:
        signal_names = file.getSignalLabels()
        for k, name in enumerate(signal_names):
            this_signal = file.readSignal(k)
            signal_dict[name] = this_signal
    return signal_dict


def load_dataset(dataset_name, load_checkpoint=True, params=None, verbose=True, **kwargs):
    # Load data
    name_to_class_map = {
        constants.MASS_KC_NAME: MassKC,
        constants.MASS_SS_NAME: MassSS,
        constants.MODA_SS_NAME: ModaSS,
        constants.CAP_SS_NAME: CapSS,
        constants.PINK_NAME: Pink,
        constants.NSRR_SS_NAME: NsrrSS,
    }
    dataset_class = name_to_class_map[dataset_name]
    dataset = dataset_class(load_checkpoint=load_checkpoint, params=params, verbose=verbose, **kwargs)
    return dataset


def load_baselines(
        baselines_to_load,
        subject_ids,
        dataset_name,
        which_expert,
        n_folds=10,
):
    baselines_data_dict = {}
    for baseline_name in baselines_to_load:
        # Check if we have results for this baseline
        folder_to_check = os.path.join(
            BASELINES_PATH, baseline_name, dataset_name, 'e%d' % which_expert)
        if os.path.exists(folder_to_check):
            print('%s found. ' % baseline_name, end='', flush=True)
            iou_axis_added = False
            iou_bins_added = False
            tmp_f1_baseline = []
            tmp_recall_baseline = []
            tmp_precision_baseline = []
            tmp_iou_hist_baseline = []
            tmp_iou_mean_baseline = []
            tmp_af1_mean_baseline = []
            right_prefix = '%s_%s_e%d' % (baseline_name, dataset_name, which_expert)
            for k in np.arange(n_folds):
                print(' %d ' % k, end='', flush=True)
                f1_seed = []
                rec_seed = []
                pre_seed = []
                iou_hist_seed = []
                iou_mean_seed = []
                af1_mean_seed = []
                for i, subject_id in enumerate(subject_ids):
                    fname = '%s_fold%d_s%02d.npz' % (
                    right_prefix, k, subject_id)
                    fname_path = os.path.join(
                        folder_to_check, 'fold%d' % k, fname)
                    this_data = np.load(fname_path)
                    if not iou_axis_added:
                        tmp_iou_axis = this_data['iou_axis']
                        iou_axis_added = True
                    if not iou_bins_added:
                        tmp_iou_bins = this_data['iou_hist_bins']
                        iou_bins_added = True
                    f1_seed.append(this_data['f1_vs_iou'])
                    rec_seed.append(this_data['recall_vs_iou'])
                    pre_seed.append(this_data['precision_vs_iou'])
                    iou_hist_seed.append(this_data['iou_hist_values'])
                    iou_mean_seed.append(this_data['subject_iou'])
                    af1_mean_seed.append(this_data['subject_af1'])
                f1_seed = np.stack(f1_seed, axis=0).mean(axis=0)
                rec_seed = np.stack(rec_seed, axis=0).mean(axis=0)
                pre_seed = np.stack(pre_seed, axis=0).mean(axis=0)
                iou_hist_seed = np.stack(iou_hist_seed, axis=0).mean(axis=0)
                iou_mean_seed = np.stack(iou_mean_seed, axis=0).mean(axis=0)
                af1_mean_seed = np.stack(af1_mean_seed, axis=0).mean(axis=0)
                tmp_f1_baseline.append(f1_seed)
                tmp_recall_baseline.append(rec_seed)
                tmp_precision_baseline.append(pre_seed)
                tmp_iou_hist_baseline.append(iou_hist_seed)
                tmp_iou_mean_baseline.append(iou_mean_seed)
                tmp_af1_mean_baseline.append(af1_mean_seed)
            tmp_f1_baseline = np.stack(tmp_f1_baseline, axis=0)
            tmp_recall_baseline = np.stack(tmp_recall_baseline, axis=0)
            tmp_precision_baseline = np.stack(tmp_precision_baseline, axis=0)
            tmp_iou_hist_baseline = np.stack(tmp_iou_hist_baseline, axis=0)
            tmp_iou_mean_baseline = np.stack(tmp_iou_mean_baseline, axis=0)
            tmp_af1_mean_baseline = np.stack(tmp_af1_mean_baseline, axis=0)
            baselines_data_dict[baseline_name] = {
                constants.F1_VS_IOU: tmp_f1_baseline,
                constants.RECALL_VS_IOU: tmp_recall_baseline,
                constants.PRECISION_VS_IOU: tmp_precision_baseline,
                constants.IOU_HIST_BINS: tmp_iou_bins,
                constants.IOU_CURVE_AXIS: tmp_iou_axis,
                constants.IOU_HIST_VALUES: tmp_iou_hist_baseline,
                constants.MEAN_IOU: tmp_iou_mean_baseline,
                constants.MEAN_AF1: tmp_af1_mean_baseline
            }
            print('Loaded.')
        else:
            print('%s not found.' % baseline_name)
            baselines_data_dict[baseline_name] = None
    return baselines_data_dict


def load_ss_expert_performance():
    expert_f1_curve_mean = np.loadtxt(
        os.path.join(EXPERT_PATH, 'ss_f1_vs_iou_expert_mean.csv'), delimiter=',')
    expert_f1_curve_std = np.loadtxt(
        os.path.join(EXPERT_PATH, 'ss_f1_vs_iou_expert_std.csv'), delimiter=',')
    expert_f1_curve_mean = expert_f1_curve_mean[1:, :]
    expert_f1_curve_std = expert_f1_curve_std[1:, :]
    expert_rec_prec = np.loadtxt(
        os.path.join(EXPERT_PATH, 'ss_pr_expert_mean.csv'), delimiter=',')
    expert_recall = expert_rec_prec[0]
    expert_precision = expert_rec_prec[1]
    expert_data_dict = {
        constants.IOU_CURVE_AXIS: expert_f1_curve_mean[:, 0],
        '%s_mean' % constants.F1_VS_IOU: expert_f1_curve_mean[:, 1],
        '%s_std' % constants.F1_VS_IOU: expert_f1_curve_std[:, 1],
        constants.RECALL: expert_recall,
        constants.PRECISION: expert_precision
    }
    return expert_data_dict
