from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from joblib import delayed, Parallel

import numpy as np

from sleeprnn.common import checks, constants, pkeys
from sleeprnn.data.stamp_correction import filter_duration_stamps
from sleeprnn.data.stamp_correction import combine_close_stamps
from sleeprnn.data.utils import seq2stamp_with_pages, extract_pages_for_stamps
from sleeprnn.data.utils import seq2stamp
from sleeprnn.data.utils import get_overlap_matrix


class PostProcessor(object):

    def __init__(self, event_name, params=None):
        checks.check_valid_value(
            event_name, 'event_name',
            [constants.SPINDLE, constants.KCOMPLEX])

        self.event_name = event_name
        self.params = pkeys.default_params.copy()
        if params is not None:
            self.params.update(params)

    def proba2stamps(
            self,
            proba_data,
            pages_indices=None,
            pages_indices_subset=None,
            thr=0.5):
        """
        If thr is None, pages_sequence is assumed to be already binarized.
        fs_input corresponds to sampling frequency of pages_sequence,
        fs_outputs corresponds to desired sampling frequency.
        """

        # Thresholding
        if thr is None:
            # We assume that sequence is already binary
            proba_data_bin_high = proba_data
            proba_data_bin_low = proba_data
        else:
            low_thr_factor = 0.85
            low_thr = thr * low_thr_factor
            # print("debug: low thr:", low_thr)
            proba_data_bin_high = (proba_data >= thr).astype(np.int32)
            proba_data_bin_low = (proba_data >= low_thr).astype(np.int32)

        # Transformation to stamps based on low thr (for duration)
        if pages_indices is None:
            stamps_low = seq2stamp(proba_data_bin_low)
            stamps_high = seq2stamp(proba_data_bin_high)
        else:
            stamps_low = seq2stamp_with_pages(proba_data_bin_low, pages_indices)
            stamps_high = seq2stamp_with_pages(proba_data_bin_high, pages_indices)

        # Only keep candidates that surpassed high threshold (for detection)
        # i.e., only stamps_low intersecting with stamps_high
        overlap_check = get_overlap_matrix(stamps_low, stamps_high)  # shape (n_low, n_high)
        if overlap_check.sum() == 0:
            stamps = np.zeros((0, 2), dtype=np.int32)
        else:
            overlap_check = overlap_check.sum(axis=1)  # shape (n_low,)
            valid_lows = np.where(overlap_check > 0)[0]
            stamps = stamps_low[valid_lows]
        # print("debug: stamps low", stamps_low.shape, "stamps high", stamps_high.shape, "stamps", stamps.shape)

        # Postprocessing
        # Note that when min_separation, min_duration, or max_duration is None,
        # that postprocessing doesn't happen.
        downsampling_factor = self.params[pkeys.TOTAL_DOWNSAMPLING_FACTOR]
        fs_input = self.params[pkeys.FS] // downsampling_factor
        fs_output = self.params[pkeys.FS]

        if self.event_name == constants.SPINDLE:
            min_separation = self.params[pkeys.SS_MIN_SEPARATION]
            min_duration = self.params[pkeys.SS_MIN_DURATION]
            max_duration = self.params[pkeys.SS_MAX_DURATION]
        else:
            min_separation = self.params[pkeys.KC_MIN_SEPARATION]
            min_duration = self.params[pkeys.KC_MIN_DURATION]
            max_duration = self.params[pkeys.KC_MAX_DURATION]

        if pkeys.REPAIR_LONG_DETECTIONS not in self.params:
            repair_long = True  # default
        else:
            repair_long = self.params[pkeys.REPAIR_LONG_DETECTIONS]

        stamps = combine_close_stamps(stamps, fs_input, min_separation)
        stamps = filter_duration_stamps(stamps, fs_input, min_duration, max_duration, repair_long=repair_long)

        # Upsampling
        if fs_output > fs_input:
            stamps = self._upsample_stamps(stamps)
        elif fs_output < fs_input:
            raise ValueError('fs_output has to be greater than fs_input')

        if pages_indices_subset is not None:
            page_size = int(self.params[pkeys.PAGE_DURATION] * fs_output)
            stamps = extract_pages_for_stamps(
                stamps, pages_indices_subset, page_size)

        return stamps

    def proba2stamps_with_list(
            self,
            pages_sequence_list,
            pages_indices_list=None,
            pages_indices_subset_list=None,
            thr=0.5):

        if pages_indices_list is None:
            pages_indices_list = [None] * len(pages_sequence_list)
        if pages_indices_subset_list is None:
            pages_indices_subset_list = [None] * len(pages_sequence_list)

        stamps_list = Parallel(n_jobs=-1)(
            delayed(self.proba2stamps)(
                pages_sequence,
                pages_indices,
                pages_indices_subset=pages_indices_subset,
                thr=thr)
            for (
                pages_sequence,
                pages_indices,
                pages_indices_subset)
            in zip(
                pages_sequence_list,
                pages_indices_list,
                pages_indices_subset_list)
        )

        return stamps_list

    def _upsample_stamps(self, stamps):
        """Upsamples timestamps of stamps to match a greater sampling frequency.
        """
        upsample_factor = self.params[pkeys.TOTAL_DOWNSAMPLING_FACTOR]
        if pkeys.ALIGNED_DOWNSAMPLING not in self.params:
            aligned_down = False
        else:
            aligned_down = self.params[pkeys.ALIGNED_DOWNSAMPLING]
        if aligned_down:
            stamps = stamps * upsample_factor
            stamps[:, 1] = stamps[:, 1] + upsample_factor - 1
            stamps = stamps.astype(np.int32)
        else:
            stamps = stamps * upsample_factor
            stamps[:, 0] = stamps[:, 0] - upsample_factor // 2
            stamps[:, 1] = stamps[:, 1] + upsample_factor // 2
            stamps = stamps.astype(np.int32)
        return stamps
