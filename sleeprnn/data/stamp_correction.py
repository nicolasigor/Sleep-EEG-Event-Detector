"""stamp_correction.py: Module for general postprocessing operations of
annotations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def combine_close_stamps(marks, fs, min_separation):
    """Combines contiguous marks that are too close to each other. Marks are
    assumed to be sample-stamps.

    If min_separation is None, the functionality is bypassed.
    """
    if marks.size == 0:
        return marks

    if min_separation is None:
        combined_marks = marks
    else:
        marks = np.sort(marks, axis=0)
        combined_marks = [marks[0, :]]
        for i in range(1, marks.shape[0]):
            last_mark = combined_marks[-1]
            this_mark = marks[i, :]
            gap = (this_mark[0] - last_mark[1]) / fs
            if gap < min_separation:
                # Combine mark, so the last mark ends in the maximum ending.
                combined_marks[-1][1] = max(this_mark[1], combined_marks[-1][1])
            else:
                combined_marks.append(this_mark)
        combined_marks = np.stack(combined_marks, axis=0)
    return combined_marks


def filter_duration_stamps(marks, fs, min_duration, max_duration, repair_long=True):
    """Removes marks that are too short or strangely long. Marks longer than
    max_duration but not strangely long are cropped to keep the central
    max_duration duration. Durations are assumed to be in seconds.
    Marks are assumed to be sample-stamps.

    If min_duration is None, no short marks are removed.
    If max_duration is None, no long marks are removed.
    """
    if marks.size == 0:
        return marks

    if min_duration is None and max_duration is None:
        return marks
    else:
        durations = (marks[:, 1] - marks[:, 0] + 1) / fs

        if min_duration is not None:
            # Remove too short spindles
            feasible_idx = np.where(durations >= min_duration)[0]
            marks = marks[feasible_idx, :]
            durations = durations[feasible_idx]

        if max_duration is not None:

            if repair_long:
                # Remove weird annotations (extremely long)
                feasible_idx = np.where(durations <= 2 * max_duration)[0]
                marks = marks[feasible_idx, :]
                durations = durations[feasible_idx]

                # For annotations with durations longer than max_duration,
                # keep the central seconds
                excess = durations - max_duration
                excess = np.clip(excess, 0, None)
                half_remove = ((fs * excess + 1) / 2).astype(np.int32)
                half_remove_array = np.stack([half_remove, -half_remove], axis=1)
                marks = marks + half_remove_array
                # marks[:, 0] = marks[:, 0] + half_remove
                # marks[:, 1] = marks[:, 1] - half_remove
            else:
                # No repairing, simply remove
                feasible_idx = np.where(durations <= max_duration)[0]
                marks = marks[feasible_idx, :]
    return marks
