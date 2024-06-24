import math

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import resample_poly, butter, filtfilt, find_peaks


def split_mass(subject_ids, split_id, train_fraction=0.75, verbose=False):
    """Subject ids is the sorted list of non-testing ids."""
    n_subjects = len(subject_ids)
    n_train = int(n_subjects * train_fraction)
    if verbose:
        print('Split IDs: Total %d -- Training %d' % (n_subjects, n_train))
    n_val = n_subjects - n_train
    start_idx = split_id * n_val
    epoch = int(start_idx / n_subjects)
    attempts = 1
    while True:
        random_idx_1 = np.random.RandomState(seed=epoch).permutation(n_subjects)
        random_idx_2 = np.random.RandomState(seed=epoch+attempts).permutation(n_subjects)
        random_idx = np.concatenate([random_idx_1, random_idx_2])
        start_idx_relative = start_idx % n_subjects
        val_idx = random_idx[start_idx_relative:(start_idx_relative + n_val)]
        if np.unique(val_idx).size == n_val:
            break
        else:
            print("Attempting new split due to replication in val set")
            attempts = attempts + 1000

    val_ids = [subject_ids[i] for i in val_idx]
    train_ids = [sub_id for sub_id in subject_ids if sub_id not in val_ids]
    val_ids.sort()
    train_ids.sort()
    return train_ids, val_ids


def preprocess_mass_signals(signal, fs_original, fs_target=200):
    """Bandpass filtering and resampling to desired sampling frequency."""
    # ######
    # Particular fix for mass dataset:
    fs_old_round = int(np.round(fs_original))
    # Transform the original fs frequency with decimals to rounded version
    signal = resample_signal_linear(signal, fs_old=fs_original, fs_new=fs_old_round)
    # ######

    # Broad bandpass filter to signal
    signal = broad_filter(signal, fs_old_round)

    # Now resample to the required frequency
    if fs_target != fs_old_round:
        print('Resampling from %d Hz to required %d Hz' % (fs_old_round, fs_target))
        signal = resample_signal(signal, fs_old=fs_old_round, fs_new=fs_target)
    else:
        print('Signal already at required %d Hz' % fs_target)

    signal = signal.astype(np.float32)
    return signal


def postprocess_mass_detections(
        signal, fs, detections, is_kcomplex,
        min_separation=0.3, min_duration=0.3, max_duration=3.0, repair_long=True
):
    """detections is (n_detections, 2) array"""
    detections = combine_close_stamps(detections, fs, min_separation)
    detections = filter_duration_stamps(detections, fs, min_duration, max_duration, repair_long=repair_long)
    if is_kcomplex:
        # For K-Complexes we perform the splitting procedure
        detections = kcomplex_stamp_split(signal, detections, fs, signal_is_filtered=False)
    return detections

# ###########################
# From here, the functions used by the above ones are defined.

def resample_signal(signal, fs_old, fs_new):
    """Returns resampled signal, from fs_old Hz to fs_new Hz."""
    gcd_freqs = math.gcd(fs_new, fs_old)
    up = int(fs_new / gcd_freqs)
    down = int(fs_old / gcd_freqs)
    signal = resample_poly(signal, up, down)
    signal = np.array(signal, dtype=np.float32)
    return signal


def resample_signal_linear(signal, fs_old, fs_new):
    """Returns resampled signal, from fs_old Hz to fs_new Hz.

    This implementation uses simple linear interpolation to achieve this.
    """
    t = np.cumsum(np.ones(len(signal)) / fs_old)
    t_new = np.arange(t[0], t[-1], 1 / fs_new)
    signal = interp1d(t, signal)(t_new)
    return signal


def broad_filter(signal, fs, lowcut=0.1, highcut=35):
    """Returns filtered signal sampled at fs Hz, with a [lowcut, highcut] Hz
    bandpass."""
    # Generate butter bandpass of order 3.
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(3, (low, high), btype='band')
    # Apply filter to the signal with zero-phase.
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


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
    max_duration duration if repair_long is True.
    Durations are assumed to be in seconds.
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
                feasible_idx = np.where(durations <= 2*max_duration)[0]
                marks = marks[feasible_idx, :]
                durations = durations[feasible_idx]

                # For annotations with durations longer than max_duration,
                # keep the central seconds
                excess = durations - max_duration
                excess = np.clip(excess, 0, None)
                half_remove = ((fs * excess + 1) / 2).astype(np.int32)
                half_remove_array = np.stack([half_remove, -half_remove], axis=1)
                marks = marks + half_remove_array
            else:
                # No repairing, simply remove
                feasible_idx = np.where(durations <= max_duration)[0]
                marks = marks[feasible_idx, :]
    return marks


def kcomplex_stamp_split(
        signal,
        stamps,
        fs,
        highcut=4,
        left_edge_tol=0.05,
        right_edge_tol=0.2,
        signal_is_filtered=False
):
    left_edge_tol = fs * left_edge_tol
    right_edge_tol = fs * right_edge_tol

    if signal_is_filtered:
        filt_signal = signal
    else:
        filt_signal = filter_iir_lowpass(signal, fs, highcut=highcut)

    new_stamps = []
    for stamp in stamps:
        stamp_size = stamp[1] - stamp[0] + 1
        filt_in_stamp = filt_signal[stamp[0]:stamp[1]]
        negative_peaks, _ = find_peaks(- filt_in_stamp)
        # peaks needs to be negative
        negative_peaks = [
            peak for peak in negative_peaks
            if filt_in_stamp[peak] < 0]

        negative_peaks = [
            peak for peak in negative_peaks
            if left_edge_tol < peak < stamp_size - right_edge_tol]

        n_peaks = len(negative_peaks)
        if n_peaks > 1:
            # Change of sign filtering
            group_peaks = [[negative_peaks[0]]]
            idx_group = 0
            for i in range(1, len(negative_peaks)):
                last_peak = group_peaks[idx_group][-1]
                this_peak = negative_peaks[i]
                signal_between_peaks = filt_in_stamp[last_peak:this_peak]
                min_value = signal_between_peaks.min()
                max_value = signal_between_peaks.max()
                if min_value < 0 < max_value:
                    # there is a change of sign, so it is a new group
                    group_peaks.append([this_peak])
                    idx_group = idx_group + 1
                else:
                    # Now change of sign, same group
                    group_peaks[idx_group].append(this_peak)
            new_peaks = []
            for single_group in group_peaks:
                new_peaks.append(int(np.mean(single_group)))
            negative_peaks = new_peaks

        n_peaks = len(negative_peaks)
        if n_peaks > 1:
            # Split marks
            edges_list = [stamp[0]]
            for i in range(n_peaks-1):
                split_point_rel = (negative_peaks[i] + negative_peaks[i+1]) / 2
                split_point_abs = int(stamp[0] + split_point_rel)
                edges_list.append(split_point_abs)
            edges_list.append(stamp[1])
            for i in range(len(edges_list)-1):
                new_stamps.append([edges_list[i], edges_list[i+1]])
        else:
            new_stamps.append(stamp)
    if len(new_stamps) > 0:
        new_stamps = np.stack(new_stamps, axis=0).astype(np.int32)
    else:
        new_stamps = np.zeros((0, 2)).astype(np.int32)
    return new_stamps


def filter_iir_lowpass(signal, fs, highcut=4):
    """Returns filtered signal sampled at fs Hz, with a highcut Hz
    lowpass."""
    # Generate butter bandpass of order 3.
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(3, high, btype='low')
    # Apply filter to the signal with zero-phase.
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal
