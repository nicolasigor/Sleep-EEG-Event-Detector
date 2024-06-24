"""utils.py: Module for general data eeg data operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import resample_poly, butter, filtfilt, firwin, lfilter, freqz, sosfiltfilt
from scipy.stats import iqr

PATH_THIS_DIR = os.path.dirname(__file__)
PATH_DATA = os.path.join(PATH_THIS_DIR, '..', '..', 'resources', 'datasets')

from sleeprnn.common import constants, checks


def seq2stamp(sequence):
    """Returns the start and end samples of stamps in a binary sequence."""
    if not np.array_equal(sequence, sequence.astype(bool)):
        raise ValueError('Sequence must have binary values only')
    n = len(sequence)
    tmp_result = np.diff(sequence, prepend=0)
    start_times = np.where(tmp_result == 1)[0]
    end_times = np.where(tmp_result == -1)[0] - 1
    # Final edge case
    if start_times.size > end_times.size:
        end_times = np.concatenate([end_times, [n - 1]])
    stamps = np.stack([start_times, end_times], axis=1)
    return stamps


def stamp2seq(stamps, start, end, allow_early_end=False):
    """Returns the binary sequence segment from 'start' to 'end',
    associated with the stamps."""
    if np.any(stamps < start):
        msg = 'Values in intervals should be within start bound'
        raise ValueError(msg)
    if np.any(stamps > end) and not allow_early_end:
        msg = 'Values in intervals should be within end bound'
        raise ValueError(msg)

    sequence = np.zeros(end - start + 1, dtype=np.int32)
    for i in range(len(stamps)):
        start_sample = stamps[i, 0] - start
        end_sample = stamps[i, 1] - start + 1
        sequence[start_sample:end_sample] = 1
    return sequence


def stamp2seq_with_separation(
        stamps, start, end, min_separation_samples, allow_early_end=False):
    """Returns the binary sequence segment from 'start' to 'end',
    associated with the stamps."""
    if np.any(stamps < start):
        msg = 'Values in intervals should be within start bound'
        raise ValueError(msg)
    if np.any(stamps > end) and not allow_early_end:
        msg = 'Values in intervals should be within end bound'
        raise ValueError(msg)

    # Force separation
    stamps = np.sort(stamps, axis=0)
    mod_stamps = [stamps[0, :]]
    for i in range(1, stamps.shape[0]):
        last_stamp = mod_stamps[-1]
        this_stamp = stamps[i, :]
        samples_gap = this_stamp[0] - last_stamp[1] - 1
        if samples_gap < min_separation_samples:
            last_stamp_size = last_stamp[1] - last_stamp[0] + 1
            this_stamp_size = this_stamp[1] - this_stamp[0] + 1
            sum_of_sizes = last_stamp_size + this_stamp_size
            needed_samples = min_separation_samples - samples_gap
            # Proportional elimination of samples
            cut_last = int(np.round(last_stamp_size * needed_samples / sum_of_sizes))
            cut_this = needed_samples - cut_last

            last_stamp[1] = last_stamp[1] - cut_last
            this_stamp[0] = this_stamp[0] + cut_this
            mod_stamps[-1] = last_stamp
            mod_stamps.append(this_stamp)
        else:
            mod_stamps.append(this_stamp)
    mod_stamps = np.stack(mod_stamps, axis=0)

    # Transform modified stamps
    sequence = stamp2seq(mod_stamps, start, end, allow_early_end=allow_early_end)
    return sequence


def seq2stamp_with_pages(pages_sequence, pages_indices):
    """Returns the start and end samples of stamps in a binary sequence that
    is split in pages."

    Args:
        pages_sequence: (2d array) binary array with shape [n_pages, page_size]
        pages_indices: (1d array) array of indices of the corresponding pages in
            pages_sequence, with shape [n_pages,]
    """
    if pages_sequence.shape[0] != pages_indices.shape[0]:
        raise ValueError('Shape mismatch. Inputs need the same number of rows.')
    tmp_sequence = pages_sequence.flatten()
    if not np.array_equal(tmp_sequence, tmp_sequence.astype(bool)):
        raise ValueError('Sequence must have binary values only')

    page_size = pages_sequence.shape[1]
    max_page = np.max(pages_indices)
    max_size = (max_page + 1) * page_size
    global_sequence = np.zeros(max_size, dtype=np.int32)
    for i, page in enumerate(pages_indices):
        sample_start = page * page_size
        sample_end = (page + 1) * page_size
        global_sequence[sample_start:sample_end] = pages_sequence[i, :]
    stamps = seq2stamp(global_sequence)
    return stamps


def pages2seq(pages_data, pages_indices):
    if pages_data.shape[0] != pages_indices.shape[0]:
        raise ValueError('Shape mismatch. Inputs need the same number of rows.')

    page_size = pages_data.shape[1]
    max_page = np.max(pages_indices)
    max_size = (max_page + 1) * page_size
    global_sequence = np.zeros(max_size, dtype=pages_data.dtype)
    for i, page in enumerate(pages_indices):
        sample_start = page * page_size
        sample_end = (page + 1) * page_size
        global_sequence[sample_start:sample_end] = pages_data[i, :]
    return global_sequence


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


def filter_fir(kernel, signal):
    filtered_signal = lfilter(kernel, 1.0, signal)
    n_shift = (kernel.size - 1) // 2
    aligned = np.zeros(filtered_signal.shape)
    aligned[..., :-n_shift] = filtered_signal[..., n_shift:]
    return aligned


def narrow_filter(signal, fs, lowcut, highcut):
    """Returns filtered signal sampled at fs Hz, with a [lowcut, highcut] Hz
    bandpass."""
    ntaps = 21
    width = 0.5
    cutoff = [lowcut, highcut]

    # Kernel design
    b_base = firwin(ntaps, cutoff, width, pass_zero=False, fs=fs)
    kernel = np.append(np.array([0, 0]), b_base) - np.append(
        b_base, np.array([0, 0]))

    # Normalize kernel
    kernel = kernel / np.linalg.norm(kernel)

    # Apply kernel
    filtered_signal = filter_fir(kernel, signal)
    return filtered_signal


def filter_windowed_sinusoidal(
        signal, fs, central_freq, ntaps,
        sinusoidal_fn=np.cos, window_fn=np.hanning):
    # Kernel design
    time_array = np.arange(ntaps) - ntaps // 2
    time_array = time_array / fs
    b_base = sinusoidal_fn(2 * np.pi * central_freq * time_array)
    cos_base = np.cos(2 * np.pi * central_freq * time_array)
    window = window_fn(b_base.size)
    norm_factor = np.sum(window * (cos_base ** 2))
    kernel = b_base * window / norm_factor

    # Apply kernel
    filtered_signal = filter_fir(kernel, signal)
    return filtered_signal


def get_kernel(ntaps, central_freq, fs=1, window_fn=np.hanning, sinusoidal_fn=np.cos):
    # Kernel design
    time_array = np.arange(ntaps) - ntaps // 2
    b_base = sinusoidal_fn(2 * np.pi * central_freq / fs * time_array)
    window = window_fn(b_base.size)
    kernel = b_base * window
    # Normalize kernel
    kernel = kernel / np.linalg.norm(kernel)
    return kernel


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


def norm_clip_signal(
        signal,
        pages_indices,
        page_size,
        norm_computation=constants.NORM_GLOBAL,
        clip_value=10,
        global_std=None,
):

    checks.check_valid_value(
        norm_computation, 'norm_computation',
        [
            constants.NORM_IQR,
            constants.NORM_STD,
            constants.NORM_GLOBAL
        ])
    # print('Clipping at %s' % clip_value)
    if norm_computation == constants.NORM_IQR:
        norm_signal, clip_mask = norm_clip_signal_iqr(signal, pages_indices, page_size, clip_value)
    elif norm_computation == constants.NORM_STD:
        norm_signal, clip_mask = norm_clip_signal_std(signal, pages_indices, page_size, clip_value)
    else:
        if global_std is None:
            raise ValueError('norm_computation is set to global, but global_std is None')
        norm_signal, clip_mask = norm_clip_signal_global(signal, global_std, clip_value)
    return norm_signal, clip_mask


def norm_clip_signal_global(signal, global_std, clip_value=10):
    # print('Normalizing with Global STD of %s' % global_std)
    norm_signal = signal / global_std
    # Now clip to clip_value (only if clip is not None)
    if clip_value:
        clip_mask = (np.abs(norm_signal) > clip_value).astype(np.int32)
        norm_signal = np.clip(norm_signal, -clip_value, clip_value)
    else:
        clip_mask = None
    return norm_signal, clip_mask


def norm_clip_signal_std(signal, pages_indices, page_size, clip_value=10):
    """Normalizes EEG data according to N2 pages statistics, and then clips.

    EEGs are very close to a Gaussian signal, but are subject to outlier values.
    To compute a more robust estimation of the underlying mean and variance of
    pages, we compute the median and the interquartile range. These
    estimations are used to normalize the signal with a Z-score. After
    normalization, the signal is clipped to the [-clip_value, clip_value] range.

    Args:
        signal: 1-D array containing EEG data.
        pages_indices: 1-D array with indices of pages of the hypnogram.
        page_size: (int) Number of samples contained in a single page of the
            hypnogram.
        clip_value: (Optional, int, Defaults to 6) Value used to clip the signal
            after normalization.
    """
    # Extract statistics only from N2 stages
    print('Normalizing with STD after outlier removal')
    n2_data = extract_pages(signal, pages_indices, page_size)
    n2_signal = np.concatenate(n2_data)
    outlier_thr = np.percentile(np.abs(n2_signal), 98)
    tmp_signal = n2_signal[np.abs(n2_signal) <= outlier_thr]
    signal_std = tmp_signal.std()
    print('Signal STD: %1.4f' % signal_std)

    # Normalize entire signal, we assume zero-centered data
    norm_signal = signal / signal_std

    # Now clip to clip_value (only if clip is not None)
    if clip_value:
        clip_mask = (np.abs(norm_signal) > clip_value).astype(np.int32)
        norm_signal = np.clip(norm_signal, -clip_value, clip_value)
    else:
        clip_mask = None

    return norm_signal, clip_mask


def norm_clip_signal_iqr(signal, pages_indices, page_size, clip_value=10):
    """Normalizes EEG data according to N2 pages statistics, and then clips.

    EEGs are very close to a Gaussian signal, but are subject to outlier values.
    To compute a more robust estimation of the underlying mean and variance of
    pages, we compute the median and the interquartile range. These
    estimations are used to normalize the signal with a Z-score. After
    normalization, the signal is clipped to the [-clip_value, clip_value] range.

    Args:
        signal: 1-D array containing EEG data.
        pages_indices: 1-D array with indices of pages of the hypnogram.
        page_size: (int) Number of samples contained in a single page of the
            hypnogram.
        clip_value: (Optional, int, Defaults to 6) Value used to clip the signal
            after normalization.
    """
    # Extract statistics only from N2 stages
    print('Normalizing with IQR')
    n2_data = extract_pages(signal, pages_indices, page_size)
    n2_signal = np.concatenate(n2_data)
    signal_std = iqr(n2_signal) / 1.349
    signal_median = np.median(n2_signal)

    print('Signal STD: %1.4f' % signal_std)

    # Normalize entire signal
    # norm_signal = (signal - signal_median) / signal_std
    norm_signal = signal / signal_std

    # Now clip to clip_value (only if clip is not None)
    if clip_value:
        clip_mask = (np.abs(norm_signal) > clip_value).astype(np.int32)
        norm_signal = np.clip(norm_signal, -clip_value, clip_value)
    else:
        clip_mask = None

    return norm_signal, clip_mask


def fir_freq_response(fir_filter, fs):
    w, h = freqz(fir_filter)
    resp_freq = w * fs / (2 * np.pi)
    resp_amp = abs(h)
    return resp_freq, resp_amp


def power_spectrum(signal, fs, apply_hanning=False):
    """Returns the single-sided power spectrum of the signal using FFT"""
    if apply_hanning:
        window_han = np.hanning(signal.size)
        signal = signal * window_han
    n = signal.size
    y = np.fft.fft(signal)
    y = np.abs(y) / n
    power = y[:n // 2]
    power[1:-1] = 2 * power[1:-1]
    freq = np.fft.fftfreq(n, d=1 / fs)
    freq = freq[:n // 2]
    return power, freq


def power_spectrum_by_sliding_window(x, fs, window_duration=5):
    """Computes FFT in non-overlapping windows, then averages.
    It assumes 1D input.
    """
    window_size = int(fs * window_duration)
    x = x.reshape(-1, window_size)
    window_shape = np.hanning(window_size).reshape(1, -1)
    x = x * window_shape
    y = np.fft.rfft(x, axis=1) / window_size
    y = np.abs(y).mean(axis=0)
    f = np.fft.rfftfreq(window_size, d=1. / fs)
    return f, y


def extract_pages_from_centers(sequence, centers, page_size, border_size=0):
    """Extracts and returns the pages centered at the given set of centers
    from the sequence, with zero padding if the extracted segment is beyond the limits
    of the sequence.

    Args:
        sequence: (1-D Array) sequence from where to extract data.
        centers: (1-D Array) array of integers indicating the center samples to be extracted.
        page_size: (int) number of samples in each page.
        border_size: (Optional, int, defaults to 0) number of samples to be
            added at each border.

    Returns:
        segments: (2-D Array) array of shape [n_pages, page_size+2*border_size]
            that contains the extracted data, where n_pages = len(centers).
    """
    sequence = np.asarray(sequence)
    centers = np.asarray(centers)

    max_sample = np.max(centers) + (page_size // 2) + border_size + 2
    if max_sample > sequence.size:
        extended_sequence = np.zeros(max_sample).astype(sequence.dtype)
        extended_sequence[:sequence.size] = sequence
    else:
        extended_sequence = sequence

    segments = []
    for center in centers:
        sample_start = center - page_size // 2 - border_size
        sample_end = sample_start + page_size + 2 * border_size
        if sample_start < 0:
            missing_samples = np.abs(sample_start)
            page_signal = extended_sequence[:sample_end]
            missing_part = np.zeros(missing_samples).astype(page_signal.dtype)
            page_signal = np.concatenate([missing_part, page_signal])
        else:
            page_signal = extended_sequence[sample_start:sample_end]
        segments.append(page_signal)
    segments = np.stack(segments, axis=0)
    return segments


def extract_pages(sequence, pages_indices, page_size, border_size=0):
    """Extracts and returns the given set of pages from the sequence
    with zero padding if border is beyond the end of the sequence.

    Args:
        sequence: (1-D Array) sequence from where to extract data.
        pages_indices: (1-D Array) array of indices of pages to be extracted.
        page_size: (int) number of samples in each page.
        border_size: (Optional, int, defaults to 0) number of samples to be
            added at each border.

    Returns:
        pages_data: (2-D Array) array of shape [n_pages,page_size+2*border_size]
            that contains the extracted data.
    """
    sequence = np.asarray(sequence)
    pages_indices = np.asarray(pages_indices)

    max_sample = (pages_indices.max() + 1) * page_size + border_size + 1
    extended_sequence = np.zeros(max_sample).astype(sequence.dtype)
    if extended_sequence.size > sequence.size:
        extended_sequence[:sequence.size] = sequence
    else:
        extended_sequence = sequence

    pages_list = []
    for page in pages_indices:
        sample_start = page * page_size - border_size
        sample_end = (page + 1) * page_size + border_size
        if sample_start < 0:
            missing_samples = np.abs(sample_start)
            page_signal = extended_sequence[:sample_end]
            missing_part = np.zeros(missing_samples).astype(page_signal.dtype)
            page_signal = np.concatenate([missing_part, page_signal])
        else:
            page_signal = extended_sequence[sample_start:sample_end]
        pages_list.append(page_signal)
    pages_data = np.stack(pages_list, axis=0)
    return pages_data


def extract_pages_for_stamps(stamps, pages_indices, page_size):
    """Returns stamps that are at least partially contained on pages."""

    stamps_start_page = np.floor(stamps[:, 0] / page_size)
    stamps_end_page = np.floor(stamps[:, 1] / page_size)
    useful_idx = np.where(
        np.isin(stamps_start_page, pages_indices) | np.isin(stamps_end_page, pages_indices)
    )[0]
    pages_list = stamps[useful_idx, :]

    # pages_list = []
    # for i in range(stamps.shape[0]):
    #     stamp_start_page = stamps[i, 0] // page_size
    #     stamp_end_page = stamps[i, 1] // page_size
    #
    #     start_inside = (stamp_start_page in pages_indices)
    #     end_inside = (stamp_end_page in pages_indices)
    #
    #     if start_inside or end_inside:
    #         pages_list.append(stamps[i, :])

    pages_data = pages_list  # np.stack(pages_list, axis=0)
    return pages_data


def simple_split_with_list(x, y, train_fraction=0.8, seed=None):
    """Splits data stored in a list.

    The data x and y are list of arrays with shape [batch, ...].
    These are split in two sets randomly using train_fraction over the number of
    element of the list. Then these sets are returned with
    the arrays concatenated along the first dimension
    """
    n_subjects = len(x)
    n_train = int(n_subjects * train_fraction)
    print('Split: Total %d -- Training %d' % (n_subjects, n_train))
    random_idx = np.random.RandomState(seed=seed).permutation(n_subjects)
    train_idx = random_idx[:n_train]
    test_idx = random_idx[n_train:]
    x_train = np.concatenate([x[i] for i in train_idx], axis=0)
    y_train = np.concatenate([y[i] for i in train_idx], axis=0)
    x_test = np.concatenate([x[i] for i in test_idx], axis=0)
    y_test = np.concatenate([y[i] for i in test_idx], axis=0)
    return x_train, y_train, x_test, y_test


def split_ids_list(subject_ids, train_fraction=0.75, seed=None, verbose=True):
    """Splits the subject_ids list randomly using train_fraction."""
    n_subjects = len(subject_ids)
    n_train = int(n_subjects * train_fraction)
    if verbose:
        print('Split IDs: Total %d -- Training %d' % (n_subjects, n_train))
    random_idx = np.random.RandomState(seed=seed).permutation(n_subjects)
    train_idx = random_idx[:n_train]
    test_idx = random_idx[n_train:]
    train_ids = [subject_ids[i] for i in train_idx]
    test_ids = [subject_ids[i] for i in test_idx]
    return train_ids, test_ids


def split_ids_list_v2(subject_ids, split_id, train_fraction=0.75, verbose=False):
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


def shuffle_data(x, y, seed=None):
    """Shuffles data assuming that they are numpy arrays."""
    n_examples = x.shape[0]
    random_idx = np.random.RandomState(seed=seed).permutation(n_examples)
    x = x[random_idx]
    y = y[random_idx]
    return x, y


def shuffle_data_collection(list_of_arrays, seed=None):
    """Shuffles data assuming that they are numpy arrays."""
    n_examples = list_of_arrays[0].shape[0]
    random_idx = np.random.RandomState(seed=seed).permutation(n_examples)
    for i in range(len(list_of_arrays)):
        list_of_arrays[i] = list_of_arrays[i][random_idx]
    return list_of_arrays


def shuffle_data_with_ids(x, y, sub_ids, seed=None):
    """Shuffles data assuming that they are numpy arrays."""
    n_examples = x.shape[0]
    random_idx = np.random.RandomState(seed=seed).permutation(n_examples)
    x = x[random_idx]
    y = y[random_idx]
    sub_ids = sub_ids[random_idx]
    return x, y, sub_ids


def stamp_intersects_set(single_stamp, set_stamps):
    """ Assumes shape (2, 1) for single stamp and (N, 2) for set_stamps"""
    candidates = np.where(
        (set_stamps[:, 0] <= single_stamp[1])
        & (set_stamps[:, 1] >= single_stamp[0]))[0]
    for j in candidates:
        intersection = min(
            single_stamp[1], set_stamps[j, 1]
        ) - max(
            single_stamp[0], set_stamps[j, 0]
        ) + 1
        if intersection > 0:
            return True
    return False


def filter_stamps(stamps, start_sample, end_sample, return_idx=False):
    useful_idx = np.where(
        (stamps[:, 1] >= start_sample) & (stamps[:, 0] <= end_sample))[0]
    useful_stamps = stamps[useful_idx, :]

    if return_idx:
        return_tuple = (useful_stamps, useful_idx)
    else:
        return_tuple = useful_stamps
    return return_tuple


def get_overlap_matrix(stamps_1, stamps_2):
    # Matrix of overlap, rows are events, columns are detections
    n_det = stamps_2.shape[0]
    n_gs = stamps_1.shape[0]
    overlaps = np.zeros((n_gs, n_det))
    for i in range(n_gs):
        candidates = np.where(
            (stamps_2[:, 0] <= stamps_1[i, 1])
            & (stamps_2[:, 1] >= stamps_1[i, 0]))[0]
        for j in candidates:
            intersection = min(
                stamps_1[i, 1], stamps_2[j, 1]
            ) - max(
                stamps_1[i, 0], stamps_2[j, 0]
            ) + 1
            if intersection > 0:
                overlaps[i, j] = 1
    return overlaps


def overlapping_groups(overlap_matrix):
    groups_overlap = [[0]]
    for i in range(overlap_matrix.shape[0]):
        visited = np.any([i in single_group for single_group in groups_overlap])
        if not visited:
            # Check if intersects with an existent group
            added = False
            for single_group in groups_overlap:
                is_overlapping = np.any(overlap_matrix[i, single_group])
                if is_overlapping:
                    single_group.append(i)
                    added = True
                    break
            if not added:
                # Then variable is a new group
                groups_overlap.append([i])

    # Sort groups
    for single_group in groups_overlap:
        single_group.sort()

    return groups_overlap


def apply_lowpass(signal, fs, cutoff, filter_duration_ref=6, wave_expansion_factor=0.5):
    numtaps = fs * filter_duration_ref / (cutoff ** wave_expansion_factor)
    numtaps = int(2 * (numtaps // 2) + 1)  # ensure odd numtaps
    lp_kernel = firwin(numtaps, cutoff=cutoff, window="hann", fs=fs).astype(np.float32)
    lp_kernel /= lp_kernel.sum()
    new_signal = filter_fir(lp_kernel, signal)
    return new_signal


def apply_highpass(signal, fs, cutoff, filter_duration_ref=6, wave_expansion_factor=0.5):
    numtaps = fs * filter_duration_ref / (cutoff ** wave_expansion_factor)
    numtaps = int(2 * (numtaps // 2) + 1)  # ensure odd numtaps
    lp_kernel = firwin(numtaps, cutoff=cutoff, window="hann", fs=fs).astype(np.float32)
    lp_kernel /= lp_kernel.sum()
    # HP = delta - LP
    hp_kernel = -lp_kernel
    hp_kernel[numtaps//2] += 1
    new_signal = filter_fir(hp_kernel, signal)
    return new_signal


def apply_bandpass(signal, fs, lowcut, highcut, filter_duration_ref=6, wave_expansion_factor=0.5):
    new_signal = signal
    if lowcut is not None:
        new_signal = apply_highpass(
            new_signal, fs, lowcut, filter_duration_ref, wave_expansion_factor)
    if highcut is not None:
        new_signal = apply_lowpass(
            new_signal, fs, highcut, filter_duration_ref, wave_expansion_factor)
    return new_signal


def broad_filter_moda(x, fs, lowcut=0.3, highcut=30, filter_order=10):
    """Returns filtered signal sampled at fs Hz, with a 0.3-30 Hz
    bandpass."""
    print("Applying MODA bandpass filter")
    sos = butter(filter_order, lowcut, btype='highpass', fs=fs, output='sos')
    x = sosfiltfilt(sos, x)
    sos = butter(filter_order, highcut, btype='lowpass', fs=fs, output='sos')
    x = sosfiltfilt(sos, x)
    return x


def compute_pagewise_fft(x, fs, window_duration=2):
    # Input x is [n_pages, n_samples]
    n_pages, n_samples = x.shape
    window_size = int(fs * window_duration)
    x = x.reshape(n_pages, -1, window_size)
    window_shape = np.hanning(window_size).reshape(1, 1, -1)
    x = x * window_shape
    y = np.fft.rfft(x, axis=2) / window_size
    y = np.abs(y).mean(axis=1)
    f = np.fft.rfftfreq(window_size, d=1. / fs)
    return f, y


def compute_pagewise_powerlaw(f, y, broad_band=(2, 30), sigma_band=(10, 17)):
    # Input y is [n_pages, n_freqs]
    locs_to_use = np.where(
        ((f >= broad_band[0]) & (f < sigma_band[0])) | ((f > sigma_band[1]) & (f <= broad_band[1]))
    )[0]
    x_data = f[locs_to_use]
    y_data = y[:, locs_to_use]
    log_x = np.log(x_data)
    log_y = np.log(y_data)
    scale_l = []
    exponent_l = []
    for page_log_y in log_y:
        polycoefs = np.polynomial.Polynomial.fit(log_x, page_log_y, deg=1).convert().coef
        scale = np.exp(polycoefs[0])
        exponent = polycoefs[1]
        # power = scale * (freq ** exponent)
        scale_l.append(scale)
        exponent_l.append(exponent)
    scale_l = np.array(scale_l)
    exponent_l = np.array(exponent_l)
    return scale_l, exponent_l
