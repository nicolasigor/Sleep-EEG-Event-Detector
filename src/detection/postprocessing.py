from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.signal import find_peaks

from src.data.utils import filter_iir_lowpass, apply_bandpass


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


def get_amplitude_spindle(x, fs, distance_in_seconds=0.04):
    no_peaks_found = False

    distance = int(fs * distance_in_seconds)
    peaks_max, _ = find_peaks(x, distance=distance)
    peaks_min, _ = find_peaks(-x, distance=distance)
    if len(peaks_max) == 0 or len(peaks_min) == 0:
        print("Second attempt to find peaks")
        # First try to fix
        distance = distance // 2
        peaks_max, _ = find_peaks(x, distance=distance)
        peaks_min, _ = find_peaks(-x, distance=distance)
        if len(peaks_max) == 0 or len(peaks_min) == 0:
            print("Third attempt to find peaks")
            # Second try to fix
            distance = distance // 2
            peaks_max, _ = find_peaks(x, distance=distance)
            peaks_min, _ = find_peaks(-x, distance=distance)
            if len(peaks_max) == 0 or len(peaks_min) == 0:
                print("SKIPPED: Segment without peaks. Found %d peaks max and %d peaks min" % (
                        len(peaks_max), len(peaks_min)))
                no_peaks_found = True
    if no_peaks_found:
        max_pp = 1e6
    else:
        peaks = np.sort(np.concatenate([peaks_max, peaks_min]))
        peak_values = x[peaks]
        peak_to_peak_diff = np.abs(np.diff(peak_values))
        max_pp = np.max(peak_to_peak_diff)
    return max_pp


def spindle_amplitude_filtering(signal, stamps, fs, max_amplitude, lowcut=9.5, highcut=16.5):
    filt_signal = apply_bandpass(signal, fs, lowcut=lowcut, highcut=highcut)
    signal_events = [filt_signal[e[0]:e[1] + 1] for e in stamps]

    amplitudes = []
    for i in range(len(signal_events)):
        s = signal_events[i]
        amp = get_amplitude_spindle(s, fs)
        if amp > 1e5:
            print("Anomaly mark is", stamps[i])
        amplitudes.append(amp)
    amplitudes = np.array(amplitudes)

    if np.any(amplitudes > 1e5):
        no_peaks_found = True
    else:
        no_peaks_found = False

    valid_locs = np.where(amplitudes <= max_amplitude)[0]
    return stamps[valid_locs], no_peaks_found
