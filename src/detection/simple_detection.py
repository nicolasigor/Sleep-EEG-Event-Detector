import numpy as np

from src.data import utils, stamp_correction


def find_envelope(x, win_size):
    half_win_size = win_size // 2
    shifts = np.arange(-half_win_size, half_win_size + 0.001).astype(np.int32)
    shifted_signals = []
    for shift in shifts:
        shifted_signals.append(np.roll(x, shift))
    shifted_signals = np.stack(shifted_signals, axis=1)
    envelope = np.max(np.abs(shifted_signals), axis=1)
    return envelope


def get_sigma_envelope(x, fs, lowcut=11, highcut=16, win_duration=0.1):
    signal_sigma = utils.apply_bandpass(x, fs, lowcut, highcut)
    win_size = int(fs * win_duration)
    signal_sigma_env = find_envelope(signal_sigma, win_size)
    return signal_sigma_env


def simple_detector_from_envelope(
        signal_sigma_env,
        fs,
        amplitude_high_thr,
        amplitude_low_thr_factor=4/9,
        min_separation=0.3,
        min_duration_low=0.5,
        min_duration_high=0.3,
        max_duration=3.0,
):
    amplitude_low_thr = amplitude_high_thr * amplitude_low_thr_factor

    feat_high = (signal_sigma_env >= amplitude_high_thr).astype(np.int32)
    feat_low = (signal_sigma_env >= amplitude_low_thr).astype(np.int32)
    feat = feat_high + feat_low
    events = utils.seq2stamp(feat_low)  # Candidates

    # Group candidate events closer than 0.3s (to remove small fluctuations)
    # and remove events shorter than 0.5s (so trivially meets criteria of duration in lower amplitude)
    events = stamp_correction.combine_close_stamps(
        events, fs, min_separation=min_separation)
    events = stamp_correction.filter_duration_stamps(
        events, fs, min_duration=min_duration_low, max_duration=None)

    # Criteria of higher amplitude
    min_duration_high = min_duration_high * fs
    new_events = []
    for e in events:
        data = feat[e[0]:e[1] + 1]
        data_in_2 = np.sum(data == 2)
        if data_in_2 >= min_duration_high:
            new_events.append(e)
    if len(new_events) == 0:
        events = np.zeros((0, 2)).astype(np.int32)
    else:
        events = np.stack(new_events, axis=0)

    # Now remove events that are too long
    events = stamp_correction.filter_duration_stamps(
        events, fs, min_duration=min_duration_low, max_duration=max_duration)

    return events


def simple_detector_absolute(
        x,
        fs,
        amplitude_high_thr,
        amplitude_low_thr_factor=4/9,
        min_separation=0.3,
        min_duration_low=0.5,
        min_duration_high=0.3,
        max_duration=3.0,
        lowcut=11, highcut=16, win_duration=0.1
):
    """Detection using absolute amplitudes"""
    signal_sigma_env = get_sigma_envelope(x, fs, lowcut=lowcut, highcut=highcut, win_duration=win_duration)

    # detect
    events = simple_detector_from_envelope(
        signal_sigma_env,
        fs,
        amplitude_high_thr,
        amplitude_low_thr_factor=amplitude_low_thr_factor,
        min_separation=min_separation,
        min_duration_low=min_duration_low,
        min_duration_high=min_duration_high,
        max_duration=max_duration)
    return events


def simple_detector_relative(
        x,
        fs,
        amplitude_high_factor,
        pages_to_compute_baseline,
        page_duration,
        amplitude_low_thr_factor=4/9,
        min_separation=0.3,
        min_duration_low=0.5,
        min_duration_high=0.3,
        max_duration=3.0,
        lowcut=11, highcut=16, win_duration=0.1
):
    """Detection using amplitudes relative to baseline sigma activity"""
    signal_sigma_env = get_sigma_envelope(x, fs, lowcut=lowcut, highcut=highcut, win_duration=win_duration)

    # compute baseline
    page_size = int(page_duration * fs)
    env_in_n2 = signal_sigma_env.reshape(-1, page_size)[pages_to_compute_baseline]
    mean_sigma_env = np.median(env_in_n2)
    # compute relative high thr
    amplitude_high_thr = amplitude_high_factor * mean_sigma_env

    # detect
    events = simple_detector_from_envelope(
        signal_sigma_env,
        fs,
        amplitude_high_thr,
        amplitude_low_thr_factor=amplitude_low_thr_factor,
        min_separation=min_separation,
        min_duration_low=min_duration_low,
        min_duration_high=min_duration_high,
        max_duration=max_duration)
    return events
