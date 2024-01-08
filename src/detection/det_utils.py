import numpy as np
from src.data import utils


def transform_predicted_proba_to_adjusted_proba(predicted_proba, optimal_threshold, eps=1e-8):
    """
    Adjusts probability vector so that
    adjusted_proba > 0.5 is equivalent to predicted_proba > optimal_threshold

    :param predicted_proba: vector of predicted probabilities.
    :param optimal_threshold: optimal threshold for class assignment in predicted probabilities.
    :param eps: for numerical stability. Defaults to 1e-8.
    :return: the vector of adjusted probabilities.
    """

    # Edge cases:
    if optimal_threshold == 0:
        # Then everything is above or at the threshold
        # We simulate this by simply mapping 0 - 1 to 0.5 - 1
        adjusted_proba = 0.5 * predicted_proba + 0.5
    elif optimal_threshold == 1:
        # Then everything is below or at the threshold
        # We simulate this by simply mapping 0-1 to 0-0.5
        adjusted_proba = 0.5 * predicted_proba
    else:
        # Prepare
        original_dtype = predicted_proba.dtype
        predicted_proba = predicted_proba.astype(np.float64)
        predicted_proba = np.clip(predicted_proba, a_min=eps, a_max=(1.0 - eps))
        # Compute
        logit_proba = np.log(predicted_proba / (1.0 - predicted_proba))
        bias_from_thr = -np.log(optimal_threshold / (1.0 - optimal_threshold))
        new_logit_proba = logit_proba + bias_from_thr
        adjusted_proba = 1.0 / (1.0 + np.exp(-new_logit_proba))
        # Go back to original dtype
        adjusted_proba = adjusted_proba.astype(original_dtype)
    return adjusted_proba


def transform_thr_for_adjusted_to_thr_for_predicted(thr_for_adjusted, optimal_threshold):
    """
    Returns a threshold that can be applied to the predicted probabilities so that
    predicted_proba > thr_for_predicted is equivalent to adjusted_proba > thr_for_adjusted

    :param thr_for_adjusted: threshold for class assignment in adjusted probabilities.
    :param optimal_threshold: optimal threshold for class assignment in predicted probabilities.
    :return: the equivalent threshold for class assignment in predicted probabilities
    """
    num = thr_for_adjusted * optimal_threshold
    den = thr_for_adjusted * optimal_threshold + (1.0 - thr_for_adjusted) * (1.0 - optimal_threshold)
    thr_for_predicted = num / den
    return thr_for_predicted


def get_event_probabilities(marks, probability, downsampling_factor=8, proba_prc=75):
    probability_upsampled = np.repeat(probability, downsampling_factor)
    # Retrieve segments of probabilities
    marks_segments = [probability_upsampled[m[0]:(m[1] + 1)] for m in marks]
    marks_proba = [np.percentile(m_seg, proba_prc) for m_seg in marks_segments]
    marks_proba = np.array(marks_proba)
    return marks_proba


