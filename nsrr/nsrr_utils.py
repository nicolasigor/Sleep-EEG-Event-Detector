import os
import xml.etree.ElementTree as ET

import numpy as np
import pyedflib

from src.data import utils


NSRR_PATH = os.path.abspath("/home/ntapia/Projects/Sleep_Databases/NSRR_Databases")

NSRR_DATA_PATHS = {
    'shhs1': {
        'edf': os.path.join(NSRR_PATH, "shhs/polysomnography/edfs/shhs1"),
        'annot': os.path.join(NSRR_PATH, "shhs/polysomnography/annotations-events-nsrr/shhs1"),
    },
    'mros1': {
        'edf': os.path.join(NSRR_PATH, "mros/polysomnography/edfs/visit1"),
        'annot': os.path.join(NSRR_PATH, "mros/polysomnography/annotations-events-nsrr/visit1")
    },
    'chat1': {
        'edf': os.path.join(NSRR_PATH, "chat/polysomnography/edfs/visit1"),
        'annot': os.path.join(NSRR_PATH, "chat/polysomnography/annotations-events-nsrr/visit1")
    },
    'ccshs': {
        'edf': os.path.join(NSRR_PATH, "ccshs/polysomnography/edfs"),
        'annot': os.path.join(NSRR_PATH, "ccshs/polysomnography/annotations-events-nsrr")
    },
    'cfs': {
        'edf': os.path.join(NSRR_PATH, "cfs/polysomnography/edfs"),
        'annot': os.path.join(NSRR_PATH, "cfs/polysomnography/annotations-events-nsrr")
    },
    'sof': {
        'edf': os.path.join(NSRR_PATH, "sof/polysomnography/edfs"),
        'annot': os.path.join(NSRR_PATH, "sof/polysomnography/annotations-events-nsrr")
    },
}

CHANNEL_PRIORITY_LABELS = [
    ("EEG",),  # C4-A1 in SHHS
    ("EEG(sec)",),  # C3-A2 in SHHS
    ("EEG2",),
    ("EEG 2",),
    ("EEG(SEC)",),
    ("EEG sec",),
    ("C3", "A2"),
    ("C3", "M2"),
    ("C3-A2",),
    ("C3-M2",),
    ("C4", "A1"),
    ("C4", "M1"),
    ("C4-A1",),
    ("C4-M1",),
]


def extract_id(fname, is_annotation):
    # Remove extension
    fname = ".".join(fname.split(".")[:-1])
    if is_annotation:
        # Remove last tag
        fname = "-".join(fname.split("-")[:-1])
    return fname


def prepare_paths(edf_folder, annot_folder):
    """
    Assuming annot_folder is the one in the NSRR format,
    nomenclature of files is
    edf: subjectid.edf
    xml: subjectid-nsrr.xml
    """

    edf_files = os.listdir(edf_folder)
    edf_files = [f for f in edf_files if '.edf' in f]

    annot_files = os.listdir(annot_folder)
    annot_files = [f for f in annot_files if '.xml' in f]

    edf_ids = [extract_id(fname, False) for fname in edf_files]
    annot_ids = [extract_id(fname, True) for fname in annot_files]

    # Keep only IDs with both files
    common_ids = list(set(edf_ids).intersection(set(annot_ids)))
    common_ids.sort()

    paths_dict = {}
    for single_id in common_ids:
        edf_loc = edf_ids.index(single_id)
        annot_loc = annot_ids.index(single_id)
        paths_dict[single_id] = {
            'edf': os.path.join(edf_folder, edf_files[edf_loc]),
            'annot': os.path.join(annot_folder, annot_files[annot_loc])
        }
    return paths_dict


def read_hypnogram(annot_path, verbose=False, assumed_epoch_length_if_missing=30):
    tree = ET.parse(annot_path)
    root = tree.getroot()
    scored_events = root.find('ScoredEvents')
    epoch_length_text = root.find("EpochLength").text
    if epoch_length_text is None:
        print("Missing epoch length, assuming %s [s]" % assumed_epoch_length_if_missing) if verbose else None
        epoch_length = assumed_epoch_length_if_missing
    else:
        epoch_length = float(epoch_length_text)
    # print(ET.tostring(root, encoding='utf8').decode('utf8'))
    stage_labels = []
    stage_stamps = []
    for event in scored_events:
        e_type = event.find("EventType").text
        if e_type == "Stages|Stages":
            stage_name = event.find("EventConcept").text
            stage_start = float(event.find("Start").text)
            stage_duration = float(event.find("Duration").text)
            # Normalize variable-length epoch to a number of fixed-length epochs
            n_epochs = int(stage_duration / epoch_length)
            for i in range(n_epochs):
                stage_stamps.append([stage_start + epoch_length * i, epoch_length])
                stage_labels.append(stage_name)
    stage_labels = np.array(stage_labels)
    stage_stamps = np.stack(stage_stamps, axis=0)
    idx_sorted = np.argsort(stage_stamps[:, 0])
    stage_labels = stage_labels[idx_sorted]
    stage_stamps = stage_stamps[idx_sorted, :]
    stage_start_times = stage_stamps[:, 0].astype(np.float32)
    return stage_labels, stage_start_times, epoch_length


def get_edf_info(edf_path):
    fs_list = []
    with pyedflib.EdfReader(edf_path) as file:
        channel_names = file.getSignalLabels()
        for chn in channel_names:
            channel_to_extract = channel_names.index(chn)
            fs = file.samplefrequency(channel_to_extract)
            fs_list.append(fs)
    return channel_names, fs_list


def read_signal_from_file(file, channel_name):
    units_to_factor_map = {
        'V': 1e6,
        'mV': 1e3,
        'uV': 1.0,
    }
    channel_names = file.getSignalLabels()
    channel_to_extract = channel_names.index(channel_name)

    signal = file.readSignal(channel_to_extract)
    units = file.getPhysicalDimension(channel_to_extract)
    factor = units_to_factor_map[units]
    signal = signal * factor
    fs = file.samplefrequency(channel_to_extract)
    return signal, fs


def read_edf_channel(edf_path, channel_priority_list):
    with pyedflib.EdfReader(edf_path) as file:
        channel_names = file.getSignalLabels()

        channel_found = None
        for chn_pair in channel_priority_list:
            if np.all([chn in channel_names for chn in chn_pair]):
                channel_found = chn_pair
                break
        if channel_found is None:
            return None

        signal, fs = read_signal_from_file(file, channel_found[0])
        if len(channel_found) == 2:
            signal_2, fs_2 = read_signal_from_file(file, channel_found[1])
            if fs != fs_2:
                return None
            signal = signal - signal_2
    return signal, fs, channel_found


def short_signal_to_n2(signal, hypnogram, epoch_samples, n2_name):
    """
    Returns a cropped signal where only N2 stages are returned, ensuring one page of real signal
    at each border. This means that some non-N2 stages are kept, but they are a small portion.
    """
    n2_pages = np.where(hypnogram == n2_name)[0]
    valid_pages = np.concatenate([n2_pages - 1, n2_pages, n2_pages + 1])
    valid_pages = np.clip(valid_pages, a_min=0, a_max=(hypnogram.size - 1))
    valid_pages = np.unique(valid_pages)  # it is ensured to have context at each side of n2 pages

    # Now simplify
    hypnogram = hypnogram[valid_pages]

    signal = utils.extract_pages(signal, valid_pages, epoch_samples)
    signal = signal.flatten()

    return signal, hypnogram
