import os
import pickle
import sys
import warnings
from collections import OrderedDict

import biosppy.signals.tools as st
import numpy as np
import wfdb
from biosppy.signals.ecg import correct_rpeaks, hamilton_segmenter
from hrv.classical import frequency_domain, time_domain
from scipy.signal import medfilt
from tqdm import tqdm

warnings.filterwarnings(action="ignore")

base_dir = "dataset"
fs = 100  # ECG sample frequency

hr_min = 20
hr_max = 300


def feature_extraction(recording, signal, labels):
    data = []
    for i in tqdm(range(len(labels)), desc=recording, file=sys.stdout):
        segment = signal[i * fs * 60:(i + 1) * fs * 60]
        segment, _, _ = st.filter_signal(segment, ftype='FIR', band='bandpass', order=int(0.3 * fs), frequency=[3, 45],
                                         sampling_rate=fs)
        # Finding R peaks
        rpeaks, = hamilton_segmenter(segment, sampling_rate=fs)
        rpeaks, = correct_rpeaks(segment, rpeaks, sampling_rate=fs, tol=0.1)
        # Extracting feature
        label = 0 if labels[i] == "N" else 1
        if 40 <= len(rpeaks) <= 200:  # Remove abnormal R peaks
            rri_tm, rri = rpeaks[1:] / float(fs), np.diff(rpeaks, axis=-1) / float(fs)
            rri = medfilt(rri, kernel_size=3)
            edr_tm, edr = rpeaks / float(fs), segment[rpeaks]
            # Remove physiologically impossible HR signal
            if np.all(np.logical_and(60 / rri >= hr_min, 60 / rri <= hr_max)):
                rri_time_features, rri_frequency_features = time_domain(rri * 1000), frequency_domain(rri, rri_tm)
                edr_frequency_features = frequency_domain(edr, edr_tm)
                # 6 + 6 + 6 + 1 = 19
                data.append([
                    rri_time_features["rmssd"], rri_time_features["sdnn"], rri_time_features["nn50"],
                    rri_time_features["pnn50"], rri_time_features["mrri"], rri_time_features["mhr"],
                    rri_frequency_features["vlf"] / rri_frequency_features["total_power"],
                    rri_frequency_features["lf"] / rri_frequency_features["total_power"],
                    rri_frequency_features["hf"] / rri_frequency_features["total_power"],
                    rri_frequency_features["lf_hf"], rri_frequency_features["lfnu"], rri_frequency_features["hfnu"],
                    edr_frequency_features["vlf"] / edr_frequency_features["total_power"],
                    edr_frequency_features["lf"] / edr_frequency_features["total_power"],
                    edr_frequency_features["hf"] / edr_frequency_features["total_power"],
                    edr_frequency_features["lf_hf"], edr_frequency_features["lfnu"], edr_frequency_features["hfnu"],
                    label
                ])
            else:
                data.append([np.nan] * 18 + [label])
        else:
            data.append([np.nan] * 18 + [label])
    data = np.array(data, dtype="float")
    return data


if __name__ == "__main__":
    apnea_ecg = OrderedDict()

    # train data
    recordings = [
        "a01", "a02", "a03", "a04", "a05", "a06", "a07", "a08", "a09", "a10",
        "a11", "a12", "a13", "a14", "a15", "a16", "a17", "a18", "a19", "a20",
        "b01", "b02", "b03", "b04", "b05",
        "c01", "c02", "c03", "c04", "c05", "c06", "c07", "c08", "c09", "c10"
    ]
    for recording in recordings:
        signal = wfdb.rdrecord(os.path.join(base_dir, recording), channels=[0]).p_signal[:, 0]
        labels = wfdb.rdann(os.path.join(base_dir, recording), extension="apn").symbol
        apnea_ecg[recording] = feature_extraction(recording, signal, labels)

    print()

    # test data
    recordings = [
        "x01", "x02", "x03", "x04", "x05", "x06", "x07", "x08", "x09", "x10",
        "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20",
        "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29", "x30",
        "x31", "x32", "x33", "x34", "x35"
    ]
    answers = {}
    filename = os.path.join(base_dir, "event-2-answers")
    with open(filename, "r") as f:
        for answer in f.read().split("\n\n"):
            answers[answer[:3]] = list("".join(answer.split()[2::2]))
    for recording in recordings:
        signal = wfdb.rdrecord(os.path.join(base_dir, recording), channels=[0]).p_signal[:, 0]
        labels = answers[recording]
        apnea_ecg[recording] = feature_extraction(recording, signal, labels)

    with open(os.path.join(base_dir, "apnea-ecg.pkl"), "wb") as f:
        pickle.dump(apnea_ecg, f, protocol=2)

    print("ok")
