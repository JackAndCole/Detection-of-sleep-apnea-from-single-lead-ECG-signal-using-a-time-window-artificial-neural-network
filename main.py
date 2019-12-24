import os
import pickle

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

np.random.seed(0)

base_dir = "dataset"

with open(os.path.join(base_dir, "apnea-ecg.pkl"), "rb") as f:
    apnea_ecg = pickle.load(f)

print(apnea_ecg.keys())


def shift(xs, n):
    e = np.empty_like(xs)
    if n > 0:
        e[:n] = np.nan
        e[n:] = xs[:-n]
    elif n < 0:
        e[n:] = np.nan
        e[:n] = xs[-n:]
    else:
        e[:] = xs[:]
    return e


def acquisition_features(recordings, time_window_size):
    features = []
    labels = []
    groups = []
    for recording in recordings:
        data = apnea_ecg[recording]
        temp = []
        for w in range(time_window_size + 1):
            temp.append(shift(data[:, :-1], w))
        temp = np.concatenate(temp, axis=1)
        mask = ~np.isnan(temp).any(axis=1)
        features.append(temp[mask])
        labels.append(data[mask, -1])
        groups.append([recording] * sum(mask))
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    groups = np.concatenate(groups, axis=0)
    return features, labels, groups


x_train, y_train, groups_train = acquisition_features(list(apnea_ecg.keys())[:35], 5)
x_test, y_test, groups_test = acquisition_features(list(apnea_ecg.keys())[35:], 5)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

clf = MLPClassifier(hidden_layer_sizes=(x_train.shape[1] * 2 + 1,), alpha=1, max_iter=1000)
clf.fit(x_train, y_train)

print(clf.score(x_test, y_test))

y_pred = clf.predict(x_test)
C = confusion_matrix(y_test, y_pred, labels=(1, 0))
TP, TN, FP, FN = C[0, 0], C[1, 1], C[1, 0], C[0, 1]
acc, sn, sp = 1. * (TP + TN) / (TP + TN + FP + FN), 1. * TP / (TP + FN), 1. * TN / (TN + FP)
print("acc: {}, sn: {}, sp: {}".format(acc, sn, sp))
