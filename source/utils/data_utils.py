import numpy as np
from collections import Counter


def undersample(X, y):
    classes = Counter(y)
    n_0 = classes[0]
    n_1 = classes[1]
    idxs_0 = np.where(y == 0)[0]
    idxs_1 = np.where(y == 1)[0]

    if n_0 < n_1:
        y_0 = y[idxs_0]
        X_0 = X[idxs_0]
        y_1 = y[idxs_1[: len(idxs_0)]]
        X_1 = X[idxs_1[: len(idxs_0)]]
        y = np.concatenate([y_0, y_1])
        X = np.concatenate([X_0, X_1])
    elif n_1 < n_0:
        y_0 = y[idxs_0[: len(idxs_1)]]
        X_0 = X[idxs_0[: len(idxs_1)]]
        y_1 = y[idxs_1[idxs_0]]
        X_1 = X[idxs_1[idxs_0]]
        y = np.concatenate([y_0, y_1])
        X = np.concatenate([X_0, X_1])

    # shuffle
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]
    return X, y
