import numpy as np


def compute_gram_matrix(X, **kwargs):
    if "y" in kwargs:
        X_gram = np.zeros((X.shape[0], kwargs["y"].shape[0]))
    else:
        X_gram = np.zeros((X.shape[0], X.shape[0]))

    for l, x in enumerate(X):
        for m, z in enumerate(X):
            summ, mult = 1.0, 1.0
            for i, k in zip(x, z):
                summ = 1.0
                if i != 0.0 and k != 0.0:
                    summ = kwargs["kernel"](i, k, kwargs["degree"])
                mult *= summ
            X_gram[l, m] = X_gram[m, l] = mult
            if m > l:
                break
    return X_gram
