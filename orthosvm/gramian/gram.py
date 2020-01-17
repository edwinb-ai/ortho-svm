import numpy as np
from sklearn.metrics.pairwise import check_pairwise_arrays
from orthosvm.kernels import hermite, gegenbauer


def give_kernel(x, z, **kwargs):
    if kwargs["kernel"] == "hermite":
        return hermite.kernel(x, z, kwargs["degree"])
    if kwargs["kernel"] == "gegenbauer":
        return gegenbauer.kernel(x, z, kwargs["degree"], kwargs["alpha"])


def iterate_over_arrays(X, y, params):
    X_gram = np.zeros((X.shape[0], y.shape[0]))
    for l, x in enumerate(X):
        for m, z in enumerate(y):
            summ, mult = 1.0, 1.0
            for i in range(len(x)):
                summ = 1.0
                if x[i] != 0.0 and z[i] != 0.0:
                    summ = give_kernel(x[i], z[i], **params)
                mult *= summ
                # * The matrix will be symmetric
                if X is y:
                    X_gram[l, m] = X_gram[m, l] = mult
                    if m > l:
                        break
                # * The matrix won't be symmetric
                else:
                    X_gram[l, m] = mult
    return X_gram


def gram_matrix(**kwargs):
    def compute_gram_matrix(X, y=None):
        X, y = check_pairwise_arrays(X, y)
        if y is not None:
            X_gram = iterate_over_arrays(X, y, kwargs)
        else:
            X_gram = iterate_over_arrays(X, X, kwargs)

        return X_gram

    return compute_gram_matrix
