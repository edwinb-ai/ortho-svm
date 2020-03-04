import numpy as np
from sklearn.metrics.pairwise import check_pairwise_arrays
from orthosvm.kernels import hermite, gegenbauer


def give_kernel(x, z, **kwargs):
    if kwargs["kernel"] == "hermite":
        return hermite.kernel(x, z, kwargs["degree"])
    if kwargs["kernel"] == "gegenbauer":
        return gegenbauer.kernel(x, z, kwargs["degree"], kwargs["alpha"])


def iterate_over_arrays(xdata, y, params):
    xgram = np.zeros((xdata.shape[0], y.shape[0]))
    for j, x in enumerate(xdata):
        for m, z in enumerate(y):
            summ, mult = 1.0, 1.0
            for i, k in zip(x, z):
                summ = 1.0
                if i != 0.0 and k != 0.0:
                    summ = give_kernel(i, k, **params)
                mult *= summ
                # * The matrix will be symmetric
                if xdata is y:
                    xgram[j, m] = xgram[m, j] = mult
                    if m > j:
                        break
                # * The matrix won't be symmetric
                else:
                    xgram[j, m] = mult
    return xgram


def gram_matrix(**kwargs):
    def compute_gram_matrix(xdata, y=None):
        xdata, y = check_pairwise_arrays(xdata, y)
        xgram = iterate_over_arrays(xdata, y, kwargs)

        return xgram

    return compute_gram_matrix
