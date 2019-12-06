from orthosvm.kernels.hermite import kernel
from orthosvm.kernels.shermite import sHerm_kernel
import numpy as np
import pytest


# Import datasets and true results
fourclass = np.loadtxt("tests/datasets/fourclass1.csv", delimiter=",", skiprows=1)
X = fourclass[:, :2]
true_grammian_fourclass = np.loadtxt(
    "tests/datasets/hermite_gramian_fourclass.csv", delimiter=",", skiprows=1
)
# Strip the first column from the matrix
true_gram_matrix = true_grammian_fourclass[..., 1:]


def test_kernel_cpp():
    X_gram = np.zeros((X.shape[0], X.shape[0]))
    for l, x in enumerate(X):
        for m, z in enumerate(X):
            summ, mult, i, j = 0.0, 1.0, 0, 0
            # Skip the upper triangular to avoid computing the same values twice
            if l > m:
                continue
            # Computer hermite kernel for the grammian matrix
            while i < x.size and j < z.size:
                if i == j:
                    summ = 1.0
                    summ += kernel(x[i], z[j], 6)
                    mult *= summ
                    i += 1
                    j += 1
                else:
                    if i > j:
                        j += 1
                    else:
                        i += 1
            X_gram[l, m] = mult
        # Complete the matrix with upper triangular
        X_gram = np.triu(X_gram) + np.triu(X_gram, 1).T

    assert pytest.approx(X_gram, rel=1e-1) == true_gram_matrix


def test_shermite_kernel():
    result = sHerm_kernel(X, degree=6)

    assert pytest.approx(result, rel=1e-1) == true_gram_matrix
