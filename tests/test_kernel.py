from orthosvm.kernels import hermite, chebyshev, gegenbauer
import numpy as np
import pytest


# Import datasets and true results
fourclass = np.loadtxt("tests/datasets/fourclass.csv", delimiter=",", skiprows=1)
X = fourclass[:, :2]


def load_and_strip(path):
    dataset = np.loadtxt(path, delimiter=",", skiprows=1)
    return dataset[..., 1:]


# Create a list of values and expected ones
common_path = "tests/datasets/"
expected_results = [
    (
        chebyshev.kernel,
        load_and_strip(common_path + "chebyshev_gramian_fourclass.csv"),
        3,
    ),
    (hermite.kernel, load_and_strip(common_path + "hermite_gramian_fourclass.csv"), 6),
]


@pytest.mark.parametrize("kernel, true_matrix, degree", expected_results)
def test_kernel_cpp(kernel, true_matrix, degree):

    X_gram = np.zeros((X.shape[0], X.shape[0]))

    for l, x in enumerate(X):
        for m, z in enumerate(X):
            summ, mult = 1.0, 1.0
            for i, k in zip(x, z):
                summ = 1.0
                if i != 0.0 and k != 0.0:
                    summ = kernel(i, k, degree)
                mult *= summ
            X_gram[l, m] = X_gram[m, l] = mult
            if m > l:
                break

    matrix_difference = X_gram - true_matrix
    print(np.where(matrix_difference.min() == matrix_difference))
    print("Maximum difference")
    print(X_gram[401, 847])
    print(true_matrix[401, 847])
    print("Minimum difference")
    print(X_gram[236, 847])
    print(true_matrix[236, 847])

    assert pytest.approx(X_gram == true_matrix, rel=1e-15)


# * Testing Gegenbauer polynomials specifically
expected_results = [
    (
        gegenbauer.kernel,
        load_and_strip(common_path + "gegenbauer_gramian_fourclass.csv"),
        6,
        -0.42,
    ),
]


@pytest.mark.parametrize("kernel, true_matrix, degree, alfa", expected_results)
def test_kernel_ggb(kernel, true_matrix, degree, alfa):

    X_gram = np.zeros((X.shape[0], X.shape[0]))

    for l, x in enumerate(X):
        for m, z in enumerate(X):
            summ, mult = 1.0, 1.0
            for i, k in zip(x, z):
                summ = 1.0
                if i != 0.0 and k != 0.0:
                    summ = kernel(i, k, degree, alfa)
                mult *= summ
            X_gram[l, m] = X_gram[m, l] = mult
            if m > l:
                break

    matrix_difference = X_gram - true_matrix
    print(np.where(matrix_difference.min() == matrix_difference))
    print("Maximum difference")
    print(X_gram[401, 847])
    print(true_matrix[401, 847])
    print("Minimum difference")
    print(X_gram[236, 847])
    print(true_matrix[236, 847])

    assert pytest.approx(X_gram == true_matrix, rel=1e-15)
