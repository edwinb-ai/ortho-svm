from orthosvm.kernels import hermite, chebyshev, gegenbauer
from orthosvm.kernels.shermite import sHerm_kernel
import numpy as np
import pytest


# Import datasets and true results
fourclass = np.loadtxt("tests/datasets/fourclass.csv", delimiter=",", skiprows=1)
X = fourclass[:, :2]
hermite_gramian_true = np.loadtxt(
    "tests/datasets/hermite_gramian_fourclass.csv", delimiter=",", skiprows=1
)
chebyshev_gramian_true = np.loadtxt(
    "tests/datasets/chebyshev_gramian_fourclass.csv", delimiter=",", skiprows=1
)
# Strip the first column from the matrix
hermite_true_matrix = hermite_gramian_true[..., 1:]
chebyshev_true_matrix = chebyshev_gramian_true[..., 1:]

# Create a list of values and expected ones
expected_results = [
    (chebyshev.kernel, chebyshev_true_matrix, 3),
    (hermite.kernel, hermite_true_matrix, 6),
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
