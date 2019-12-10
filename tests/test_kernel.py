from orthosvm.kernels import hermite, chebyshev
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
    # (hermite.kernel, hermite_true_matrix, 6),
]


@pytest.mark.parametrize("kernel, true_matrix, degree", expected_results)
def test_kernel_cpp(kernel, true_matrix, degree):

    X_gram = np.zeros((X.shape[0], X.shape[0]))

    for l, x in enumerate(X):
        for m, z in enumerate(X):
            summ, mult, i, j = 1.0, 1.0, 0, 0
            # Skip the upper triangular to avoid computing the same values twice
            if l > m:
                continue
            # Computer hermite kernel for the grammian matrix
            while i < len(x) and j < len(z):
                if i == j:
                    # summ = 0.0
                    summ = kernel(x[i], z[j], degree)
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

    matrix_difference = X_gram - true_matrix
    print(np.where(matrix_difference.min() == matrix_difference))
    print("Maximum difference")
    print(X_gram[401, 847])
    print(true_matrix[401, 847])
    print("Minimum difference")
    print(X_gram[236, 847])
    print(true_matrix[236, 847])

    assert pytest.approx(X_gram, rel=1e-2) == true_matrix
