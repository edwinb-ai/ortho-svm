from orthosvm.gramian import gram
import numpy as np
import pytest
from orthosvm.kernels import hermite
from orthosvm.gramian import compute_gram_matrix


def test_gram_arrays():
    """Should pass the test because arrays are the same size
    """
    gram_matrix = gram.Gram(np.zeros((5, 5)), np.zeros(5), "linear")


def test_gram_arrays_fail():
    """When the arrays are of wrong size, raise an error
    """
    with pytest.raises(ValueError):
        gram_matrix = gram.Gram(np.zeros((5, 5)), np.zeros(10), "linear")


def load_and_strip(path):
    dataset = np.loadtxt(path, delimiter=",", skiprows=1)
    return dataset[..., 1:]


def test_decorator():
    fourclass = np.loadtxt("tests/datasets/fourclass.csv", delimiter=",", skiprows=1)
    X = fourclass[:, :2]

    common_path = "tests/datasets/"
    expected_results = load_and_strip(common_path + "hermite_gramian_fourclass.csv")

    assert pytest.approx(
        compute_gram_matrix(X, kernel=hermite.kernel, degree=6) == expected_results,
        rel=1e-15,
    )
