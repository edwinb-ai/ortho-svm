from orthosvm.gramian import gram
import numpy as np
import pytest
from orthosvm.gramian import compute_gram_matrix
from sklearn.svm import SVC


def test_gram_arrays():
    """Should pass the test because arrays are the same size
    """
    gram_matrix = gram.Gram(np.zeros((5, 5)), y=np.zeros(5), kernel="linear")


def test_gram_arrays_fail():
    """When the arrays are of wrong size, raise an error
    """
    with pytest.raises(ValueError):
        gram_matrix = gram.Gram(np.zeros((5, 5)), y=np.zeros(10), kernel="linear")


def load_and_strip(path):
    dataset = np.loadtxt(path, delimiter=",", skiprows=1)
    return dataset[..., 1:]


fourclass = np.loadtxt("tests/datasets/fourclass.csv", delimiter=",", skiprows=1)
X = fourclass[:, :2]

common_path = "tests/datasets/"
expected_results = load_and_strip(common_path + "hermite_gramian_fourclass.csv")

gram_matrix = gram.Gram(X, kernel="hermite", degree=6)


def test_gram_object_attributes():
    assert pytest.approx(gram_matrix.gram == expected_results, rel=1e-15)


def test_sklearn_integration():
    gram_matrix = gram.Gram(X, kernel="hermite", degree=6)
    svc = SVC(kernel=gram_matrix.gram_matrix())
