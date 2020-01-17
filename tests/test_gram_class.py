from orthosvm.gramian import gram
import numpy as np
import pytest
from orthosvm.gramian import compute_gram_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def load_and_strip(path):
    dataset = np.loadtxt(path, delimiter=",", skiprows=1)
    return dataset[..., 1:]


fourclass = np.loadtxt("tests/datasets/fourclass.csv", delimiter=",", skiprows=1)
X = fourclass[:, :2]
y = fourclass[:, 2]

common_path = "tests/datasets/"
expected_results = load_and_strip(common_path + "hermite_gramian_fourclass.csv")

gram_matrix = gram.gram_matrix(kernel="hermite", degree=6)


def test_gram_object_attributes():
    assert pytest.approx(gram_matrix(X) == expected_results, rel=1e-15)


def test_sklearn_integration():
    params = {"C": 25.20, "kernel": gram_matrix}
    svc = SVC(**params)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    svc.fit(X_train, y_train)
    accuracy = svc.score(X_test, y_test)
    print(type(accuracy))

    assert 0.81 == pytest.approx(accuracy, abs=1e-2)
