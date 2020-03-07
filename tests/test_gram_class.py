from orthosvm.gramian import gram
import numpy as np
import pytest
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


def test_gram_callable_from_callable():
    gram_matrix = gram.gram_matrix(kernel="hermite", degree=6)
    result = np.allclose(gram_matrix(X), expected_results)
    assert result


def test_gram_matrix_callable():
    params = dict(kernel="hermite", degree=6)
    result = np.allclose(gram.iterate_over_arrays(X, X, params), expected_results)
    assert pytest.approx(result, rel=1e-15)


def test_sklearn_integration_hermite():
    gram_matrix = gram.gram_matrix(kernel="hermite", degree=6)
    params = {"C": 25.2, "kernel": gram_matrix}
    svc = SVC(**params)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    svc.fit(X_train, y_train)
    accuracy = svc.score(X_test, y_test)
    print(accuracy)

    assert 0.8 == pytest.approx(accuracy, rel=1e-1)


def test_sklearn_integration_gegenbauer():
    gram_matrix = gram.gram_matrix(kernel="gegenbauer", degree=6, alpha=-0.42)
    params = {"C": 31.52, "kernel": gram_matrix}
    svc = SVC(**params)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    svc.fit(X_train, y_train)
    accuracy = svc.score(X_test, y_test)
    print(accuracy)

    assert 0.99 == pytest.approx(accuracy, rel=1e-1)
