from orthosvm.gramian import gram
import numpy as np
import pytest


def test_gram_arrays():
    """Should pass the test because arrays are the same size
    """
    gram_matrix = gram.Gram(np.zeros((5, 5)), np.zeros(5), "linear")


def test_gram_arrays_fail():
    """When the arrays are of wrong size, raise an error
    """
    with pytest.raises(ValueError):
        gram_matrix = gram.Gram(np.zeros((5, 5)), np.zeros(10), "linear")
