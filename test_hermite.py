import pytest
from shermite import hermite


vals_list = [(5.0, 0, 1.0), (5.0, 1, 5.0), (5.0, 2, 24.0), (5.0, 3, 110.0)]


@pytest.mark.parametrize("x, n, true_val", vals_list)
def test_hermite_eval(x, n, true_val):
    assert hermite(x, n) == true_val
