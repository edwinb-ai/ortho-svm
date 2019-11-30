from shermite import hermite
import pytest


expected_values = [(5.0, 0, 1.0), (5.0, 1, 5.0), (5.0, 2, 24.0), (5.0, 3, 110.0)]


@pytest.mark.parametrize("x, n, result", expected_values)
def test_values_degree(x, n, result):
    val = hermite(x, n)

    assert val == result
