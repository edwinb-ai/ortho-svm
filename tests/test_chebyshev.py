from scipy.special import eval_chebyt
import pytest
from orthosvm.kernels import chebyshev


some_value = 0.5
cheby_values = [(some_value, n, eval_chebyt(n, some_value)) for n in range(11)]


@pytest.mark.parametrize("x, n, expected", cheby_values)
def test_values_chebyshev(x, n, expected):
    assert expected == chebyshev.chebyshev(x, n)
