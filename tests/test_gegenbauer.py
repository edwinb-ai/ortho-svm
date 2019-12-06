import pytest
from orthosvm.kernels import gegenbauer

pochhammer_values = [
    (5.0, 6, 151200.0),
    (1.0, 0, 1.0),
    (10.0, 3, 1320.0),
    (1.0, -1, 0.0),
    (0.0, 5, 0.0),
]


@pytest.mark.parametrize("x, n, expected", pochhammer_values)
def test_pochhammer(x, n, expected):
    assert gegenbauer.pochhammer(x, n) == expected
