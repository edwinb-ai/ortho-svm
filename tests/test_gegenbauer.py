import pytest
from orthosvm.kernels import gegenbauer

pochhammer_values = [
    (5.0, 6, 151200.0),
    (1.0, 0, 1.0),
    (10.0, 3, 1320.0),
    (1.0, -1, 0.0),
    (0.0, 5, 0.0),
]

gegenbauer_values = [
    (5.0, 6, 2.0, 6702996.0),
    (15.0, 7, 1.0, 21724469880.0),
    (2.0, 0, 3.0, 1.0),
    (3.0, 1, 2.0, 12.0),
]

weights_values = [(1.0, 1.0, -0.4, 1.0), (1.0, 1.0, 0.5, 1.0)]


@pytest.mark.parametrize("x, n, expected", pochhammer_values)
def test_pochhammer(x, n, expected):
    assert gegenbauer.pochhammer(x, n) == expected


@pytest.mark.parametrize("x, n, a, expected", gegenbauer_values)
def test_gegenbauerc(x, n, a, expected):
    assert gegenbauer.gegenbauerc(x, n, a) == expected


@pytest.mark.parametrize("x, y, alfa, expected", weights_values)
def test_weights(x, y, alfa, expected):
    assert gegenbauer.weights(x, y, alfa) == expected
