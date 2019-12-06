import pytest
from .hermite_impls import hermite_recurrence, hermite_expr, hermite_iterative
from orthosvm.kernels import hermite
from time import time


vals_list = [(5.0, 0, 1.0), (5.0, 1, 5.0), (5.0, 2, 24.0), (5.0, 3, 110.0)]
timing_list = [(0.006, i) for i in range(41)]


@pytest.mark.parametrize("x, n, true_val", vals_list)
def test_hermite_recurrence_eval(x, n, true_val):
    assert hermite_recurrence(x, n) == true_val


@pytest.mark.parametrize("x, n, true_val", vals_list)
def test_hermite_cpp(x, n, true_val):
    assert hermite.hermite(x, n) == true_val


@pytest.mark.parametrize("x, n, true_val", vals_list)
def test_hermite_iterative_eval(x, n, true_val):
    assert hermite_iterative(x, n) == true_val


@pytest.mark.parametrize("x, n, true_val", vals_list)
def test_hermite_expressions(x, n, true_val):
    assert hermite_expr(x, n) == true_val


# Just use some of them to make testing faster
@pytest.mark.parametrize("x, n", timing_list[:31])
def test_time_recurrence(x, n):
    start = time()
    result = hermite_recurrence(x, n)
    stop = time()

    print("Time elapsed, recurrent: {}".format(stop - start))


@pytest.mark.parametrize("x, n", timing_list)
def test_time_expr(x, n):
    start = time()
    result = hermite_expr(x, n)
    stop = time()

    print("Time elapsed, expressions: {}".format(stop - start))


# @pytest.mark.parametrize("x, n", timing_list)
# def test_time_iterative(x, n):
#     start = time()
#     result = hermite_iterative(x, n)
#     stop = time()

#     print("Time elapsed, iterative: {}".format(stop - start))


@pytest.mark.parametrize("x, n", timing_list)
def test_time_cpp_iterative(x, n):
    start = time()
    result = hermite.hermite(x, n)
    stop = time()

    print("Time elapsed, C++: {}".format(stop - start))
