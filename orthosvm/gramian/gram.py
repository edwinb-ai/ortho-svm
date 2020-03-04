import numpy as np
from sklearn.metrics.pairwise import check_pairwise_arrays
from orthosvm.kernels import hermite, gegenbauer, chebyshev


def give_kernel(x: float, z: float, **kwargs):
    """Compute a specific Mercer kernel for `x` and `z`.

    Using the keyword arguments one can specify the type of kernel,
    the degree and the special parameters, if any. This will return
    the evaluation of the Mercer kernel for `x` and `z`.

    Args:
        x (float): First value to evaluate the kernel.
        z (float): Second value to evaluate the kernel.
        **kwargs: The following keyword arguments are expected:
            kernel (str): One of "hermite", "chebyshev" or "gegenbauer".
            degree (int): Degree of the different polynomials.
            alpha (float): This is only useful for the "gegenbauer" kernel.

    Returns:
        float: The evaluation of the kernel with the given parameters.
    """
    # Split the variables for easier handling
    kernel = kwargs["kernel"]
    degree = kwargs["degree"]

    if kernel == "hermite":
        return hermite.kernel(x, z, degree)

    if kernel == "gegenbauer":
        alpha = kwargs["alpha"]
        
        # When alfa is 0 use the Chebyshev polynomials definition
        # because the Gegenbauer will always reduce to 0.0
        if alpha == 0.0:
            return chebyshev.kernel(x, z, degree)
        else:
            # Every other case can be handled by the general formulation
            return gegenbauer.kernel(x, z, degree, alpha)

    if kernel == "chebyshev":
        return chebyshev.kernel(x, z, degree)


def iterate_over_arrays(xdata, y, params):
    xgram = np.zeros((xdata.shape[0], y.shape[0]))
    for j, x in enumerate(xdata):
        for m, z in enumerate(y):
            summ, mult = 1.0, 1.0
            for i, k in zip(x, z):
                summ = 1.0
                if i != 0.0 and k != 0.0:
                    summ = give_kernel(i, k, **params)
                mult *= summ
                # * The matrix will be symmetric
                if xdata is y:
                    xgram[j, m] = xgram[m, j] = mult
                    if m > j:
                        break
                # * The matrix won't be symmetric
                else:
                    xgram[j, m] = mult
    return xgram


def gram_matrix(**kwargs):
    def compute_gram_matrix(xdata, y=None):
        xdata, y = check_pairwise_arrays(xdata, y)
        xgram = iterate_over_arrays(xdata, y, kwargs)

        return xgram

    return compute_gram_matrix
