import numpy as np
from sklearn.metrics.pairwise import check_pairwise_arrays
from orthosvm.kernels import hermite, gegenbauer, chebyshev
from typing import Optional, Callable


def give_kernel(x: float, z: float, **kwargs) -> float:
    """Compute a specific Mercer kernel for `x` and `z`.

    Using the keyword arguments one can specify the type of kernel,
    the degree and the special parameters, if any. This will return
    the evaluation of the Mercer kernel for `x` and `z`.

    Args:
        x: First value to evaluate the kernel.
        z: Second value to evaluate the kernel.
        **kwargs: The following keyword arguments are expected:
            kernel (str): One of "hermite", "chebyshev" or "gegenbauer".
            degree (int): Degree of the different polynomials.
            alpha (float): This is only useful for the "gegenbauer" kernel.

    Returns:
        The evaluation of the kernel with the given parameters.
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


def iterate_over_arrays(xdata: np.array, y: np.array, params: dict) -> np.array:
    """Given arrays `xdata` and `y` build a Grammian matrix with a specfici
    Mercer kernel.

    This function handles the creation of a Grammian matrix for either training
    or prediction for an arbitrary custom kernel. When in the training stage,
    the Grammian y expected to be symmetric but in the prediction stage the matrix
    will not be symmetric; both cases are handled in this function.
    
    Arguments:
        xdata {np.array} -- Data containing the number of observations and features.
        y {np.array} -- The labels (classification) or predicting values (regression).
        params {dict} -- Multiple parameters, mostly useful for the custom kernels that
            require special parameters.
    
    Returns:
        np.array -- An array containing the Grammian matrix.
    """
    # Pre-allocate space for the Grammian matrix
    xgram = np.zeros((xdata.shape[0], y.shape[0]))
    # Create allocating variables to hold multiplication and sum values
    summ: float = 1.0
    mult: float = 1.0

    # Loop over the indices and elements, both are required
    for j, x in enumerate(xdata):
        for m, z in enumerate(y):
            # Reset the variables
            summ = 1.0
            mult = 1.0
            # Loop over the elements in both xdata and y
            for i, k in zip(x, z):
                # When the matrices are not sparse, compute the required kernel
                if i != 0.0 and k != 0.0:
                    summ = give_kernel(i, k, **params)
                mult *= summ
                # If this is the case, where are in the training stage,
                # the matrix is symmetric
                if xdata is y:
                    xgram[j, m] = xgram[m, j] = mult
                # This handles the case when we are in the prediction stage,
                # the matrix is not symmetric
                else:
                    xgram[j, m] = mult

    return xgram


def gram_matrix(**kwargs) -> Callable:
    """Return a callable that computes the Grammian matrix.

    Return a callable that then computes the Grammian matrix, as required by the
    scikit-learn API. The API requires a closure that returns a callable so that
    then it can use it in both the training and prediction stages.

    Args:
        **kwargs (dict): These are special parameters required by the custom kernels,
            e.g. the alpha parameter for the Gegenbauer kernel. This is valid syntax
            for the scikit-learn API as well.
    
    Returns:
        Callable: The function that computes the Grammian matrix.
    """
    # The labels are actually not required, so we always set the as Optional
    def compute_gram_matrix(xdata: np.array, y: Optional[np.array] = None) -> np.array:
        # This makes it so that both arrays are copies of each other, for consistency
        xdata, y = check_pairwise_arrays(xdata, y)
        # This computes the actual Grammian matrix, notice that we pass the special
        # parameters here
        xgram = iterate_over_arrays(xdata, y, kwargs)

        return xgram

    return compute_gram_matrix
