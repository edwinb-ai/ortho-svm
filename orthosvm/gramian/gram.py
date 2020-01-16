import numpy as np
from sklearn.utils import check_X_y


class Gram:
    def __init__(self, X, **kwargs):
        self.x = X

        if "y" in kwargs:
            self.y = kwargs["y"]
            # Check the validity of the arrays
            self._check_arrays()
        if "kernel" in kwargs:
            self.kernel_type = kwargs["kernel"]
        if "degree" in kwargs:
            self.degree = kwargs["degree"]
        if "alpha" in kwargs:
            self.alpha = kwargs["alpha"]
        if "gamma" in kwargs:
            self.gamma = kwargs["gamma"]

    def _check_arrays(self):
        self.x, self.y = check_X_y(self.x, self.y)

    # TODO: Put the computation of the gram matrix here!
    def kernel(self):
        pass
