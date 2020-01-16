import numpy as np
from sklearn.utils import check_X_y


class Gram:
    def __init__(self, x_data, y_data, name):
        self.x = x_data
        self.y = y_data
        self.kernel_type = name
        # Check the validity of the arrays
        self._check_arrays()

    def _check_arrays(self):
        self.x, self.y = check_X_y(self.x, self.y)

    def kernel(self):
        pass
