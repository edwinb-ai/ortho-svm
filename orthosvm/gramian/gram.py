import numpy as np
from sklearn.utils import check_X_y
from orthosvm.kernels import hermite, gegenbauer


class Gram:
    def __init__(self, X, **kwargs):
        self.x = X
        self.y = None
        # self.kernel_type = None
        # self.degree = None
        # self.alpha = None
        # self.gamma = None

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

    @property
    def gram(self):
        return self._compute_gram_matrix(self.x, y=self.y)

    def _check_arrays(self):
        self.x, self.y = check_X_y(self.x, self.y)

    def _give_kernel(self, x, z):
        if self.kernel_type == "hermite":
            return hermite.kernel(x, z, self.degree)
        if self.kernel_type == "gegenbauer":
            return gegenbauer.kernel(x, z, self.degree, self.alpha)

    def _iterate_over_arrays(self, X, y, gram):
        for l, x in enumerate(X):
            for m, z in enumerate(y):
                summ, mult = 1.0, 1.0
                for i, k in zip(x, z):
                    summ = 1.0
                    if i != 0.0 and k != 0.0:
                        summ = self._give_kernel(i, k)
                    mult *= summ
                    # * The matrix will be symmetric
                    if y is not None:
                        gram[l, m] = gram[m, l] = mult
                        if m > l:
                            break
                    # * The matrix won't be symmetric
                    else:
                        gram[l, m] = mult
        return np.copy(gram)

    def _compute_gram_matrix(self, X, y=None):
        if y is not None:
            X_gram = np.zeros((X.shape[0], y.shape[0]))
            X_gram = self._iterate_over_arrays(X, y, X_gram)
        else:
            X_gram = np.zeros((X.shape[0], X.shape[0]))
            X_gram = self._iterate_over_arrays(X, X, X_gram)

        return X_gram

    def gram_matrix(self):
        return self._compute_gram_matrix
