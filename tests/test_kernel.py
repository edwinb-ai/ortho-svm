from orthosvm.kernels.hermite import kernel
from orthosvm.kernels.shermite import sHerm_kernel
import numpy as np
from time import time


fourclass = np.loadtxt("datasets/fourclass1.csv", delimiter=",", skiprows=1)
X = fourclass[:, :2]


def test_kernel_cpp():
    start = time()
    X_gram = np.zeros((X.shape[0], X.shape[0]))
    for l, x in enumerate(X):
        for m, z in enumerate(X):
            summ, mult, i, j = 0.0, 1.0, 0, 0
            # Saltarse la triangular inferior
            if l > m:
                continue
            # Computar la matriz con kernel Hermite
            while i < x.size and j < z.size:
                if i == j:
                    summ = 1.0
                    summ += kernel(x[i], z[j], 6)
                    mult *= summ
                    i += 1
                    j += 1
                else:
                    if i > j:
                        j += 1
                    else:
                        i += 1
            X_gram[l, m] = mult
        # Completar la matriz simétrica con la parte superior de la matriz triangular
        X_gram = np.triu(X_gram) + np.triu(X_gram, 1).T
    stop = time()

    print(X_gram, stop - start)


def test_shermite_kernel():
    start = time()
    result = sHerm_kernel(X, degree=6)
    stop = time()
    print(result, stop - start)
