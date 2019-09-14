from scipy.special import eval_hermitenorm
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from numba import njit
from joblib import Memory, Parallel, delayed
from padierna_modules.plots import plot_svc_decision_function
import matplotlib.pyplot as plt
from sklearn.svm import SVC


cache_dir = "/tmp/"
memory = Memory(cache_dir, verbose=0)


@njit
def hermite(x, n):

    primer_valor = 1.0
    segundo_valor = x

    if n == 0:
        return primer_valor
    elif n == 1:
        return segundo_valor
    else:
        resultado = 0.0

        for i in range(1, n):
            resultado = x * segundo_valor - i * primer_valor
            primer_valor = segundo_valor
            segundo_valor = resultado

        return resultado


@njit
def special_hermite(x, z, n=6):

    prod_1 = hermite(x, n) * hermite(z, n)
    prod_2 = np.exp(-0.5 * (x ** 2 + z ** 2))
    result = prod_1 * prod_2 / 2.0 ** (2.0 * n)
    # result = prod_1 / 2.0 ** (2.0 * n)

    return result


@memory.cache
def kernel_special_hermite(X, y=None, degree=2):
    X_gram = np.zeros((X.shape[0], X.shape[0]))
    for l, x in enumerate(X):
        for m, z in enumerate(X):
            # Aprovechar la simetría de la matriz
            if l > m:
                continue
            mult = 1.0
            for j, u in zip(x, z):
                sum_res = 1.0 + sum(
                    special_hermite(j, u, n=k) for k in range(1, degree + 1)
                )
                mult *= sum_res
            X_gram[l, m] = mult
    # Completar la matriz simétrica con la parte superior de la matriz triangular
    X_gram = np.triu(X_gram) + np.triu(X_gram, 1).T

    if not y is None:
        result = np.dot(np.dot(X_gram, X), y.T)

        return result

    return X_gram


fourclass = np.genfromtxt("fourclass1.csv", delimiter=",", skip_header=1)
X, y = fourclass[:, 0:2], fourclass[:, 2]
# PARÁMETROS ÓPTIMOS (Del artículo sobre Gegenbauer)
# *********************************************
C_rbf, gamma_rbf = 30.42, 3.82  # RBF    - Fourclass
C_sH, degree_sH = 25.20, 6  # s-Herm - Fourclass
# VERIFICANDO MATRIZ GRAMIANA DE S-HERM
# *********************************************
X_gram = kernel_special_hermite(X, degree=degree_sH)
print(X_gram)
print("\n****VERIFICANDO MATRIZ GRAM*****")
print(type(X_gram))
print("Gram Max =", X_gram.max(), "Gram min =", X_gram.min())
NANs = np.argwhere(np.isnan(X_gram))
print("Valores tipo NAN: ", NANs)
# # ENTRENANDO MSV CON RBF y S-HERM.
list_dicts = [
    {"C": C_rbf, "kernel": "rbf", "gamma": gamma_rbf},
    {"C": C_sH, "kernel": kernel_special_hermite, "degree": degree_sH},
]
svc_lists = [SVC(**i).fit(X, y) for i in list_dicts]
for i in svc_lists:
    print(i.support_)
    # GRAFICANDO MSV CON RBF y S-HERM.
    i.support_vectors_ = X[i.support_, :]
    # plot_svc_decision_function(i, X, plot_support=True, customKernel=True)
    print("\n*************************************************************")
    print("RESULTADOS DE MODELOS RBF Y S-HERM")
    print("***************************************************************")
    print("Vectores Soporte (VS) RBF: {}".format(len(i.support_)))
    print("PSV: {}".format(len(i.support_) * 100.0 / X.shape[0]))
    print("VS por Clase RBF: {}".format(i.n_support_))
    print("Indices VS RBF: {}".format(i.support_))
