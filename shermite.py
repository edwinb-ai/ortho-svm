import numpy as np
from numba import njit
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from padierna_modules.plots import plot_svc_decision_function
import matplotlib.pyplot as plt
from multiprocessing import Pool
from itertools import repeat
from sklearn.model_selection import StratifiedKFold


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


# @njit
# def special_hermite(x, z, n):

#     prod_1 = hermite(x, n) * hermite(z, n)
#     # prod_2 = np.exp(-0.5 * (x ** 2 + z ** 2))
#     # result = prod_1 * prod_2 / 2.0 ** (2.0 * n)
#     # result = prod_1

#     return prod_1


# @memory.cache
def kernel_special_hermite(X, degree=2):

    X_gram = np.zeros((X.shape[0], X.shape[0]))

    for l, x in enumerate(X):
        for m, z in enumerate(X):
            # Aprovechar la simetría de la matriz
            if l > m:
                continue
            mult = 1.0
            sum_res = 1.0
            for j, u in zip(x, z):
                sum_res += sum(
                    hermite(j, n) * hermite(u, n) / (2.0 ** (2.0 * degree))
                    for n in range(1, degree + 1)
                )
            mult *= sum_res
            X_gram[l, m] = mult
    # Completar la matriz simétrica con la parte superior de la matriz triangular
    X_gram = np.triu(X_gram) + np.triu(X_gram, 1).T

    return X_gram


fourclass = np.genfromtxt("fourclass1.csv", delimiter=",", skip_header=1)
X, y = fourclass[:, 0:2], fourclass[:, 2]
# Escalar a -1 y 1
# X = MinMaxScaler(feature_range=(-1.0, 1.0)).fit_transform(X)
# PARÁMETROS ÓPTIMOS (Del artículo sobre Gegenbauer)
# *********************************************
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
# dict_hermite = {"C": C_sH, "kernel": "precomputed"}

# svc_hermite = SVC(**dict_hermite)

# rscv = StratifiedKFold(n_splits=10)

# # def train_model(params):
# def train_model(model, x, y):

#     x = np.array(x)
#     y = np.array(y)
#     result = []
#     for i, j in zip(x, y):
#         gram = kernel_special_hermite(i, degree=degree_sH)
#         vectors = model.fit(gram, j).support_
#         result.append(len(vectors))

#     return result


# svc_1 = []
# x_training = 0
# y_training = []

# for train_idx, _ in rscv.split(X, y):

#     x_training = len(X[train_idx])
#     X_gram = kernel_special_hermite(X[train_idx], degree=degree_sH)
#     svc_1.append(len(svc_hermite.fit(X_gram, y[train_idx]).support_))

# # x_split = np.split(np.array(x_training), 5)
# # y_split = np.split(np.array(y_training), 5)

# # with Pool(4) as pool:

# #     svc_1.append(
# #         pool.map_async(
# #             train_model,
# #             zip(
# #                 repeat(svc_hermite),
# #                 [i for i in x_split[:4]],
# #                 [i for i in y_split[:4]],
# #             ),
# #         ).get()
# #     )

# # svc_1 = np.array(svc_1).ravel()
# # svc_1 = np.append(svc_1, train_model((svc_hermite, x_split[-1], y_split[-1])))
# psv = np.array(svc_1) * 100.0 / x_training
# print(psv.mean(), psv.std())
