import numpy as np
from numba import njit
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from multiprocessing import Pool
from itertools import repeat
from sklearn.model_selection import RepeatedKFold


def pochhammer(x, n):

    if n == 0:
        return 1.0

    elif n < 0 or x == 0:
        return 0.0

    result = np.prod([x + i for i in range(n)])

    return result


@njit
def gegenbauerc(x, n, a):

    first_value = 1.0
    second_value = 2.0 * a * x

    if n == 0:
        return first_value
    elif n == 1:
        return second_value
    else:
        result = 0.0

        for i in range(2, n + 1):
            result = 2.0 * x * (i + a - 1.0) * second_value - (
                (i + 2.0 * a - 2.0) * first_value
            )
            result /= i
            first_value = second_value
            second_value = result

        return result


@njit
def weights(x, z, a):
    if -0.5 < a <= 0.5:
        return 1.0

    elif a > 0.5:
        term_1 = (1.0 - x ** 2.0) * (1.0 - z ** 2.0)
        result = np.power(term_1, a - 0.5) + 0.1

        return result


def u_scaling(k, a):

    term_1 = 1.0 / np.sqrt(k + 1.0)
    term_2 = pochhammer(2.0 * a, k) / pochhammer(1.0, k)
    result = term_1 * term_2

    return result


def kernel_ggb(x, z, a, n):

    term_1 = gegenbauerc(x, n, a) * gegenbauerc(z, n, a)
    term_1 *= weights(x, z, a)
    term_1 *= u_scaling(n, a) ** 2.0

    return term_1


def ggb_gram(X, alpha, degree=2):
    X_gram = np.zeros((X.shape[0], X.shape[0]))
    for l, x in enumerate(X):
        for m, z in enumerate(X):
            # Aprovechar la simetría de la matriz
            if l > m:
                continue
            mult = 1.0
            for j, u in zip(x, z):
                sum_res = 1.0 + sum(
                    kernel_ggb(j, u, alpha, p) for p in range(1, degree + 1)
                )
                mult *= sum_res
            X_gram[l, m] = mult
    # Completar la matriz simétrica con la parte superior de la matriz triangular
    X_gram = np.triu(X_gram) + np.triu(X_gram, 1).T

    return X_gram


fourclass = np.genfromtxt("fourclass1.csv", delimiter=",", skip_header=1)
X, y = fourclass[:, 0:2], fourclass[:, 2]
# Escalar a -1 y 1
X = MinMaxScaler(feature_range=(-1.0, 1.0)).fit_transform(X)
X_gram = ggb_gram(X, -0.42, degree=6)
print(X_gram)
print(X_gram.shape)
print("\n****VERIFICANDO MATRIZ GRAM*****")
print(type(X_gram))
print("Gram Max =", X_gram.max(), "Gram min =", X_gram.min())
NANs = np.argwhere(np.isnan(X_gram))
print("Valores tipo NAN: ", NANs)

# dict_ggb = {"C": 31.52, "kernel": "precomputed"}

# svc_ggb = SVC(**dict_ggb)

# rscv = RepeatedKFold(n_splits=10, n_repeats=35)


# def train_model(params):

#     model, x, y = params
#     x = np.array(x)
#     y = np.array(y)
#     result = []
#     for i, j in zip(x, y):
#         gram = ggb_gram(i, -0.42, degree=6)
#         vectors = model.fit(gram, j).support_
#         result.append(len(vectors))

#     return result


# svc_1 = []
# x_training = []
# y_training = []

# for train_idx, _ in rscv.split(X):

#     x_training.append(X[train_idx])
#     y_training.append(y[train_idx])

# x_split = np.split(np.array(x_training), 5)
# y_split = np.split(np.array(y_training), 5)

# with Pool(4) as pool:

#     svc_1.append(
#         pool.map_async(
#             train_model,
#             zip(
#                 repeat(svc_ggb), [i for i in x_split[:4]], [i for i in y_split[:4]]
#             ),
#         ).get()
#     )

# svc_1 = np.array(svc_1).ravel()
# svc_1 = np.append(svc_1, train_model((svc_ggb, x_split[-1], y_split[-1])))
# psv = np.array(svc_1) * 100.0 / len(x_training[0])
# print(psv.mean(), psv.std())
