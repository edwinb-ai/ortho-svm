from scipy.special import eval_hermitenorm
import timeit
import numpy as np
import cProfile
from sklearn.metrics.pairwise import pairwise_kernels
from numba import njit, jit
from joblib import Memory, Parallel, delayed


# cache_dir = "/var/tmp/"
# memory = Memory(cache_dir, verbose=0)


def H(x_i, n):

    if n == 0:
        return 1.0

    if n == 1:
        return x_i

    return x_i * H(x_i, n - 1) - (n - 1) * H(x_i, n - 2)


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


def special_hermite(x, z, n=6):

    prod_1 = hermite(x, n) * hermite(z, n)
    # prod_2 = np.exp(-0.5 * (x ** 2 + z ** 2))
    # result = prod_1 * prod_2 * 2.0 ** (- 2.0 * n)
    result = prod_1 / 2.0 ** (2.0 * n)

    return result


def sHerm_kernel(X, Y=None, degree=2):
    X_gram = np.zeros((X.shape[0], X.shape[0]))
    for l, x in enumerate(X):
        for m, z in enumerate(X):
            summ, mult, i, j = 0, 1, 0, 0
            xlen, zlen = x.size, z.size
            while i < xlen and j < zlen:
                if i == j:
                    summ = 1
                    for k in range(1, degree + 1, 1):
                        summ += H(x[i], k) * H(z[j], k) * (2 ** (-2 * k))
                        # summ += hermite(x[i], k) * hermite(z[j], k) * (2 ** (-2 * k))
                    mult *= summ
                    i += 1
                    j += 1
                else:
                    if i > j:
                        j += 1
                    else:
                        i += 1
            X_gram[l][m] = mult
    return np.array(X_gram)


# @memory.cache
def kernel_special_hermite(X, Y, degree=2):
    X_gram = np.zeros((X.shape[0], Y.shape[0]))
    for l, x in enumerate(X):
        for m, z in enumerate(Y):
            if l > m:
                continue
            mult = 1.0
            for j, u in zip(x, z):
                sum_res = 1.0 + sum(
                    special_hermite(j, u, n=k) for k in range(1, degree + 1)
                )
                mult *= sum_res
            X_gram[l, m] = mult
    
    X_gram = np.triu(X_gram) + np.triu(X_gram, 1).T
    
    return X_gram


# def tri_kernel_special_hermite(X, Y, degree=2):
#     X_gram = np.zeros_like(X)
#     idx_triu = np.triu_indices(X_gram.shape[0])
#     # res = 1.0 + sum(
#     #     special_hermite(X, X, n=k) for k in range(1, degree + 1)
#     # )
#     # print(res)
#     for l in idx_triu:
#         sum_res = 0.0
#         mult = 1.0
#         for j in X[l]:
#             sum_res = 1.0 + sum(
#                 special_hermite(j, j, n=k) for k in range(1, degree + 1)
#             )
#             mult *= sum_res
#         X_gram[l] = mult
#     return X_gram


if __name__ == "__main__":

    # rand_vals = np.random.randint(1, 20, size=1000)
    # cProfile.run("[hermite(i, 100) for i in rand_vals]")
    # cProfile.run("eval_hermitenorm(100, rand_vals)")

    # print(H(25.0, 10))
    # print(eval_hermitenorm(10, 25.0))

    test_matrix = np.random.randint(0, 100, size=(1000, 3))
    # test_matrix = np.array([[1.0, 2.0], [2.0, 1.0]])

    # mult = sum(special_hermite(test_matrix, i) for i in range(1, 4, 1))
    # mult_2 = sum(special_hermite(test_matrix.T, i) for i in range(1, 4, 1))
    # print(mult)
    # print(mult_2)
    # print(mult @ mult_2)
    # print(sHerm_kernel(test_matrix, test_matrix, degree=6))
    print(timeit.timeit("sHerm_kernel(test_matrix, test_matrix, degree=6)", number=1, globals=globals()))

    # print(pairwise_kernels(test_matrix, metric=special_hermite))

    print(timeit.timeit("kernel_special_hermite(test_matrix, test_matrix, degree=6)", number=1, globals=globals()))
    # print(kernel_special_hermite(test_matrix, test_matrix, degree=6))

    # print(tri_kernel_special_hermite(test_matrix, test_matrix, degree=6))

    # cProfile.run("kernel_special_hermite(test_matrix, test_matrix, degree=6)")
