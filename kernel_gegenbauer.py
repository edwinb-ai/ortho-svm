import numpy as np
from numba import njit


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


def u_scaling(k, a, n):

    term_1 = 1.0 / np.sqrt(n + 1.0)
    term_2 = pochhammer(2.0 * a, k) / pochhammer(1.0, k)
    result = term_1 * term_2

    return result


def Kgeg(x, z, a, n):
    mult = 1
    for j in range(0, len(x)):
        suma = 0
        for i in range(1, n + 1):
            suma += Cn(x[j], i, a) * Cn(z[j], i, a) * w(x[j], z[j], a) * u(i, a, n) ** 2
        mult *= suma
    return mult


def KGEG_generator(alpha, degree):
    def KGEG(X, Z):
        matgram = np.empty((len(X), len(Z)))
        for i in range(len(X)):
            for j in range(i + 1):
                matgram[i, j] = Kgeg(X[i], Z[j], alpha, degree)
                matgram[j, i] = matgram[i, j]
        return matgram

    return KGEG


if __name__ == "__main__":

    print(pochhammer(3, 4))
    print(gegenbauerc(5.0, 10, 2))
    print(weights(2.0, 3.0, 1.5))
    print(u_scaling(4, 1.5, 4))
