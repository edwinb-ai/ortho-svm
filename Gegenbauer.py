import math
import numpy as np


def Pochhammer(a, n):
    if n == 0:
        return 1
    elif n < 0 or a == 0:
        return 0

    prod = 1

    for i in range(n):
        prod *= a + i

    return prod


def Cn(x, n, a):
    if n == 0:
        return 1.0
    else:
        if a == 0:
            if n == 1:
                return x
            else:
                return 2.0 * x * Cn(x, n - 1, 0) - Cn(x, n - 2, 0)
        else:
            if n == 1:
                return 2.0 * a * x
            else:
                poli = (
                    2.0 * (n - 1.0 + a) * x * Cn(x, n - 1, a)
                    - ((n - 2 + 2 * a) * Cn(x, n - 2, a))
                ) / float(n)
            return poli


def w(x, z, a):
    if -0.5 < a <= 0.5:
        return 1
    elif a > 0.5:
        return ((1 - x ** 2) * (1 - z ** 2)) ** (a - 0.5) + 0.1


def u(k, alpha, n):

    return Pochhammer(2 * alpha, k) / (Pochhammer(1, k) * math.sqrt(n + 1))


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

    print(Pochhammer(3, 4))
    print(Cn(5.0, 10, 2))
