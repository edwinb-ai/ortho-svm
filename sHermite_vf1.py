import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets.samples_generator import make_circles
from padierna_modules.plots import plot_svc_decision_function


# SELECCIÓN DE PROCESAMIENTO
# *********************************************
analizarFourclass = True
graficarPolinomios = True
calcularGram = True

# POLINOMIO HERMITE
# *******************************************
def H(x_i, n):

    if n == 0:
        return 1.0

    if n == 1:
        return x_i

    return x_i * H(x_i, n - 1) - (n - 1) * H(x_i, n - 2)


# MOSTRANDO POLINOMIOS DE S-HERMITE PARA VALIDAR LA FUNCIÓN H(x_i,n) ESCALADA
# *********************************************
if graficarPolinomios:
    plt.figure()
    t = np.arange(-1, 1.1, 0.1)  # Rango de prueba (test)
    for i in range(1, 6):
        plt.plot(t, hermite(t, i) * 2 ** (-i), label="Grado " + str(i))
        # plt.plot(t, H(t, i) * 2 ** (-i), label="Grado " + str(i))
        plt.legend()
        plt.title("Polinomios Ortogonales de s-Hermite")

# KERNEL S-HERMITE:
# ********************************************
# FALTA OPTIMIZAR EL CICLO PARA APROVECHAR SIMETRÍA DE MATRIZ
# PUEDE OMITIRSE VALIDACIÓN DE VECTORES SPARSE.
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


# SELECCIÓN DEL DATASET
# *********************************************
if analizarFourclass:
    fourclass = pd.read_csv("fourclass1.csv")
    X, Y = (fourclass.iloc[:, 0:2]), fourclass["Y"]
    # PARÁMETROS ÓPTIMOS (Del artículo sobre Gegenbauer)
    # *********************************************
    C_rbf, gamma_rbf = 30.42, 3.82  # RBF    - Fourclass
    C_sH, degree_sH = 25.20, 6  # s-Herm - Fourclass
else:
    X, Y = make_circles(100, factor=0.3, noise=0.1)
    # PARÁMETROS DE PRUEBA
    # *********************************************
    C_rbf, gamma_rbf = 100, 0.1  # RBF     - Dataset X
    C_sH, degree_sH = 100, 2  # s-Herm - Dataset X

# VERIFICANDO MATRIZ GRAMIANA DE S-HERM
# *********************************************
X = np.array(X)
if calcularGram:
    # X_gram = sHerm_kernel(X, Y, degree=degree_sH)
    X_gram = sHerm_kernel(X, Y, degree=degree_sH)
    print(X_gram)
    print("\n****VERIFICANDO MATRIZ GRAM*****")
    print(type(X_gram))
    print("Gram Max =", X_gram.max(), "Gram min =", X_gram.min())
    NANs = np.argwhere(np.isnan(X_gram))
    print("Valores tipo NAN: ", NANs)

# # ENTRENANDO Y GRAFICANDO MSV CON RBF y S-HERM.
plt.figure()
clf = SVC(kernel="rbf", C=C_rbf, gamma=gamma_rbf).fit(X, Y)
clf2 = SVC(kernel=sHerm_kernel, C=C_sH, degree=degree_sH).fit(X, Y)
plt.subplot(1, 2, 1)
plt.title(
    "MSV-RBF: C="
    + str(C_rbf)
    + ",gamma="
    + str(gamma_rbf)
    + ",SVs="
    + str(len(clf.support_))
)
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap="cool")
plot_svc_decision_function(clf, X, plot_support=True)
plt.subplot(1, 2, 2)
plt.title(
    "MSV-sHerm: C="
    + str(C_sH)
    + " n="
    + str(degree_sH)
    + ",SVs="
    + str(len(clf2.support_))
)
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap="cool")
clf2.support_vectors_ = X[np.array(clf2.support_), :]
plot_svc_decision_function(clf2, X, plot_support=True, customKernel=True)
print("\n*************************************************************")
print("RESULTADOS DE MODELOS RBF Y S-HERM")
print("***************************************************************")
print(
    "Vectores Soporte (VS) RBF:\t" + str(len(clf.support_)),
    "\ts-Herm: ",
    str(len(clf2.support_)),
)
print(
    "PSV: {0} {1}".format(
        len(clf.support_) * 100.0 / X.shape[0], len(clf2.support_) * 100.0 / X.shape[0]
    )
)
print("VS por Clase RBF:\t\t" + str(clf.n_support_), "\ts-Herm: ", str(clf2.n_support_))
print("Indices VS RBF:\t\t\t" + str(clf.support_), "\ts-Herm: ", str(clf2.support_))

