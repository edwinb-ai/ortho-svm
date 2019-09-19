import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool
from itertools import repeat
from sklearn.model_selection import  RepeatedKFold


fourclass = np.genfromtxt("fourclass1.csv", delimiter=",", skip_header=1)
X, y = fourclass[:, 0:2], fourclass[:, 2]
# Escalar a -1 y 1
X = MinMaxScaler(feature_range=(-1.0, 1.0)).fit_transform(X)
# PARÁMETROS ÓPTIMOS (Del artículo sobre Gegenbauer)
# *********************************************
C_rbf, gamma_rbf = 30.42, 3.82  # RBF    - Fourclass
# # ENTRENANDO MSV CON RBF y S-HERM.
rbf_dict = {"C": C_rbf, "kernel": "rbf", "gamma": gamma_rbf}
svc_rbf = SVC(**rbf_dict)

rscv = RepeatedKFold(n_splits=10, n_repeats=35)

def train_model(params):

    model, x, y = params
    x = np.array(x)
    y = np.array(y)
    result = []
    for i, j in zip(x, y):
        vectors = model.fit(i, j).support_
        result.append(len(vectors))

    return result


svc_1 = []
x_training = []
y_training = []

for train_idx, _ in rscv.split(X):

    x_training.append(X[train_idx])
    y_training.append(y[train_idx])

x_split = np.split(np.array(x_training), 5)
y_split = np.split(np.array(y_training), 5)

with Pool(4) as pool:

    svc_1.append(
        pool.map_async(
            train_model,
            zip(
                repeat(svc_rbf),
                [i for i in x_split[:4]],
                [i for i in y_split[:4]],
            ),
        ).get()
    )

svc_1 = np.array(svc_1).ravel()
svc_1 = np.append(svc_1, train_model((svc_rbf, x_split[-1], y_split[-1])))
print(svc_1.shape)
print(len(x_training[0]))
psv = np.array(svc_1) * 100.0 / len(x_training[0])
print(psv.mean(), psv.std())
# print(psv)
